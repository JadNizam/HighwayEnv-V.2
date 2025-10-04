# scripts/eval_idqn_intersection.py
import os
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
from gymnasium import spaces
from scripts.shared_context_wrapper import SharedContextWrapper

# ---------- model must match training ----------
class QNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def flatten_obs(x):
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1)

def make_env(
    n_agents=8,
    render=False,
    duration=100,             # match training
    vehicles_count=8,         # match training
    screen_width=1200,
    screen_height=800,
    centering_y=0.5,
    scaling=5.0,              # slightly zoomed out
    fixed_cam=False,
    cam_x=0.0,
    cam_y=0.0
):
    return gym.make(
        "intersection-v1",
        render_mode="human" if render else None,
        config={
            "controlled_vehicles": n_agents,
            "duration": duration,
            "vehicles_count": vehicles_count,
            "spawn_probability": 0.15,
            "destination": None,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    "features": ["presence","x","y","vx","vy","cos_h","sin_h","cos_d","sin_d"],
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": True,
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 3.0, 6.0],
                },
            },
            # rewards (match training)
            "collision_reward": -12.0,
            "high_speed_reward": 0.4,
            "arrived_reward": 8.0,
            "reward_speed_range": [4.0, 9.5],
            "normalize_reward": False,
            "offroad_terminal": False,

            # viewer sizing
            "screen_width": int(screen_width),
            "screen_height": int(screen_height),
            "centering_position": [0.5, float(centering_y)],
            "scaling": float(scaling),

            # env-native camera flags (fine to keep)
            "fixed_camera": bool(fixed_cam),
            "camera_center": [float(cam_x), float(cam_y)],
            "camera_zoom": float(scaling) if fixed_cam else None,
        },
    )

# --- load checkpoints and detect expected input dims ---
def load_checkpoints(dir_path, n_agents, act_dims, device):
    models, in_dims = [], []
    for i in range(n_agents):
        best = os.path.join(dir_path, f"agent_{i}_best.pt")
        if os.path.isfile(best):
            ckpt = best
        else:
            eps = sorted(glob.glob(os.path.join(dir_path, f"agent_{i}_ep*.pt")))
            if not eps:
                raise FileNotFoundError(f"No checkpoint for agent {i} in {dir_path}")
            ckpt = eps[-1]
        sd = torch.load(ckpt, map_location=device)
        expected_in = sd["net.0.weight"].shape[1]
        net = QNet(expected_in, act_dims[i]).to(device)
        net.load_state_dict(sd, strict=True)
        net.eval()
        models.append(net)
        in_dims.append(expected_in)
        print(f"Loaded {ckpt} (expects in_dim={expected_in})")
    return models, in_dims

# --- fit obs to model input (crop or zero-pad) ---
def fit_obs(vec: np.ndarray, target: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = vec.shape[0]
    if n == target:
        return vec
    if n > target:
        return vec[:target]
    out = np.zeros(target, dtype=np.float32)
    out[:n] = vec
    return out

@torch.no_grad()
def greedy_action(net, obs_vec, expected_in_dim, device):
    x_np = fit_obs(obs_vec, expected_in_dim)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
    q = net(x)
    return int(q.argmax(dim=-1).item())

# -------- pin the viewer by monkey-patching its display() --------
def _install_display_patch(env, center_xy=(0.0, 0.0), zoom=None):
    """
    Patch viewer.display so every frame:
      - follow flags are disabled
      - tracked handles cleared
      - center and zoom forced to our values
    """
    try:
        u = env.unwrapped
        v = getattr(u, "viewer", None)
        if v is None:
            return False
        if getattr(v, "_fixed_cam_patched", False):
            return True

        orig_display = v.display

        def fixed_display(*args, **kwargs):
            # kill follow flags every frame
            for attr in ("follow_ego", "track_vehicle", "track_agent", "dynamic_display", "auto_zoom"):
                if hasattr(v, attr):
                    try: setattr(v, attr, False)
                    except Exception: pass
            # clear tracked handles
            for attr in ("vehicle_to_track", "ego_vehicle", "vehicle"):
                if hasattr(v, attr):
                    try: setattr(v, attr, None)
                    except Exception: pass
            # pin center & zoom
            center = np.array(center_xy, dtype=float)
            for attr in ("center", "world_center", "camera_center"):
                if hasattr(v, attr):
                    try: setattr(v, attr, center)
                    except Exception: pass
            if zoom is not None:
                for attr in ("scaling", "zoom"):
                    if hasattr(v, attr):
                        try: setattr(v, attr, float(zoom)); break
                        except Exception: pass
            # draw
            return orig_display(*args, **kwargs)

        v.display = fixed_display
        v._fixed_cam_patched = True
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="saved_idqn")
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--vehicles", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS when rendering")
    parser.add_argument("--delay", type=float, default=None, help="Seconds per step; overrides --fps")
    # Viewer tuning
    parser.add_argument("--screen-width", type=int, default=1200)
    parser.add_argument("--screen-height", type=int, default=800)
    parser.add_argument("--center-y", type=float, default=0.5, help="0.0 bottom to 1.0 top; 0.5 centers")
    parser.add_argument("--scale", type=float, default=5.0, help="Zoom: larger is more zoomed in")
    # Fixed camera options
    parser.add_argument("--fixed-cam", action="store_true", help="Lock view to intersection center")
    parser.add_argument("--cam-x", type=float, default=0.0, help="World X for camera center")
    parser.add_argument("--cam-y", type=float, default=0.0, help="World Y for camera center")

    args = parser.parse_args()

    device = torch.device("cpu")
    env = make_env(
        n_agents=args.n_agents,
        render=args.render,
        duration=args.duration,
        vehicles_count=args.vehicles,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        centering_y=args.center_y,
        scaling=args.scale,
        fixed_cam=args.fixed_cam,
        cam_x=args.cam_x,
        cam_y=args.cam_y,
    )

    if args.fixed_cam:
        print(f"[Camera] Forcing center=({args.cam_x}, {args.cam_y}), zoom={args.scale}")
    else:
        print("[Camera] Default viewer behavior (may follow ego).")

    # training feature extension
    env = SharedContextWrapper(env, k_nearest=3, box_xy=12.0, include_last_actions=True)

    # --- reset ---
    obs, info = env.reset(seed=args.seed)

    # Install display patch as soon as a viewer exists; if not yet, it'll get created on first display.
    if args.render and args.fixed_cam:
        # many builds create viewer the first time they display (which may happen in step),
        # so try now and then keep trying in the loop
        _install_display_patch(env, (args.cam_x, args.cam_y), zoom=args.scale)

    # infer dims (info)
    if len(obs) < args.n_agents:
        print(f"[Warn] Env returned {len(obs)} controlled agents but --n-agents={args.n_agents}.")
    assert isinstance(env.action_space, spaces.Tuple)
    env_obs_dims = [flatten_obs(o).shape[0] for o in obs]
    act_dims = [env.action_space.spaces[i].n for i in range(len(obs))]
    print(f"[Eval] env_obs_dims={env_obs_dims}, act_dims={act_dims}")

    # load models
    models, model_in_dims = load_checkpoints(args.checkpoint_dir, len(obs), act_dims, device)
    print(f"[Eval] env_obs_dim={env_obs_dims[0]} | model_in_dim={model_in_dims[0]}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)

        terminated = truncated = False
        ret = 0.0
        steps = 0
        while not (terminated or truncated):
            # keep trying to install the patch until viewer exists, then it sticks
            if args.render and args.fixed_cam:
                _install_display_patch(env, (args.cam_x, args.cam_y), zoom=args.scale)

            actions = [
                greedy_action(models[i], flatten_obs(obs[i]), model_in_dims[i], device)
                for i in range(len(obs))
            ]
            obs, reward, terminated, truncated, info = env.step(tuple(actions))

            if args.render:
                # some builds render inside step; patch again just in case
                if args.fixed_cam:
                    _install_display_patch(env, (args.cam_x, args.cam_y), zoom=args.scale)
                if args.delay is not None:
                    time.sleep(max(0.0, args.delay))
                else:
                    time.sleep(max(0.0, 1.0 / args.fps))

            ret += float(np.sum(reward)) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
            steps += 1

        print(f"Episode {ep} | steps={steps} | return={ret:.2f}")

    env.close()

if __name__ == "__main__":
    main()
