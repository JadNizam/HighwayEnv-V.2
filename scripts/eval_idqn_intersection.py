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
    duration=10,
    vehicles_count=15,
    screen_width=1200,
    screen_height=800,
    centering_y=0.5,
    scaling=5.5
):
    # mirror TRAINING config exactly, plus viewer settings
    return gym.make(
        "intersection-v1",
        render_mode="human" if render else None,
        config={
            "controlled_vehicles": n_agents,
            "duration": duration,
            "vehicles_count": vehicles_count,
            "spawn_probability": 0.35,
            "destination": None,  # random
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
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
                    "target_speeds": [0, 4.5, 9.0],
                },
            },
            "collision_reward": -10.0,
            "high_speed_reward": 0.4,
            "arrived_reward": 5.0,
            "reward_speed_range": [4.0, 9.5],
            "normalize_reward": True,
            "offroad_terminal": False,

            # -------- Viewer settings --------
            "screen_width": int(screen_width),
            "screen_height": int(screen_height),
            "centering_position": [0.5, float(centering_y)],
            "scaling": float(scaling),
        },
    )

# --- NEW: load checkpoints and detect expected input dims ---
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
        # infer expected input dim from first layer weight
        expected_in = sd["net.0.weight"].shape[1]
        net = QNet(expected_in, act_dims[i]).to(device)
        net.load_state_dict(sd, strict=True)
        net.eval()
        models.append(net)
        in_dims.append(expected_in)
        print(f"Loaded {ckpt} (expects in_dim={expected_in})")
    return models, in_dims

# --- NEW: fit obs to model input (crop or zero-pad) ---
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="saved_idqn")
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--vehicles", type=int, default=15)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS when rendering")
    parser.add_argument("--delay", type=float, default=None, help="Seconds per step; overrides --fps")
    # Viewer tuning
    parser.add_argument("--screen-width", type=int, default=1200)
    parser.add_argument("--screen-height", type=int, default=800)
    parser.add_argument("--center-y", type=float, default=0.5, help="0.0 bottom to 1.0 top; 0.5 centers")
    parser.add_argument("--scale", type=float, default=5.5, help="Zoom: larger is more zoomed in")

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
        scaling=args.scale
    )

    # match training obs extension
    env = SharedContextWrapper(env, k_nearest=3, box_xy=12.0, include_last_actions=True)

    # infer env dims (for info only)
    obs, info = env.reset(seed=args.seed)
    if len(obs) < args.n_agents:
        print(f"[Warn] Env returned {len(obs)} controlled agents but --n-agents={args.n_agents}.")
    assert isinstance(env.action_space, spaces.Tuple)
    env_obs_dims = [flatten_obs(o).shape[0] for o in obs]
    act_dims = [env.action_space.spaces[i].n for i in range(len(obs))]
    print(f"[Eval] env_obs_dims={env_obs_dims}, act_dims={act_dims}")

    # load models and get expected input dims from checkpoints
    models, model_in_dims = load_checkpoints(args.checkpoint_dir, len(obs), act_dims, device)
    print(f"[Eval] env_obs_dim={env_obs_dims[0]} | model_in_dim={model_in_dims[0]}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        ret = 0.0
        steps = 0
        while not (terminated or truncated):
            actions = []
            for i in range(len(obs)):
                a = greedy_action(models[i], flatten_obs(obs[i]), model_in_dims[i], device)
                actions.append(a)
            obs, reward, terminated, truncated, info = env.step(tuple(actions))

            if args.render:
                if args.delay is not None:
                    time.sleep(max(0.0, args.delay))
                else:
                    time.sleep(max(0.0, 1.0 / args.fps))

            if isinstance(reward, (list, tuple, np.ndarray)):
                ret += float(np.sum(reward))
            else:
                ret += float(reward)
            steps += 1
        print(f"Episode {ep} | steps={steps} | return={ret:.2f}")

    env.close()

if __name__ == "__main__":
    main()
