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

def make_env(n_agents=4, render=False, duration=10, vehicles_count=8):
    # mirror TRAINING config exactly
    return gym.make(
        "intersection-v1",
        render_mode="human" if render else None,
        config={
            "controlled_vehicles": n_agents,
            "duration": duration,
            "vehicles_count": vehicles_count,
            "spawn_probability": 0.35,
            "destination": None, #random
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 15,  # <-- important: 15 * 7 = 105 obs dim
                    "features": ["presence","x","y","vx","vy","cos_h","sin_h", "cos_d","sin_d"],
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": True,  # <-- important: adds cos_d,sin_d
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,          # <-- no lane-change; matches 3 actions below
                    "target_speeds": [0, 4.5, 9.0],  # <-- 3 actions
                },
            },
            "collision_reward": -10.0,
            "high_speed_reward": 0.4,
            "arrived_reward": 5.0,
            "reward_speed_range": [4.0, 9.5],
            "normalize_reward": True,
            "offroad_terminal": False,
        },
    )

def load_checkpoints(dir_path, n_agents, obs_dims, act_dims, device):
    models = []
    for i in range(n_agents):
        best = os.path.join(dir_path, f"agent_{i}_best.pt")
        if os.path.isfile(best):
            ckpt = best
        else:
            eps = sorted(glob.glob(os.path.join(dir_path, f"agent_{i}_ep*.pt")))
            if not eps:
                raise FileNotFoundError(f"No checkpoint for agent {i} in {dir_path}")
            ckpt = eps[-1]
        net = QNet(obs_dims[i], act_dims[i]).to(device)
        sd = torch.load(ckpt, map_location=device)
        net.load_state_dict(sd, strict=True)
        net.eval()
        models.append(net)
        print(f"Loaded {ckpt}")
    return models

@torch.no_grad()
def greedy_action(net, obs, device):
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = net(x)
    return int(q.argmax(dim=-1).item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="saved_idqn")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--vehicles", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS when rendering")
    parser.add_argument("--delay", type=float, default=None, help="Seconds per step; overrides --fps")
    args = parser.parse_args()

    device = torch.device("cpu")
    env = make_env(n_agents=args.n_agents, render=args.render,
                   duration=args.duration, vehicles_count=args.vehicles)

    # infer dims (should be 105 obs, 3 actions with the config above)
    obs, info = env.reset(seed=args.seed)
    assert isinstance(env.action_space, spaces.Tuple)
    obs_dims = [flatten_obs(o).shape[0] for o in obs]
    act_dims = [env.action_space.spaces[i].n for i in range(args.n_agents)]
    print(f"[Eval] obs_dims={obs_dims}, act_dims={act_dims}")

    models = load_checkpoints(args.checkpoint_dir, args.n_agents, obs_dims, act_dims, device)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        ret = 0.0
        steps = 0
        while not (terminated or truncated):
            actions = []
            for i in range(args.n_agents):
                a = greedy_action(models[i], flatten_obs(obs[i]), device)
                actions.append(a)
            obs, reward, terminated, truncated, info = env.step(tuple(actions))

            # throttle render speed
            if args.render:
                if args.delay is not None:
                    time.sleep(max(0.0, args.delay))
                else:
                    time.sleep(max(0.0, 1.0 / args.fps))

            # scalar or per-agent vector reward
            if isinstance(reward, (list, tuple, np.ndarray)):
                ret += float(np.sum(reward))
            else:
                ret += float(reward)
            steps += 1
        print(f"Episode {ep} | steps={steps} | return={ret:.2f}")

    env.close()

if __name__ == "__main__":
    main()
