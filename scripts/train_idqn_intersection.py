# scripts/train_idqn_intersection.py
import os
import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
import gymnasium as gym
import highway_env
from tqdm import trange

from scripts.shared_context_wrapper import SharedContextWrapper
from scripts.reward_multiagent import MultiAgentShapingWrapper  # reward script to assist w training

# ----------------------- utils -----------------------
def flatten_obs(x):
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1)  # 1D vector

Transition = namedtuple("Transition", "state action reward next_state done")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buf)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

# ----------------------- DQN -----------------------
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

class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99, device="cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma

        self.policy = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = ReplayBuffer(100_000)
        self.batch_size = 64

        # epsilon-greedy
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 80_000  # explore a bit longer (was 50k)
        self.steps_done = 0

    def epsilon(self):
        frac = min(1.0, self.steps_done / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-3.0 * frac)

    def select_action(self, obs):
        eps = self.epsilon()
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s)
            return int(q.argmax(dim=-1).item())

    def push(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = self.buffer.sample(self.batch_size)
        state  = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(-1)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_s = torch.as_tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        done   = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q = self.policy(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target(next_s).max(dim=1, keepdim=True)[0]
            target = reward + (1.0 - done) * self.gamma * next_q

        loss = nn.functional.mse_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optim.step()
        return float(loss.item())

    def soft_update(self, tau=0.01):
        with torch.no_grad():
            for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
                tp.data.mul_(1 - tau).add_(pp.data * tau)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

# ----------------------- env maker -----------------------
def make_env(n_agents=8, render=False, duration=100, vehicles_count=8):
    return gym.make(
        "intersection-v1",
        render_mode="human" if render else None,
        config={
            "controlled_vehicles": n_agents,
            "duration": duration,
            "vehicles_count": vehicles_count,
            "spawn_probability": 0.15,  # was 0.35 (lighter traffic while learning)
            "destination": None,  # random
            # --- observations/actions (multi-agent) ---
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 10,  # was 10 (kept smaller than 15 to speed up & simplify)
                    "features": ["presence","x","y","vx","vy","cos_h","sin_h","cos_d","sin_d"],
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": True
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 3.0, 6.0],  # was [0, 4.5, 9.0]
                },
            },
            # --- base reward knobs (env-level) ---
            "collision_reward": -12.0,
            "high_speed_reward": 0.4,
            "arrived_reward": 8.0,
            "reward_speed_range": [4.0, 9.5],
            "normalize_reward": False,  # was True; needed for reliable crash penalty logic
            "offroad_terminal": False,
        },
    )

def per_agent_action_space(env):
    assert isinstance(env.action_space, spaces.Tuple)
    return [sp.n for sp in env.action_space.spaces]

# ----------------------- training loop -----------------------
def train(seed=0, n_agents=8, episodes=20000, render=False, save_dir="saved_idqn"):
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # base env → wrap with shaping
    base_env = make_env(n_agents=n_agents, render=render, duration=100, vehicles_count=8)
    env = MultiAgentShapingWrapper(
        base_env,
        # flow/progress
        progress_alpha=0.02,
        step_beta=0.006,          # slightly stronger time cost
        block_gamma=0.03,         # stronger anti-blocking
        block_speed_thresh=0.5,
        box_xy=12.0,
        coop_bonus=0.2,
        arrival_threshold=4.0,
        reduce_to_scalar="mean",
        # safety shaping (stronger)
        danger_radius=10.0,
        near_k=0.10,
        close_k=0.04,
        ttc_thresh=4.0,
        ttc_k=0.12,
        team_collision_penalty=0.6,
        collision_threshold=-6.0
    )

    # add shared context features
    env = SharedContextWrapper(env, k_nearest=3, box_xy=12.0, include_last_actions=True)

    obs, info = env.reset(seed=seed)

    # per-agent dims
    obs_dim = [flatten_obs(o).shape[0] for o in obs]
    act_dim = per_agent_action_space(env)

    agents = [DQNAgent(obs_dim[i], act_dim[i], lr=1e-3, gamma=0.99, device=device)
              for i in range(n_agents)]

    global_step = 0
    best_return = -1e9
    episode_returns = []  # store total return per episode

    for ep in trange(episodes, desc="Training"):
        obs, info = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        ep_loss = 0.0
        steps = 0

        while not (terminated or truncated):
            # select actions
            actions = []
            for i in range(n_agents):
                oi = flatten_obs(obs[i])
                a  = agents[i].select_action(oi)
                actions.append(a)

            # take a step
            obs2, reward, terminated, truncated, info = env.step(tuple(actions))

            # per-agent rewards (wrapper fills info['agents_rewards'])
            if isinstance(info, dict) and "agents_rewards" in info:
                rew_vec = [float(r) for r in info["agents_rewards"]]
            elif isinstance(reward, (list, tuple, np.ndarray)):
                rew_vec = list(map(float, reward))
            else:
                rew_vec = [float(reward)] * n_agents

            done_flag = float(terminated or truncated)

            # store transitions and update each agent
            step_loss = 0.0
            for i in range(n_agents):
                si  = flatten_obs(obs[i])
                s2i = flatten_obs(obs2[i])
                agents[i].push(si, actions[i], rew_vec[i], s2i, done_flag)
                step_loss += agents[i].update()

            # soft update targets occasionally
            if global_step % 100 == 0:
                for ag in agents:
                    ag.soft_update(tau=0.01)

            ep_loss += step_loss
            ep_return += sum(rew_vec)
            steps += 1
            global_step += 1
            obs = obs2

        # checkpointing
        if ep_return > best_return:
            best_return = ep_return
            for i, ag in enumerate(agents):
                ag.save(os.path.join(save_dir, f"agent_{i}_best.pt"))

        if ep % 50 == 0:
            for i, ag in enumerate(agents):
                ag.save(os.path.join(save_dir, f"agent_{i}_ep{ep}.pt"))

        episode_returns.append(ep_return)
        print(f"Episode {ep} | steps {steps} | return {ep_return:.2f} | loss {ep_loss:.3f}")

    # -------- save reward curve (CSV + PNG) --------
    rewards = np.asarray(episode_returns, dtype=float)
    csv_path = os.path.join(save_dir, "episode_rewards.csv")
    np.savetxt(csv_path, rewards, delimiter=",", header="episode_reward", comments="")
    print(f"Saved episode rewards to {csv_path}")

    try:
        import matplotlib.pyplot as plt  # optional
        window = max(1, len(rewards) // 20)  # ~5% moving average
        if window > 1:
            cumsum = np.cumsum(np.insert(rewards, 0, 0.0))
            ma = (cumsum[window:] - cumsum[:-window]) / float(window)
            ma_aligned = np.concatenate([np.full(window - 1, np.nan), ma])
        else:
            ma_aligned = rewards

        plt.figure()
        plt.plot(rewards, label="Episode reward")
        plt.plot(ma_aligned, label=f"Moving avg ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        png_path = os.path.join(save_dir, "reward_curve.png")
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"Saved reward curve plot to {png_path}")
    except Exception as e:
        print(f"(Skipping plot: matplotlib not available or plotting failed: {e})")
    # ------------------------------------------------------

    env.close()
    print("Training finished. Best return:", best_return)

if __name__ == "__main__":
    # train with any N; keep eval config identical
    train(seed=0, n_agents=8, episodes=25000, render=False)
