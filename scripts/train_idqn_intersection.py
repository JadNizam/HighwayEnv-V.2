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
        self.eps_decay = 50_000  # steps to decay over
        self.steps_done = 0

    def epsilon(self):
        # exponential decay
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

        q = self.policy(state).gather(1, action)  # Q(s,a)
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
def make_env(n_agents=2, render=False, duration=300, vehicles_count=6):
    return gym.make(
        "intersection-v1",
        render_mode="human" if render else None,
        config={
            "controlled_vehicles": n_agents,
            "duration": duration,
            "vehicles_count": vehicles_count,
            "spawn_probability": 0.35,  # gentler traffic for learnability
            # --- observations/actions (multi-agent) ---
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence","x","y","vx","vy","cos_h","sin_h"],
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False
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
            # --- reward shaping knobs ---
            "collision_reward": -10.0,   # harsher penalty
            "high_speed_reward": 0.4,    # gentle dense reward for moving
            "arrived_reward": 5.0,       # strong sparse success
            "reward_speed_range": [4.0, 9.5],  # reward "reasonable fast", not reckless
            "normalize_reward": True,     # scales to [0,1] using [collision, arrived]
            "offroad_terminal": False,
        },
    )

def per_agent_action_space(env):
    # env.action_space is Tuple(Discrete, Discrete, ...)
    assert isinstance(env.action_space, spaces.Tuple)
    return [sp.n for sp in env.action_space.spaces]

# ----------------------- training loop -----------------------
def train(seed=0, n_agents=2, episodes=1000, render=False, save_dir="saved_idqn"):
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = make_env(n_agents=n_agents, render=render, duration=200, vehicles_count=8)
    obs, info = env.reset(seed=seed)

    # per-agent dims
    obs_dim = [flatten_obs(o).shape[0] for o in obs]
    act_dim = per_agent_action_space(env)

    agents = [DQNAgent(obs_dim[i], act_dim[i], lr=1e-3, gamma=0.99, device="cpu")
              for i in range(n_agents)]

    global_step = 0
    best_return = -1e9

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

            # >>> per-agent reward handling <<<
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

        print(f"Episode {ep} | steps {steps} | return {ep_return:.2f} | loss {ep_loss:.3f}")

    env.close()
    print("Training finished. Best return:", best_return)

if __name__ == "__main__":
    # set render=True if you want to see it while training (slower)
    train(seed=0, n_agents=2, episodes=10000, render=False)
