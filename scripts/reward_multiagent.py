# scripts/reward_multiagent.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiAgentShapingWrapper(gym.Wrapper):
    """
    Dense multi-agent shaping on top of HighwayEnv:

      Safety (new):
        • Near-headway penalty (front & close)
        • Closing-speed penalty (front & closing)
        • Time-to-collision penalty (min TTC)
        • Team penalty when a collision spike is detected

      Progress & flow (existing):
        • progress reward: +alpha * max(0, v · d_hat)
        • step penalty: -beta   (finish sooner)
        • blocking penalty in center box

    Requires Kinematics obs with observe_intentions=True and features:
    ["presence","x","y","vx","vy","cos_h","sin_h","cos_d","sin_d"]
    Ego is row 0 of each agent's observation.
    """
    def __init__(self,
                 env: gym.Env,
                 # progress/flow
                 progress_alpha: float = 0.02,
                 step_beta: float = 0.005,
                 block_gamma: float = 0.02,
                 block_speed_thresh: float = 0.5,
                 box_xy: float = 12.0,
                 coop_bonus: float = 0.2,
                 arrival_threshold: float = 4.0,
                 # safety knobs
                 danger_radius: float = 8.0,       # meters considered "too close" in front
                 near_k: float = 0.06,             # weight for near-headway penalty
                 close_k: float = 0.02,            # weight for closing-speed penalty
                 ttc_thresh: float = 3.0,          # seconds; penalize TTC below this
                 ttc_k: float = 0.08,              # weight for TTC penalty
                 team_collision_penalty: float = 0.4,  # applied to all agents if collision spike
                 collision_threshold: float = -6.0,    # base reward ≤ this ⇒ treat as crash
                 reduce_to_scalar: str = "mean"):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Tuple), "Expect Tuple action space."
        self.n_agents = len(env.action_space.spaces)

        # flow/progress
        self.alpha = float(progress_alpha)
        self.beta = float(step_beta)
        self.gamma = float(block_gamma)
        self.block_speed = float(block_speed_thresh)
        self.box_xy = float(box_xy)
        self.coop_bonus = float(coop_bonus)
        self.arrival_threshold = float(arrival_threshold)

        # safety
        self.danger_radius = float(danger_radius)
        self.near_k = float(near_k)
        self.close_k = float(close_k)
        self.ttc_thresh = float(ttc_thresh)
        self.ttc_k = float(ttc_k)
        self.team_collision_penalty = float(team_collision_penalty)
        self.collision_threshold = float(collision_threshold)

        assert reduce_to_scalar in ("mean", "sum")
        self.reduce = reduce_to_scalar

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # base per-agent rewards
        if isinstance(info, dict) and "agents_rewards" in info:
            base = np.asarray(info["agents_rewards"], dtype=np.float32)
        elif isinstance(reward, (list, tuple, np.ndarray)):
            base = np.asarray(reward, dtype=np.float32)
        else:
            base = np.full(len(obs), float(reward), dtype=np.float32)

        shaped = base.copy()

        # -------- progress toward destination: v · d_hat (clip at 0)
        for i, o in enumerate(obs):
            ego = np.asarray(o[0], dtype=np.float32)
            # indices: 0:presence 1:x 2:y 3:vx 4:vy 5:cos_h 6:sin_h 7:cos_d 8:sin_d
            vx, vy = float(ego[3]), float(ego[4])
            cos_d = float(ego[7]) if len(ego) >= 8 else 0.0
            sin_d = float(ego[8]) if len(ego) >= 9 else 0.0
            vt = max(0.0, vx * cos_d + vy * sin_d)
            shaped[i] += self.alpha * vt

        # -------- per-step time penalty (finish sooner)
        shaped -= self.beta

        # -------- blocking penalty (idling in the box)
        for i, o in enumerate(obs):
            ego = np.asarray(o[0], dtype=np.float32)
            x, y = float(ego[1]), float(ego[2])
            vx, vy = float(ego[3]), float(ego[4])
            speed = float(np.hypot(vx, vy))
            if abs(x) < self.box_xy and abs(y) < self.box_xy and speed < self.block_speed:
                shaped[i] -= self.gamma

        # -------- SAFETY: near-headway & closing-speed & TTC
        eps = 1e-6
        for i, o in enumerate(obs):
            ego = np.asarray(o[0], dtype=np.float32)
            ex, ey = float(ego[1]), float(ego[2])
            evx, evy = float(ego[3]), float(ego[4])
            hx, hy = float(ego[5]), float(ego[6])  # heading unit vector

            neigh = np.asarray(o[1:], dtype=np.float32)
            if neigh.size == 0:
                continue
            # keep only present neighbors
            mask = neigh[:, 0] > 0.5
            neigh = neigh[mask]
            if neigh.size == 0:
                continue

            dx = neigh[:, 1] - ex
            dy = neigh[:, 2] - ey
            dist = np.hypot(dx, dy) + eps

            # "in front" = positive projection on heading
            front = (dx * hx + dy * hy) > 0.0

            # near-headway penalty: inside danger radius & in front
            close = (dist < self.danger_radius) & front
            if np.any(close):
                # hinge on distance inside radius (max over neighbors)
                near_pen = np.max((self.danger_radius - dist[close]) / self.danger_radius)
                shaped[i] -= self.near_k * float(near_pen)

                # closing-speed penalty (radial component toward ego; negative ⇒ closing)
                rvx = neigh[:, 3] - evx
                rvy = neigh[:, 4] - evy
                vr = (rvx * dx + rvy * dy) / dist  # along line of sight
                closing = (-vr > 0) & close
                if np.any(closing):
                    max_closing = float(np.max(-vr[closing]))  # m/s closing
                    shaped[i] -= self.close_k * max_closing

            # TTC penalty: compute TTC for closing neighbors; penalize if < threshold
            rvx = neigh[:, 3] - evx
            rvy = neigh[:, 4] - evy
            rdotv = dx * rvx + dy * rvy
            speed2 = rvx * rvx + rvy * rvy + eps
            # TTC = - (r·v) / ||v||^2 if closing (r·v < 0); else +inf
            ttc = np.where(rdotv < 0.0, -(rdotv / speed2), np.inf)
            min_ttc = float(np.min(ttc)) if ttc.size else np.inf
            if min_ttc < self.ttc_thresh:
                shaped[i] -= self.ttc_k * (self.ttc_thresh - min_ttc) / self.ttc_thresh

        # -------- cooperative arrival splash if someone gets a big positive spike
        if np.max(base) >= self.arrival_threshold:
            winner = int(np.argmax(base))
            for j in range(len(shaped)):
                if j != winner:
                    shaped[j] += self.coop_bonus

        # -------- team penalty on detected collision spike in base reward
        if float(np.min(base)) <= self.collision_threshold:
            shaped -= self.team_collision_penalty

        # expose vector and return scalar for Gym API
        info = dict(info or {})
        info["agents_rewards"] = tuple(map(float, shaped))
        scalar = float(np.mean(shaped)) if self.reduce == "mean" else float(np.sum(shaped))
        return obs, scalar, terminated, truncated, info
