# scripts/shared_context_wrapper.py
import numpy as np
import gymnasium as gym

class SharedContextWrapper(gym.Wrapper):
    """
    Augment each agent's observation with shared/global context + peer-relative info.

    Adds (per agent):
      - Global context (same for all agents):
          [ num_controlled, num_in_conflict_box, mean_speed, min_dist_to_center ]
      - k-nearest peer relatives (per peer):
          [ dx, dy, dvx, dvy ]  (relative to this agent)
      - Cheap-talk (optional): one-hot of each OTHER agent's last discrete action,
        concatenated across peers => shape: (n_agents - 1) * action_space.n

    Assumes base observation is MultiAgent Kinematics with ego at row 0:
      features: ["presence","x","y","vx","vy", ...]
    """
    def __init__(self, env, k_nearest=3, box_xy=12.0, include_last_actions=True):
        super().__init__(env)
        self.k_nearest = int(k_nearest)
        self.box_xy = float(box_xy)
        self.include_last_actions = bool(include_last_actions)
        self._last_actions = None  # per-agent ints

    # ------------- Gym API -------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # number of controlled agents is the length of the multi-agent obs list
        n = len(obs)
        # default last actions to zeros at episode start
        self._last_actions = [0 for _ in range(n)]
        return self._augment(obs), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        # Save last actions (defensive: accept tuple/list/np.ndarray/int)
        if isinstance(actions, (list, tuple, np.ndarray)):
            self._last_actions = [int(a) for a in actions]
        else:
            # single int should never happen in MultiAgentAction, but be safe
            self._last_actions = [int(actions)] * len(obs)

        return self._augment(obs), reward, terminated, truncated, info

    # ------------- Helpers -------------
    def _augment(self, obs_list):
        """
        obs_list: list of per-agent arrays with shape [vehicles_count, features]
                  or possibly already flattened (we handle both).
        Returns: list of flattened arrays with appended shared context.
        """
        n = len(obs_list)

        # --- Extract per-agent ego (x,y,vx,vy) ---
        egos = []
        for oi in obs_list:
            arr = np.asarray(oi, dtype=np.float32)
            if arr.ndim == 1 or arr.size == 0:
                # Already flat or empty; cannot parse rows -> zeros fallback
                egos.append((0.0, 0.0, 0.0, 0.0))
                continue
            # row 0 is ego; expect indices: [presence(0), x(1), y(2), vx(3), vy(4), ...]
            v0 = arr[0]
            if v0.shape[0] >= 5:
                x, y, vx, vy = float(v0[1]), float(v0[2]), float(v0[3]), float(v0[4])
                # If presence is 0 (rare), zero-out to avoid garbage influence
                if v0[0] < 0.5:
                    x = y = vx = vy = 0.0
            else:
                x = y = vx = vy = 0.0
            egos.append((x, y, vx, vy))

        # --- Global context (same for all agents) ---
        inside = sum(1 for (x, y, _, _) in egos if abs(x) <= self.box_xy and abs(y) <= self.box_xy)
        speeds = [np.hypot(vx, vy) for (_, _, vx, vy) in egos] or [0.0]
        mean_speed = float(np.mean(speeds))
        dists = [np.hypot(x, y) for (x, y, _, _) in egos] or [0.0]
        min_dist = float(np.min(dists))
        global_vec = np.array(
            [float(n), float(inside), mean_speed, min_dist],
            dtype=np.float32
        )

        # --- Action space size for cheap-talk one-hots ---
        a_n = None
        if self.include_last_actions:
            try:
                # Expect Tuple of Discrete
                a_n = int(self.action_space.spaces[0].n)
            except Exception:
                a_n = None  # disable cheap-talk if not available

        # --- Build per-agent augmentation and concat ---
        aug_obs = []
        for i, oi in enumerate(obs_list):
            base_flat = np.asarray(oi, dtype=np.float32).reshape(-1)

            # k-nearest peer relatives
            xi, yi, vxi, vyi = egos[i]
            peers = []
            for j, (xj, yj, vxj, vyj) in enumerate(egos):
                if j == i:
                    continue
                # distance used for sorting; keep deltas to append
                peers.append(
                    (j, np.hypot(xj - xi, yj - yi), xj - xi, yj - yi, vxj - vxi, vyj - vyi)
                )
            peers.sort(key=lambda t: t[1])

            rels = []
            for k in range(self.k_nearest):
                if k < len(peers):
                    _, _, dx, dy, dvx, dvy = peers[k]
                    rels += [dx, dy, dvx, dvy]
                else:
                    rels += [0.0, 0.0, 0.0, 0.0]

            # last actions of peers (one-hot)
            if self.include_last_actions and a_n is not None and a_n > 0:
                onehots = []
                for j in range(n):
                    if j == i:
                        continue
                    one = np.zeros(a_n, dtype=np.float32)
                    idx = int(self._last_actions[j]) if 0 <= int(self._last_actions[j]) < a_n else 0
                    one[idx] = 1.0
                    onehots.append(one)
                act_vec = np.concatenate(onehots, axis=0) if onehots else np.zeros(0, dtype=np.float32)
            else:
                act_vec = np.zeros(0, dtype=np.float32)

            full = np.concatenate(
                [base_flat, global_vec, np.asarray(rels, np.float32), act_vec],
                axis=0
            )
            aug_obs.append(full)

        return aug_obs
