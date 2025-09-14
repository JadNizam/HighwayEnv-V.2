# scripts/run_intersection_multiagent.py
import time
import gymnasium as gym
import highway_env
from gymnasium import spaces

def make_env(n_agents=2, render=False, duration=120):
    return gym.make(
        "intersection-v0",
        render_mode="human" if render else None,
        config={
            "controlled_vehicles": n_agents,
            "duration": duration,  # increase horizon
            "vehicles_count": 8,   # keep traffic modest
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {"type": "Kinematics"}
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"}
            },
        },
    )

def sample_multi_action(action_space):
    """Return a valid per-agent action tuple."""
    if isinstance(action_space, spaces.Tuple):
        # sample each agent's subspace
        return tuple(sp.sample() for sp in action_space.spaces)
    # fallback: entire space already encodes all agents
    return action_space.sample()

if __name__ == "__main__":
    env = make_env(n_agents=2, render=True, duration=200)
    obs, info = env.reset(seed=0)

    steps = 0
    terminated = truncated = False

    # OPTIONAL: keep everyone idle for a bit so you can see it persists
    # DiscreteMetaAction usually maps: 0=IDLE, 1=LEFT, 2=RIGHT, 3=FASTER, 4=SLOWER
    idle_warmup_steps = 20

    while not (terminated or truncated):
        if steps < idle_warmup_steps and isinstance(env.action_space, spaces.Tuple):
            action = tuple(0 for _ in env.action_space.spaces)  # hold still
        else:
            action = sample_multi_action(env.action_space)

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        # slow down rendering a touch
        if env.render_mode == "human":
            time.sleep(0.10)

    print(f"Episode finished after {steps} steps. reward={reward}")
    env.close()
