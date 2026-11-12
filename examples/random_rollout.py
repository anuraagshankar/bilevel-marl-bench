"""Run a short random rollout in every environment."""

from bilevel_marl_bench import ENVS, make

for name in ENVS:
    env = make(name, render_mode="human")
    obs, infos = env.reset(seed=0)
    for _ in range(100):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
    print(f"{name}: OK ({len(env.possible_agents)} agents)")
    env.close()
