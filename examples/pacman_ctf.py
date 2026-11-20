"""Random rollout in Pacman capture the flag."""

from bilevel_marl_bench import make

env = make("pacman_ctf", render_mode="human", map_name="open")
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
