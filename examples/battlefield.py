"""Random rollout in battlefield with custom melee/ranged counts per team."""

from bilevel_marl_bench import make

env = make(
    "battlefield",
    red_melee=8,
    red_ranged=4,
    blue_melee=4,
    blue_ranged=8,
    render_mode="human",
)
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
