"""Random rollout in combined arms with custom melee/ranged counts per team."""

from bilevel_marl_bench import make

env = make(
    "combined_arms",
    red_melee=2,
    red_ranged=3,
    blue_melee=3,
    blue_ranged=2,
    render_mode="human",
)
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
