"""Random rollout in football with custom roles per team."""

from bilevel_marl_bench import make

env = make(
    "football",
    blue_roles={"gk": 1, "fwd": 2},
    red_roles={"gk": 1, "def": 2},
    max_steps=500,
    render_mode="human",
)
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
