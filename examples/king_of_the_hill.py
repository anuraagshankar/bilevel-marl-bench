"""Random rollout in king of the hill with custom shooter types per team."""

from bilevel_marl_bench import make

env = make(
    "king_of_the_hill", 
    red_wide=3, red_long=2, 
    blue_wide=2, blue_long=3, 
    render_mode="human"
)

obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
