"""Random rollout in Pacman capture the flag with custom roles per team."""

from bilevel_marl_bench import make

env = make(
    "pacman_ctf",
    map_name="crowded",
    red_roles={"border": 1, "greedy": 1},
    blue_roles={"deep": 1, "capped": 1},
    attacker_food=5,
    render_mode="human",
)
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
