# bilevel-marl-bench

Multi-agent environments for testing bilevel strategies. Every environment is a PettingZoo `ParallelEnv`.

| Name | Source |
|---|---|
| `combined_arms` | MAgent2 |
| `battlefield` | MAgent2 |
| `football` | VMAS |
| `paintball` | Melting Pot (via Shimmy) |

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/anuraagshankar/bilevel-marl-bench.git
cd bilevel-marl-bench
pip install -e ".[all]"
```

To install only some backends, use `[magent]`, `[vmas]` or `[meltingpot]` instead of `[all]`.

## Run a rollout

```python
from bilevel_marl_bench import make

env = make("combined_arms")  # or "battlefield", "football", "paintball"
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

To run a short rollout in every environment:

```bash
python examples/random_rollout.py
```
