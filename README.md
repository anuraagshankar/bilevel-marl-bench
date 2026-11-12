# Bilevel MARL Benchmark

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

Multi-agent environments for testing bilevel strategies. Every environment is a [PettingZoo `ParallelEnv`](https://pettingzoo.farama.org/api/parallel/).

## Environments

| Name | Based on | Changes |
|---|---|---|
| `combined_arms` | [MAgent2](https://magent2.farama.org/environments/combined_arms/) | Melee and ranged counts set per team |
| `battlefield` | [MAgent2](https://magent2.farama.org/environments/battlefield/) | None |
| `football` | [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) | None |
| `paintball` | [Melting Pot](https://github.com/google-deepmind/meltingpot) via [Shimmy](https://shimmy.farama.org/environments/meltingpot/) | None |

## Installation

```bash
git clone https://github.com/anuraagshankar/bilevel-marl-bench.git
cd bilevel-marl-bench
pip install -e ".[all]"
```

Replace `all` with `magent`, `vmas` or `meltingpot` to install a single backend.

## Usage

```python
from bilevel_marl_bench import make

env = make("combined_arms", render_mode="human")
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

`render_mode`: `"human"` opens a window, `"rgb_array"` returns frames from `env.render()`, `None` (default) disables rendering.

### Combined arms

<p align="center">
  <img src="assets/combined_arms.png" width="360" alt="Combined arms">
</p>

```python
env = make("combined_arms", red_melee=2, red_ranged=3, blue_melee=3, blue_ranged=2)
```

Defaults are 45 melee and 36 ranged per team.

## Examples

```bash
python examples/random_rollout.py   # all environments
python examples/combined_arms.py    # combined arms with custom team composition
```
