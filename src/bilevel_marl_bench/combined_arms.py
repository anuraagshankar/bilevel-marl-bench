"""MAgent2 combined arms with configurable melee/ranged counts per team."""

import math

from gymnasium.utils import EzPickle
from magent2.environments.combined_arms.combined_arms import (
    _parallel_env,
    default_reward_args,
)


def _formation(n_melee, n_ranged):
    """Columns of (type, count), alternating melee (0) / ranged (1) as in the original."""
    rows = math.ceil(math.sqrt(n_melee + n_ranged))
    remaining = [n_melee, n_ranged]
    columns, t = [], 0
    while sum(remaining):
        if remaining[t] == 0:
            t = 1 - t
        k = min(rows, remaining[t])
        columns.append((t, k))
        remaining[t] -= k
        t = 1 - t
    return rows, columns


class CombinedArms(_parallel_env):
    def __init__(
        self,
        red_melee=45,
        red_ranged=36,
        blue_melee=45,
        blue_ranged=36,
        map_size=45,
        max_cycles=1000,
        minimap_mode=False,
        extra_features=False,
        render_mode=None,
        seed=None,
        **reward_args,
    ):
        # Must be set before super().__init__, which calls generate_map().
        self.counts = [(red_melee, red_ranged), (blue_melee, blue_ranged)]
        super().__init__(
            map_size,
            minimap_mode,
            {**default_reward_args, **reward_args},
            max_cycles,
            extra_features,
            render_mode,
            seed,
        )
        EzPickle.__init__(
            self, red_melee=red_melee, red_ranged=red_ranged, blue_melee=blue_melee,
            blue_ranged=blue_ranged, map_size=map_size, max_cycles=max_cycles,
            minimap_mode=minimap_mode, extra_features=extra_features,
            render_mode=render_mode, seed=seed, **reward_args,
        )

    walls = set()  # Wall cells; formation columns touching them are skipped.

    def generate_map(self):
        w = h = self.map_size
        gap = 3
        for team, counts in enumerate(self.counts):
            rows, columns = _formation(*counts)
            y0 = (h - 2 * rows) // 2
            # Column x positions, outward from the centre, skipping columns that hit a wall.
            step = -2 if team == 0 else 2
            x = w // 2 - gap - 2 if team == 0 else w // 2 + gap
            xs = []
            while len(xs) < len(columns):
                if not any((x, y0 + 2 * j) in self.walls for j in range(rows)):
                    xs.append(x)
                x += step
            if team == 0:
                xs.reverse()  # Original order: left team's first column is the outermost.
            pos = [[], []]
            for i, (t, k) in enumerate(columns):
                pos[t] += [[xs[i], y0 + 2 * j] for j in range(k)]
            for x, y in pos[0] + pos[1]:
                if not (0 < x < w - 1 and 0 < y < h - 1):
                    raise ValueError(
                        f"Team {team} with counts {counts} does not fit in map_size={w}."
                    )
            self.env.add_agents(self.handles[2 * team], method="custom", pos=pos[0])
            self.env.add_agents(self.handles[2 * team + 1], method="custom", pos=pos[1])
