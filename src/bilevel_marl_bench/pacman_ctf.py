"""Pacman capture the flag (UC Berkeley CS188) as a turn-based PettingZoo AEC env."""

import os
import random
import time

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from ._internal.pacman_ctf import layout as layouts
from ._internal.pacman_ctf.capture import SCARED_TIME, SONAR_NOISE_RANGE, GameState
from ._internal.pacman_ctf.game import Directions

FRAME_TIME = 0.03  # Seconds per move in human rendering.
MAPS = os.path.join(os.path.dirname(layouts.__file__), "maps")
ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
# Grid channels, from the observing agent's point of view.
CHANNELS = ["walls", "my_food", "enemy_food", "my_capsules", "enemy_capsules", "self", "teammates", "enemies"]


class PacmanCTF(AECEnv):
    """Agents move one at a time, as in the original game, each observing right before its move.

    Observation: {"grid": (C, H, W) binary channels as in CHANNELS,
                  "vector": [score, timeleft, noisy_distance x n, is_pacman x n,
                             scared_timer x n, num_carrying x n, num_returned x n]}
    Per-agent entries are ordered self, teammates, enemies. The score is from the agent's team's view.
    Reward: change in the team's score (zero-sum). Illegal actions become Stop.
    """

    metadata = {"name": "pacman_ctf", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(self, map_name="default", max_steps=1200, render_mode=None):
        """map_name: a .txt file in maps/. max_steps: total moves over all agents, as in the original."""
        self.layout = layouts.get_layout(os.path.join(MAPS, f"{map_name}.txt"))
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._display = None

        state = GameState()
        state.initialize(self.layout, self.layout.get_num_ghosts())
        self._red = state.red_team
        names, counts = [], {"red": 0, "blue": 0}
        for i in range(state.get_num_agents()):
            team = "red" if i in self._red else "blue"
            names.append(f"{team}_{counts[team]}")
            counts[team] += 1
        self.possible_agents = names
        self._index = {a: i for i, a in enumerate(names)}
        # Per agent: its team (self first) and the enemy team.
        self._teams = {}
        for i, a in enumerate(names):
            mine = self._red if i in self._red else state.blue_team
            self._teams[a] = ([i] + [j for j in mine if j != i], [j for j in range(len(names)) if j not in mine])

        n, w, h, food = len(names), self.layout.width, self.layout.height, self.layout.total_food
        noise = (SONAR_NOISE_RANGE - 1) // 2
        low = [-food, 0] + [-noise] * n + [0] * 4 * n
        high = [food, max_steps] + [w + h + noise] * n + [1] * n + [SCARED_TIME] * n + [food] * 2 * n
        obs_space = spaces.Dict({
            "grid": spaces.Box(0, 1, (len(CHANNELS), h, w), dtype=np.int8),
            "vector": spaces.Box(np.array(low), np.array(high), dtype=np.int32),
        })
        self.observation_spaces = {a: obs_space for a in names}
        self.action_spaces = {a: spaces.Discrete(len(ACTIONS)) for a in names}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)  # The game's sonar noise uses the global random module.
        self.state = GameState()
        self.state.initialize(self.layout, len(self.possible_agents))
        self.state.data.timeleft = self.max_steps
        self._over = False

        # As in the original, red or blue moves first at random; turns then go in index order.
        start = random.randint(0, 1)
        self.agents = self.possible_agents[start:] + self.possible_agents[:start]
        self._turn = 0
        self.agent_selection = self.agents[0]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        if self.render_mode == "human":
            from ._internal.pacman_ctf.capture_graphics_display import PacmanGraphics

            self._display = PacmanGraphics("Red", None, "Blue", None)
            self._display.initialize(self.state.data)

    def observe(self, agent):
        i = self._index[agent]
        obs = self.state.make_observation(i)
        red = i in self._red
        h = self.layout.height

        def grid(g):
            return np.array(g.data, dtype=np.int8).T[::-1]

        def mark(positions):
            m = np.zeros((h, self.layout.width), dtype=np.int8)
            for p in positions:
                if p is not None:
                    m[h - 1 - p[1], p[0]] = 1
            return m

        team, enemies = self._teams[agent]
        order = team + enemies
        my_food, enemy_food = obs.get_red_food(), obs.get_blue_food()
        my_caps, enemy_caps = obs.get_red_capsules(), obs.get_blue_capsules()
        if not red:
            my_food, enemy_food, my_caps, enemy_caps = enemy_food, my_food, enemy_caps, my_caps
        channels = [
            grid(obs.get_walls()), grid(my_food), grid(enemy_food), mark(my_caps), mark(enemy_caps),
            mark([obs.get_agent_position(i)]),
            mark([obs.get_agent_position(j) for j in team[1:]]),
            mark([obs.get_agent_position(j) for j in enemies]),
        ]

        agents = [obs.get_agent_state(j) for j in order]
        distances = obs.get_agent_distances()
        vector = (
            [obs.get_score() if red else -obs.get_score(), obs.data.timeleft]
            + [distances[j] for j in order]
            + [int(s.is_pacman) for s in agents]
            + [s.scared_timer for s in agents]
            + [s.num_carrying for s in agents]
            + [s.num_returned for s in agents]
        )
        return {"grid": np.stack(channels), "vector": np.array(vector, dtype=np.int32)}

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()

        # Once the game is over, the rest of the round's moves are ignored: PettingZoo's
        # aec_to_parallel requires agents to finish together at the end of a round.
        if not self._over:
            i = self._index[agent]
            move = ACTIONS[action]
            if move not in self.state.get_legal_actions(i):
                move = Directions.STOP
            score = self.state.get_score()
            self.state = self.state.generate_successor(i, move)
            change = self.state.get_score() - score
            for a in self.agents:
                self.rewards[a] = change if self._index[a] in self._red else -change
            self._over = self.state.is_over() or self.state.data.timeleft <= 0
            if self.render_mode == "human":
                self.render()
        self._accumulate_rewards()

        self._turn = (self._turn + 1) % len(self.agents)
        self.agent_selection = self.agents[self._turn]
        if self._turn == 0 and self._over:
            won = self.state.is_over()  # Enough food returned; otherwise time ran out.
            for a in self.agents:
                self.terminations[a] = won
                self.truncations[a] = not won

    def render(self):
        if self._display is not None and self.state.data._agent_moved is not None:
            self._display.update(self.state.data)
            time.sleep(FRAME_TIME)

    def close(self):
        if self._display is not None:
            self._display.finish()
            self._display = None
