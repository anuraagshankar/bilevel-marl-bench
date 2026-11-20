"""Pacman capture the flag (UC Berkeley CS188) as a turn-based PettingZoo AEC env."""

import math
import os
import random
import time
from collections import deque

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from ._internal.pacman_ctf import layout as layouts
from ._internal.pacman_ctf.capture import SCARED_TIME, SONAR_NOISE_RANGE, GameState
from ._internal.pacman_ctf.game import Actions, Directions

FRAME_TIME = 0.03  # Seconds per move in human rendering.
MAPS = os.path.join(os.path.dirname(layouts.__file__), "maps")
ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
# Grid channels, from the observing agent's point of view.
CHANNELS = ["walls", "my_food", "enemy_food", "my_capsules", "enemy_capsules", "self", "teammates", "enemies"]
# free: unrestricted. Defenders never eat food. border: keeps to the border zone and the enemy side.
# deep: keeps to the rest of the home side.
# greedy: can't return home until it carries attacker_food. capped: can't carry more than attacker_food.
ROLES = ["free", "border", "deep", "greedy", "capped"]


def _neighbours(cell):
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _pieces(cells):
    """Connected pieces of a set of cells, largest first."""
    pieces, seen = [], set()
    for start in sorted(cells):
        if start in seen:
            continue
        piece, queue = {start}, deque([start])
        while queue:
            for n in _neighbours(queue.popleft()):
                if n in cells and n not in piece:
                    piece.add(n)
                    queue.append(n)
        seen |= piece
        pieces.append(piece)
    return sorted(pieces, key=len, reverse=True)


def _connect(pieces, allowed):
    """Joins the pieces into one zone through the shortest paths between them inside allowed."""
    zone, rest = set(pieces[0]), pieces[1:]
    while rest:
        parent, queue, hit = {c: None for c in zone}, deque(sorted(zone)), None
        while queue and hit is None:
            cell = queue.popleft()
            hit = next((p for p in rest if cell in p), None)
            if hit is None:
                for n in _neighbours(cell):
                    if n in allowed and n not in parent:
                        parent[n] = cell
                        queue.append(n)
        if hit is None:  # Not reachable inside allowed.
            break
        while cell is not None:
            zone.add(cell)
            cell = parent[cell]
        zone |= hit
        rest.remove(hit)
    return zone


class PacmanCTF(AECEnv):
    """Agents move one at a time, as in the original game, each observing right before its move.

    Observation: {"grid": (C, H, W) binary channels as in CHANNELS,
                  "vector": [score, timeleft, noisy_distance x n, is_pacman x n,
                             scared_timer x n, num_carrying x n, num_returned x n,
                             one-hot role x team size]}
    Per-agent entries are ordered self, teammates, enemies. The score is from the agent's team's view.
    Reward: change in the team's score (zero-sum). Moves that are illegal or break the agent's
    role become Stop; infos[agent]["action_mask"] marks the allowed actions.
    """

    metadata = {"name": "pacman_ctf", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(self, map_name="default", red_roles=None, blue_roles=None, attacker_food=5,
                 border_fraction=0.25, max_steps=1200, render_mode=None):
        """map_name: a .txt file in maps/. red_roles/blue_roles: {role: count}, all free by default.
        attacker_food: food a greedy attacker must carry before returning, and the most a capped one carries.
        border_fraction: fraction of each home side's cells, those fewest moves from the enemy side,
        that form the border zone; the rest is the deep zone.
        max_steps: total moves over all agents, as in the original.
        """
        if not 0 < border_fraction < 1:
            raise ValueError("border_fraction must be between 0 and 1")
        self.layout = layouts.get_layout(os.path.join(MAPS, f"{map_name}.txt"))
        self.attacker_food = attacker_food
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

        self._roles = {}
        for team, roles in (("red", red_roles), ("blue", blue_roles)):
            expanded = [r for r, c in (roles or {"free": counts[team]}).items() for _ in range(c)]
            if len(expanded) != counts[team] or not set(expanded) <= set(ROLES):
                raise ValueError(f"{team}_roles must give {counts[team]} agents with roles from {ROLES}")
            for j, r in enumerate(expanded):
                self._roles[f"{team}_{j}"] = r

        # Defender zones: the home cells fewest moves from the enemy side are the border zone, the
        # rest the deep zone. Border defenders may also cross the enemy side. Border pieces are
        # joined by their shortest connecting paths, and the deep zone keeps only its largest
        # piece, so a defender is never shut in a small pocket.
        # Stored as the maze distance to the zone, from every cell.
        w, h = self.layout.width, self.layout.height
        cells = [(x, y) for x in range(w) for y in range(h) if not self.layout.walls[x][y]]
        self._zones = {}
        for team in ("red", "blue"):
            home = {c for c in cells if (c[0] < w // 2) == (team == "red")}
            to_enemy = self._distances([c for c in cells if c not in home])
            moves = sorted(to_enemy.get(c, np.inf) for c in home)
            d = moves[max(0, math.ceil(border_fraction * len(moves)) - 1)]
            border = {c for c in home if to_enemy.get(c, np.inf) <= d}
            if border == home:
                raise ValueError("border_fraction leaves no deep zone")
            border |= set(cells) - home
            zones = {"border": _connect(_pieces(border), set(cells)), "deep": _pieces(home - border)[0]}
            for a in names:
                if a.startswith(team) and self._roles[a] in zones:
                    self._zones[a] = self._distances(zones[self._roles[a]])

        n, food = len(names), self.layout.total_food
        noise = (SONAR_NOISE_RANGE - 1) // 2
        self.observation_spaces = {}
        for a in names:
            m = len(self._teams[a][0]) * len(ROLES)
            low = [-food, 0] + [-noise] * n + [0] * 4 * n + [0] * m
            high = [food, max_steps] + [w + h + noise] * n + [1] * n + [SCARED_TIME] * n + [food] * 2 * n + [1] * m
            self.observation_spaces[a] = spaces.Dict({
                "grid": spaces.Box(0, 1, (len(CHANNELS), h, w), dtype=np.int8),
                "vector": spaces.Box(np.array(low), np.array(high), dtype=np.int32),
            })
        self.action_spaces = {a: spaces.Discrete(len(ACTIONS)) for a in names}

    def _distances(self, sources):
        """Breadth-first maze distance from the nearest source cell."""
        walls = self.layout.walls
        dist = {c: 0 for c in sources}
        queue = deque(sorted(dist))
        while queue:
            cell = queue.popleft()
            for n in _neighbours(cell):
                if not walls[n[0]][n[1]] and n not in dist:
                    dist[n] = dist[cell] + 1
                    queue.append(n)
        return dist

    def _mask(self, agent):
        """Actions that are legal and keep to the agent's role."""
        i = self._index[agent]
        role = self._roles[agent]
        legal = self.state.get_legal_actions(i)
        x, y = (int(round(v)) for v in self.state.get_agent_position(i))
        red, stuck = i in self._red, False
        if role == "greedy":
            c = self.state.get_agent_state(i).num_carrying
            left = (self.state.get_blue_food() if red else self.state.get_red_food()).count()
            stuck = 0 < c < min(self.attacker_food, c + left)  # Can't go home yet.
        mask = np.zeros(len(ACTIONS), dtype=np.int8)
        for a, move in enumerate(ACTIONS):
            ok = move in legal
            if ok and move != Directions.STOP:
                dx, dy = Actions.direction_to_vector(move)
                nxt = (int(x + dx), int(y + dy))
                if role in ("border", "deep"):  # Never move farther from the zone.
                    dist = self._zones[agent]
                    ok = dist.get(nxt, np.inf) <= dist.get((x, y), np.inf)
                elif role == "greedy" and stuck:
                    ok = (nxt[0] < self.layout.width // 2) != red
            mask[a] = ok
        return mask

    def _can_eat(self, agent, carrying):
        role = self._roles[agent]
        return role not in ("border", "deep") and not (role == "capped" and carrying >= self.attacker_food)

    def _update_infos(self):
        self.infos = {a: {"action_mask": self._mask(a)} for a in self.agents}

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
        self._update_infos()

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
        roles = [int(self._roles[self.possible_agents[j]] == r) for j in team for r in ROLES]
        vector = (
            [obs.get_score() if red else -obs.get_score(), obs.data.timeleft]
            + [distances[j] for j in order]
            + [int(s.is_pacman) for s in agents]
            + [s.scared_timer for s in agents]
            + [s.num_carrying for s in agents]
            + [s.num_returned for s in agents]
            + roles
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
            move = ACTIONS[action] if self._mask(agent)[action] else Directions.STOP
            score = self.state.get_score()
            carrying = [s.num_carrying for s in self.state.data.agent_states]
            self.state = self.state.generate_successor(i, move)
            # The game credits food to the first teammate on the cell, not always the mover.
            for j, s in enumerate(self.state.data.agent_states):
                if s.num_carrying > carrying[j] and not self._can_eat(self.possible_agents[j], carrying[j]):
                    x, y = self.state.data._food_eaten  # Put the food back.
                    self.state.data.food[x][y] = True
                    self.state.data._food_eaten = None
                    s.num_carrying -= 1
            change = self.state.get_score() - score
            for a in self.agents:
                self.rewards[a] = change if self._index[a] in self._red else -change
            self._over = self.state.is_over() or self.state.data.timeleft <= 0
            self._update_infos()
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
