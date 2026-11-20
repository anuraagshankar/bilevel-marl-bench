# capture.py
# ----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# capture.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
Capture.py holds the logic for Pacman capture the flag.

    (i) Your interface to the pacman world:
            Pacman is a complex environment.  You probably don't want to
            read through all the code we wrote to make the game runs
            correctly.  This section contains the parts of the code
            that you will need to understand in order to complete the
            project.  There is also some code in game.py that you should
            understand.

    (ii)  The hidden secrets of pacman:
            This section contains all the logic code that the pacman
            environment uses to decide who can move where, who dies when
            things collide, etc.  You shouldn't need to read this section
            of code, but you can if you want.
"""
import random
import sys

from .game import Actions
from .game import GameStateData, Game, Grid, Configuration
from .util import nearest_point, manhattan_distance

# If you change these, you won't affect the server, so you can't cheat
KILL_POINTS = 0
SONAR_NOISE_RANGE = 13  # Must be odd
SONAR_NOISE_VALUES = [i - (SONAR_NOISE_RANGE - 1) / 2 for i in range(SONAR_NOISE_RANGE)]
SIGHT_RANGE = 5  # Manhattan distance
MIN_FOOD = 2
TOTAL_FOOD = 60

DUMP_FOOD_ON_DEATH = True  # if we have the gameplay element that dumps dots on death

SCARED_TIME = 40


def compute_noisy_distance(pos1, pos2):
    return int(manhattan_distance(pos1, pos2) + random.choice(SONAR_NOISE_VALUES))


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
  A GameState specifies the full game state, including the food, capsules,
  agent configurations and score changes.

  GameStates are used by the Game object to capture the actual state of the game and
  can be used by agents to reason about the game.

  Much of the information in a GameState is stored in a GameStateData object.  We
  strongly suggest that you access that data via the accessor methods below rather
  than referring to the GameStateData object directly.
  """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def get_legal_actions(self, agent_index=0):
        """Returns the legal actions for the agent specified."""
        return AgentRules.get_legal_actions(self, agent_index)

    def generate_successor(self, agent_index, action):
        """Returns the successor state (a GameState object) after the specified agent takes the action."""
        # Copy current state
        state = GameState(self)

        # Find appropriate rules for the agent
        AgentRules.apply_action(state, action, agent_index)
        AgentRules.check_death(state, agent_index)
        AgentRules.decrement_timer(state.data.agent_states[agent_index])

        # Bookkeeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        state.data.timeleft = self.data.timeleft - 1
        return state

    def get_agent_state(self, index):
        return self.data.agent_states[index]

    def get_agent_position(self, index):
        """
        Returns a location tuple if the agent with the given index is observable;
        if the agent is unobservable, returns None.
        """
        agent_state = self.data.agent_states[index]
        ret = agent_state.get_position()
        if ret:
            return tuple(int(x) for x in ret)
        return ret

    def get_num_agents(self):
        return len(self.data.agent_states)

    def get_score(self):
        """Returns a number corresponding to the current score."""
        return self.data.score

    def get_red_food(self):
        """
        Returns a matrix of food that corresponds to the food on the red team's side.
        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
        red (meaning red is protecting it, blue is trying to eat it).
        """
        return make_half_grid(self.data.food, red=True)

    def get_blue_food(self):
        """
        Returns a matrix of food that corresponds to the food on the blue team's side.
        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
        blue (meaning blue is protecting it, red is trying to eat it).
        """
        return make_half_grid(self.data.food, red=False)

    def get_red_capsules(self):
        return half_list(self.data.capsules, self.data.food, red=True)

    def get_blue_capsules(self):
        return half_list(self.data.capsules, self.data.food, red=False)

    def get_walls(self):
        """Just like get_food but for walls"""
        return self.data.layout.walls

    def has_food(self, x, y):
        """
        Returns true if the location (x,y) has food, regardless of
        whether it's blue team food or red team food.
        """
        return self.data.food[x][y]

    def has_wall(self, x, y):
        """Returns true if (x,y) has a wall, false otherwise."""
        return self.data.layout.walls[x][y]

    def is_over(self):
        return self.data._win

    def get_red_team_indices(self):
        """Returns a list of agent index numbers for the agents on the red team."""
        return self.red_team[:]

    def get_blue_team_indices(self):
        """Returns a list of the agent index numbers for the agents on the blue team."""
        return self.blue_team[:]

    def is_on_red_team(self, agent_index):
        """Returns true if the agent with the given agent_index is on the red team."""
        return self.teams[agent_index]

    def get_agent_distances(self):
        """Returns a noisy distance to each agent."""
        if 'agent_distances' in dir(self):
            return self.agent_distances
        else:
            return None

    @staticmethod
    def get_distance_prob(true_distance, noisy_distance):
        """Returns the probability of a noisy distance given the true distance"""
        if noisy_distance - true_distance in SONAR_NOISE_VALUES:
            return 1.0 / SONAR_NOISE_RANGE
        else:
            return 0

    def get_initial_agent_position(self, agent_index):
        """Returns the initial position of an agent."""
        return self.data.layout.agent_positions[agent_index][1]

    def get_capsules(self):
        """Returns a list of positions (x,y) of the remaining capsules."""
        return self.data.capsules

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prev_state=None):
        """Generates a new state by copying information from its predecessor."""
        if prev_state is not None:  # Initial state
            self.data = GameStateData(prev_state.data)
            self.blue_team = prev_state.blue_team
            self.red_team = prev_state.red_team
            self.data.timeleft = prev_state.data.timeleft

            self.teams = prev_state.teams
            self.agent_distances = prev_state.agent_distances
        else:
            self.data = GameStateData()
            self.agent_distances = []

    def deep_copy(self):
        state = GameState(self)
        state.data = self.data.deep_copy()
        state.data.timeleft = self.data.timeleft

        state.blue_team = self.blue_team[:]
        state.red_team = self.red_team[:]
        state.teams = self.teams[:]
        state.agent_distances = self.agent_distances[:]
        return state

    def make_observation(self, index):
        state = self.deep_copy()

        # Adds the sonar signal
        pos = state.get_agent_position(index)
        n = state.get_num_agents()
        distances = [compute_noisy_distance(pos, state.get_agent_position(i)) for i in range(n)]
        state.agent_distances = distances

        # Remove states of distant opponents
        if index in self.blue_team:
            team = self.blue_team
            other_team = self.red_team
        else:
            other_team = self.blue_team
            team = self.red_team

        for enemy in other_team:
            seen = False
            enemy_pos = state.get_agent_position(enemy)
            for teammate in team:
                if manhattan_distance(enemy_pos, state.get_agent_position(teammate)) <= SIGHT_RANGE:
                    seen = True
            if not seen: state.data.agent_states[enemy].configuration = None
        return state

    def __eq__(self, other):
        """Allows two states to be compared."""
        if other is None: return False
        return self.data == other.data

    def __hash__(self):
        """Allows states to be keys of dictionaries."""
        return int(hash(self.data))

    def __str__(self):
        return str(self.data)

    def initialize(self, layout, num_agents):
        """Creates an initial game state from a layout array (see layout.py)."""
        self.data.initialize(layout, num_agents)
        positions = [a.configuration for a in self.data.agent_states]
        self.blue_team = [i for i, p in enumerate(positions) if not self.is_red(p)]
        self.red_team = [i for i, p in enumerate(positions) if self.is_red(p)]
        self.teams = [self.is_red(p) for p in positions]
        # This is usually 60 (always 60 with random maps)
        # However, if layout map is specified otherwise, it could be less
        global TOTAL_FOOD
        TOTAL_FOOD = layout.total_food

    def is_red(self, config_or_pos):
        width = self.data.layout.width
        if config_or_pos is tuple:
            return config_or_pos[0] < width // 2
        else:
            return config_or_pos.pos[0] < width // 2


def make_half_grid(grid, red):
    halfway = grid.width // 2
    half_grid = Grid(grid.width, grid.height, False)
    if red:
        xrange = range(halfway)
    else:
        xrange = range(halfway, grid.width)

    for y in range(grid.height):
        for x in xrange:
            if grid[x][y]: half_grid[x][y] = True

    return half_grid


def half_list(locations, grid, red):
    halfway = grid.width // 2
    new_list = []
    for x, y in locations:
        if red and x < halfway:
            new_list.append((x, y))
        elif not red and x >= halfway:
            new_list.append((x, y))
    return new_list


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill


class CaptureRules:
    """
  These game rules manage the control flow of a game, deciding when
  and how the game starts and ends.
  """

    def __init__(self, quiet=False):
        self._init_blue_food = None
        self._init_red_food = None
        self.quiet = quiet

    def new_game(self, layout, agents, display, length, mute_agents, catch_exceptions):
        init_state = GameState()
        init_state.initialize(layout, len(agents))
        starter = random.randint(0, 1)
        print('%s team starts' % ['Red', 'Blue'][starter])
        game = Game(agents, display, self, starting_index=starter, mute_agents=mute_agents,
                    catch_exceptions=catch_exceptions)
        game.state = init_state
        game.length = length
        game.state.data.timeleft = length
        if 'drawCenterLine' in dir(display):
            display.drawCenterLine()
        self._init_blue_food = init_state.get_blue_food().count()
        self._init_red_food = init_state.get_red_food().count()
        return game

    @staticmethod
    def process(state, game):
        """Checks to see whether it is time to end the game."""
        if 'move_history' in dir(game):
            if len(game.move_history) == game.length:
                state.data._win = True

        if state.is_over():
            game.game_over = True
            if not game.rules.quiet:
                red_count = 0
                blue_count = 0
                food_to_win = (TOTAL_FOOD / 2) - MIN_FOOD
                for index in range(state.get_num_agents()):
                    agent_state = state.data.agent_states[index]
                    if index in state.get_red_team_indices():
                        red_count += agent_state.num_returned
                    else:
                        blue_count += agent_state.num_returned

                if blue_count >= food_to_win:  # state.getRedFood().count() == MIN_FOOD:
                    print(f'The Blue team has returned at least {food_to_win} of the opponents\' dots.')
                elif red_count >= food_to_win:  # state.getBlueFood().count() == MIN_FOOD:
                    print(f'The Red team has returned at least {food_to_win} of the opponents\' dots.')
                else:  # if state.getBlueFood().count() > MIN_FOOD and state.getRedFood().count() > MIN_FOOD:
                    print('Time is up.')
                    if state.data.score == 0:
                        print('Tie game!')
                    else:
                        winner = 'Red'
                        if state.data.score < 0: winner = 'Blue'
                        print(f'The {winner} team wins by {abs(state.data.score)} points.')

    @staticmethod
    def agent_crash(game, agent_index):
        if agent_index % 2 == 0:
            print("Red agent crashed", file=sys.stderr)
            game.state.data.score = -1
        else:
            print("Blue agent crashed", file=sys.stderr)
            game.state.data.score = 1

    @staticmethod
    def get_max_total_time():
        return 900  # Move limits should prevent this from ever happening

    @staticmethod
    def get_max_startup_time():
        return 15  # 15 seconds for register_initial_state

    @staticmethod
    def get_move_warning_time():
        return 1  # One second per move

    @staticmethod
    def get_move_timeout():
        return 3  # Three seconds results in instant forfeit

    @staticmethod
    def get_max_time_warnings():
        return 2  # Third violation loses the game


class AgentRules:
    """
    These functions govern how each agent interacts with her environment.
    """

    @staticmethod
    def get_legal_actions(state, agent_index):
        """
        Returns a list of legal actions (which are both possible & allowed)
        """
        agent_state = state.get_agent_state(agent_index)
        conf = agent_state.configuration
        possible_actions = Actions.get_possible_actions(conf, state.data.layout.walls)
        return AgentRules.filter_for_allowed_actions(possible_actions)

    @staticmethod
    def filter_for_allowed_actions(possible_actions):
        return possible_actions

    @staticmethod
    def apply_action(state, action, agent_index):
        """Edits the state to reflect the results of the action."""
        legal = AgentRules.get_legal_actions(state, agent_index)
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        # Update Configuration
        agent_state = state.data.agent_states[agent_index]
        speed = 1.0
        # if agent_state.is_pacman: speed = 0.5
        vector = Actions.direction_to_vector(action, speed)
        old_config = agent_state.configuration
        agent_state.configuration = old_config.generate_successor(vector)

        # Eat
        current_position = agent_state.configuration.get_position()
        nearest = nearest_point(current_position)

        if current_position == nearest:
            is_red = state.is_on_red_team(agent_index)
            # Change agent type
            agent_state.is_pacman = [is_red, state.is_red(agent_state.configuration)].count(True) == 1
            # if he's no longer pacman, he's on his own side, so reset the num carrying timer
            # agent_state.numCarrying *= int(agent_state.is_pacman)
            if agent_state.num_carrying > 0 and not agent_state.is_pacman:
                score = agent_state.num_carrying if is_red else -1 * agent_state.num_carrying
                state.data.score_change += score

                agent_state.num_returned += agent_state.num_carrying
                agent_state.num_carrying = 0

                red_count = 0
                blue_count = 0
                for index in range(state.get_num_agents()):
                    agent = state.data.agent_states[index]
                    if index in state.get_red_team_indices():
                        red_count += agent.num_returned
                    else:
                        blue_count += agent.num_returned
                if red_count >= (TOTAL_FOOD / 2) - MIN_FOOD or blue_count >= (TOTAL_FOOD / 2) - MIN_FOOD:
                    state.data._win = True

        if agent_state.is_pacman and manhattan_distance(nearest, current_position) <= 0.9:
            AgentRules.consume(nearest, state, state.is_on_red_team(agent_index))

    @staticmethod
    def consume(position, state, is_red):
        x, y = position
        # Eat food
        if state.data.food[x][y]:

            # blue case is the default
            team_indices_func = state.get_blue_team_indices
            # score = -1
            if is_red:
                # switch if its red
                # score = 1
                team_indices_func = state.get_red_team_indices

            # go increase the variable for the pacman who ate this
            agents = [state.data.agent_states[agentIndex] for agentIndex in team_indices_func()]
            for agent in agents:
                if agent.get_position() == position:
                    agent.num_carrying += 1
                    break  # the above should only be true for one agent...

            # do all the score and food grid maintenance
            # state.data.scoreChange += score
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._food_eaten = position
            # if (isRed and state.get_blue_food().count() == MIN_FOOD) or
            # (not isRed and state.get_red_food().count() == MIN_FOOD):
            #  state.data._win = True

        # Eat capsule
        if is_red:
            my_capsules = state.get_blue_capsules()
        else:
            my_capsules = state.get_red_capsules()
        if position in my_capsules:
            state.data.capsules.remove(position)
            state.data._capsule_eaten = position

            # Reset all ghosts' scared timers
            if is_red:
                other_team = state.get_blue_team_indices()
            else:
                other_team = state.get_red_team_indices()
            for index in other_team:
                state.data.agent_states[index].scared_timer = SCARED_TIME

    @staticmethod
    def decrement_timer(state):
        timer = state.scared_timer
        if timer == 1:
            state.configuration.pos = nearest_point(state.configuration.pos)
        state.scared_timer = max(0, timer - 1)

    @staticmethod
    def dump_food_from_death(state, agent_state):
        if not DUMP_FOOD_ON_DEATH:
            # this feature is not turned on
            return

        if not agent_state.is_pacman:
            raise Exception('something is seriously wrong, this agent isn\'t a pacman!')

        # ok so agent_state is this:
        if agent_state.num_carrying == 0:
            return

        # first, score changes!
        # we HACK pack that ugly bug by just determining if its red based on the first position
        # to die...
        dummy_config = Configuration(agent_state.get_position(), 'North')
        is_red = state.is_red(dummy_config)

        # the score increases if red eats dots, so if we are refunding points,
        # the direction should be -1 if the red agent died, which means he dies
        # on the blue side
        # score_direction = (-1) ** (int(is_red) + 1)

        # state.data.scoreChange += scoreDirection * agent_state.numCarrying

        def on_right_side(from_state, from_x, from_y):
            new_dummy_config = Configuration((from_x, from_y), 'North')
            return from_state.is_red(new_dummy_config) == is_red

        # we have food to dump
        # -- expand out in BFS. Check:
        #   - that it's within the limits
        #   - that it's not a wall
        #   - that no other agents are there
        #   - that no power pellets are there
        #   - that it's on the right side of the grid
        def all_good(from_state, from_x, from_y):
            width, height = from_state.data.layout.width, from_state.data.layout.height
            food, walls = from_state.data.food, from_state.data.layout.walls

            # bounds check
            if from_x >= width or from_y >= height or from_x <= 0 or from_y <= 0:
                return False

            if walls[from_x][from_y]:
                return False
            if food[from_x][from_y]:
                return False

            # dots need to be on the side where this agent will be a pacman :P
            if not on_right_side(from_state, from_x, from_y):
                return False

            if (from_x, from_y) in from_state.data.capsules:
                return False

            # loop through agents
            agent_poses = [from_state.get_agent_position(i) for i in range(from_state.get_num_agents())]
            if (from_x, from_y) in agent_poses:
                return False

            return True

        num_to_dump = agent_state.num_carrying
        state.data.food = state.data.food.copy()
        food_added = []

        def gen_successors(from_x, from_y):
            dirs_x = [-1, 0, 1]
            dirs_y = [-1, 0, 1]
            return [(from_x + dx, from_y + dy) for dx in dirs_x for dy in dirs_y]

        # BFS graph search
        position_queue = [agent_state.get_position()]
        seen = set()
        while num_to_dump > 0:
            if not len(position_queue):
                raise Exception('Exhausted BFS! uh oh')
            # pop one off, graph check
            popped = position_queue.pop(0)
            if popped in seen:
                continue
            seen.add(popped)

            x, y = popped[0], popped[1]
            x = int(x)
            y = int(y)
            if all_good(state, x, y):
                state.data.food[x][y] = True
                food_added.append((x, y))
                num_to_dump -= 1

            # generate successors
            position_queue = position_queue + gen_successors(x, y)

        if state.data._food_added is None:
            state.data._food_added = food_added
        else:
            state.data._food_added.extend(food_added)
        # now our agent_state is no longer carrying food
        agent_state.num_carrying = 0
        pass

    @staticmethod
    def check_death(state, agent_index):
        agent_state = state.data.agent_states[agent_index]
        if state.is_on_red_team(agent_index):
            other_team = state.get_blue_team_indices()
        else:
            other_team = state.get_red_team_indices()
        if agent_state.is_pacman:
            for index in other_team:
                other_agent_state = state.data.agent_states[index]
                if other_agent_state.is_pacman: continue
                ghost_position = other_agent_state.get_position()
                if ghost_position is None: continue
                if manhattan_distance(ghost_position, agent_state.get_position()) <= COLLISION_TOLERANCE:
                    # award points to the other team for killing Pacmen
                    if other_agent_state.scared_timer <= 0:
                        AgentRules.dump_food_from_death(state, agent_state)

                        score = KILL_POINTS
                        if state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        agent_state.is_pacman = False
                        agent_state.configuration = agent_state.start
                        agent_state.scared_timer = 0
                    else:
                        score = KILL_POINTS
                        if state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        other_agent_state.is_pacman = False
                        other_agent_state.configuration = other_agent_state.start
                        other_agent_state.scared_timer = 0
        else:  # Agent is a ghost
            for index in other_team:
                other_agent_state = state.data.agent_states[index]
                if not other_agent_state.is_pacman: continue
                pac_pos = other_agent_state.get_position()
                if pac_pos is None: continue
                if manhattan_distance(pac_pos, agent_state.get_position()) <= COLLISION_TOLERANCE:
                    # award points to the other team for killing Pacmen
                    if agent_state.scared_timer <= 0:
                        AgentRules.dump_food_from_death(state, other_agent_state)

                        score = KILL_POINTS
                        if not state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        other_agent_state.is_pacman = False
                        other_agent_state.configuration = other_agent_state.start
                        other_agent_state.scared_timer = 0
                    else:
                        score = KILL_POINTS
                        if state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        agent_state.is_pacman = False
                        agent_state.configuration = agent_state.start
                        agent_state.scared_timer = 0

    @staticmethod
    def place_ghost(ghost_state):
        ghost_state.configuration = ghost_state.start
