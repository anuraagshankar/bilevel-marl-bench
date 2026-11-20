"""Render the default Pacman CTF map with a few random moves."""

import os
import random
import time

from bilevel_marl_bench._internal.pacman_ctf import layout
from bilevel_marl_bench._internal.pacman_ctf.capture import GameState
from bilevel_marl_bench._internal.pacman_ctf.capture_graphics_display import PacmanGraphics

MAPS = os.path.join(os.path.dirname(layout.__file__), "maps")

lay = layout.get_layout(os.path.join(MAPS, "defaultCapture.txt"))
state = GameState()
state.initialize(lay, lay.get_num_ghosts())
state.data.timeleft = 100

display = PacmanGraphics("Red", None, "Blue", None)
display.initialize(state.data)

for t in range(100):
    i = t % state.get_num_agents()
    state = state.generate_successor(i, random.choice(state.get_legal_actions(i)))
    display.update(state.data)
    time.sleep(0.05)

display.finish()
