"""Melting Pot paintball with a simplified look."""

import copy

from meltingpot import substrate
from meltingpot.utils.substrates import specs
from shimmy import MeltingPotCompatibilityV0

SOLID = "\n".join(["*" * 8] * 8)  # Plain cell, replacing the brick pattern.
WALLS = ("wall", "destroyable_wall", "destroyed_wall")


def _simplify(settings):
    sim = settings["simulation"]
    # Drop the outer ring of hill-control indicator tiles; the grey wall is the boundary.
    rows = sim["map"].strip("\n").split("\n")
    sim["map"] = "\n".join(row[1:-1] for row in rows[1:-1])
    # Plain grey (wall) and dark grey (destroyable wall) cells.
    sim["prefabs"] = copy.deepcopy(sim["prefabs"])
    for name in WALLS:
        for c in sim["prefabs"][name]["components"]:
            if c["component"] == "Appearance":
                c["kwargs"]["spriteShapes"] = [SOLID] * len(c["kwargs"]["spriteShapes"])
    return settings


def Paintball(variant, **kwargs):
    """variant: "capture_the_flag" or "king_of_the_hill"."""
    config = substrate.get_config(f"paintball__{variant}")
    build = config.lab2d_settings_builder
    with config.unlocked():
        config.lab2d_settings_builder = lambda **kw: _simplify(build(**kw))
        config.timestep_spec = specs.timestep({**config.timestep_spec.observation, "WORLD.RGB": specs.rgb(168, 168)})
    env = substrate.build_from_config(config, roles=config.default_player_roles)
    return MeltingPotCompatibilityV0(env=env, **kwargs)
