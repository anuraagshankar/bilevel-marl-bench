"""Melting Pot paintball with a simplified look and wide/long shooter types."""

import copy

from meltingpot import substrate
from meltingpot.utils.substrates import specs
from shimmy import MeltingPotCompatibilityV0

SOLID = "\n".join(["*" * 8] * 8)  # Plain cell, replacing the brick pattern.
WALLS = ("wall", "destroyable_wall", "destroyed_wall")


def _simplify(settings, shooters):
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
    # The single fire action shoots the primary beam; long shooters get the original long beam's settings.
    for avatar, shooter in zip(sim["gameObjects"], shooters):
        for c in avatar["components"]:
            if c["component"] == "ColorZapper" and shooter == "long":
                k = c["kwargs"]
                k["cooldownTime"] = k["secondaryBeamCooldownTime"]
                k["beamLength"] = k["secondaryBeamLength"]
                k["beamRadius"] = k["secondaryBeamRadius"]
    return settings


def Paintball(variant, red_wide=2, red_long=2, blue_wide=2, blue_long=2, **kwargs):
    """variant: "capture_the_flag" or "king_of_the_hill"."""
    if red_wide + red_long != blue_wide + blue_long:
        raise ValueError("Both teams must have the same size.")
    red = ["wide"] * red_wide + ["long"] * red_long
    blue = ["wide"] * blue_wide + ["long"] * blue_long
    shooters = [s for pair in zip(red, blue) for s in pair]  # Melting Pot alternates red, blue.

    config = substrate.get_config(f"paintball__{variant}")
    build = config.lab2d_settings_builder
    with config.unlocked():
        config.lab2d_settings_builder = lambda **kw: _simplify(build(**kw), shooters)
        config.action_set = config.action_set[:-1]  # Drop the second fire action.
        config.action_spec = specs.action(len(config.action_set))
        config.timestep_spec = specs.timestep({**config.timestep_spec.observation, "WORLD.RGB": specs.rgb(168, 168)})
    env = substrate.build_from_config(config, roles=("default",) * len(shooters))
    return MeltingPotCompatibilityV0(env=env, **kwargs)
