"""MAgent2 battlefield (walls) with combined arms' melee/ranged units."""

from .combined_arms import CombinedArms


class Battlefield(CombinedArms):
    metadata = {**CombinedArms.metadata, "name": "battlefield"}

    def __init__(self, red_melee=6, red_ranged=6, blue_melee=6, blue_ranged=6, map_size=80, **kwargs):
        super().__init__(
            red_melee=red_melee, red_ranged=red_ranged, blue_melee=blue_melee,
            blue_ranged=blue_ranged, map_size=map_size, **kwargs,
        )

    def generate_map(self):
        # Wall layout copied from MAgent2 battlefield_v5.
        w = h = self.map_size
        walls = []
        for y in [*range(10, 45), *range(50, h // 2 + 25)]:
            walls += [(w // 2 - 5, y), (w // 2 - 4, y)]
        for y in [*range(h // 2 - 25, h - 50), *range(h - 45, h - 10)]:
            walls += [(w // 2 + 5, y), (w // 2 + 4, y)]
        self.env.add_walls(pos=walls, method="custom")
        self.walls = set(walls)
        super().generate_map()
