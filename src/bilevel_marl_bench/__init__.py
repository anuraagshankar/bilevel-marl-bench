"""Bilevel MARL benchmark. Every environment is a PettingZoo ParallelEnv."""

ENVS = ["combined_arms", "battlefield", "football", "paintball"]


def make(name, **kwargs):
    if name == "combined_arms":
        from .combined_arms import CombinedArms

        return CombinedArms(**kwargs)
    if name == "battlefield":
        from .battlefield import Battlefield

        return Battlefield(**kwargs)
    if name == "football":
        from .football import FootballScenario
        from .vmas_parallel import VmasParallelEnv

        return VmasParallelEnv(FootballScenario(), title="Football", **kwargs)
    if name == "paintball":
        from shimmy import MeltingPotCompatibilityV0

        variant = kwargs.pop("variant", "capture_the_flag")  # or "king_of_the_hill"
        return MeltingPotCompatibilityV0(substrate_name=f"paintball__{variant}", **kwargs)
    raise ValueError(f"Unknown env '{name}'. Available: {ENVS}")
