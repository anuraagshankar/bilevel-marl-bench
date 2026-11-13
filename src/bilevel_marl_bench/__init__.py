"""Bilevel MARL benchmark. Every environment is a PettingZoo ParallelEnv."""

ENVS = ["combined_arms", "battlefield", "football", "capture_the_flag", "king_of_the_hill"]


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
    if name in ("capture_the_flag", "king_of_the_hill"):
        from .paintball import Paintball

        return Paintball(name, **kwargs)
    raise ValueError(f"Unknown env '{name}'. Available: {ENVS}")
