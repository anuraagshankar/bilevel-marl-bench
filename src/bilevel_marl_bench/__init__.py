"""Bilevel MARL benchmark. Every environment is a PettingZoo ParallelEnv."""

ENVS = ["combined_arms", "battlefield", "football", "paintball"]


def make(name, **kwargs):
    if name == "combined_arms":
        from magent2.environments import combined_arms_v6

        return combined_arms_v6.parallel_env(**kwargs)
    if name == "battlefield":
        from magent2.environments import battlefield_v5

        return battlefield_v5.parallel_env(**kwargs)
    if name == "football":
        from .vmas_parallel import VmasParallelEnv

        return VmasParallelEnv("football", **kwargs)
    if name == "paintball":
        from shimmy import MeltingPotCompatibilityV0

        variant = kwargs.pop("variant", "capture_the_flag")  # or "king_of_the_hill"
        return MeltingPotCompatibilityV0(substrate_name=f"paintball__{variant}", **kwargs)
    raise ValueError(f"Unknown env '{name}'. Available: {ENVS}")
