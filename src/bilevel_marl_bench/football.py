"""VMAS football with configurable player roles for both teams."""

import torch
from vmas.scenarios.football import Scenario
from vmas.simulator.core import Agent, Sphere
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation

# Offsets from VMAS's standard player: (radius, u_multiplier, max_speed, shoot_multiplier).
# Values from VMAS's physically different roles. Fix: upstream adds the goalkeeper's
# +0.2 shot boost to the rotation multiplier instead of the shoot multiplier.
ROLES = {
    "gk": (0.01, -0.05, -0.1, 0.2),
    "def": (0.0, 0.0, 0.0, 0.0),
    "fwd": (-0.005, 0.1, 0.05, -0.2),
}


class FootballScenario(Scenario):
    def init_params(self, **kwargs):
        # Roles as {role: count}, expanded to one role per player.
        default = {"gk": 1, "def": 1, "fwd": 1}
        self.blue_roles = [r for r, n in kwargs.pop("blue_roles", default).items() for _ in range(n)]
        self.red_roles = [r for r, n in kwargs.pop("red_roles", default).items() for _ in range(n)]
        kwargs["n_blue_agents"] = len(self.blue_roles)
        kwargs["n_red_agents"] = len(self.red_roles)
        kwargs.setdefault("physically_different", True)
        kwargs.setdefault("enable_shooting", True)
        kwargs.setdefault("ai_red_agents", False)
        super().init_params(**kwargs)

    def init_agents(self, world):
        # Run the original with no players: it only sets colours and AI controllers.
        n_blue, n_red, pd = self.n_blue_agents, self.n_red_agents, self.physically_different
        self.n_blue_agents = self.n_red_agents = 0
        self.physically_different = False
        super().init_agents(world)
        self.n_blue_agents, self.n_red_agents, self.physically_different = n_blue, n_red, pd

        self.blue_agents = world.blue_agents = [
            self._add_agent(world, "blue", i, r) for i, r in enumerate(self.blue_roles)
        ]
        self.red_agents = world.red_agents = [
            self._add_agent(world, "red", i, r) for i, r in enumerate(self.red_roles)
        ]

    def extra_render(self, env_index=0):
        # The original draws shooting fans and numbers for blue only; draw them for both teams.
        blue, red = self.blue_agents, self.red_agents
        self.blue_agents = [a for a in blue + red if a.action_script is None]
        self.red_agents = []
        try:
            return super().extra_render(env_index)
        finally:
            self.blue_agents, self.red_agents = blue, red

    def _add_agent(self, world, team, i, role):
        blue = team == "blue"
        ai = self.ai_blue_agents if blue else self.ai_red_agents
        controller = self.blue_controller if blue else self.red_controller
        shooting = self.enable_shooting and not ai
        dr, du, dv, ds = ROLES[role] if self.physically_different else ROLES["def"]
        u = self.u_multiplier + du
        agent = Agent(
            name=f"agent_{team}_{i}",
            shape=Sphere(radius=self.agent_size + dr),
            action_script=controller.run if ai else None,
            u_multiplier=[u, u, self.u_rot_multiplier, self.u_shoot_multiplier + ds]
            if shooting
            else [u, u],
            max_speed=self.max_speed + dv,
            dynamics=HolonomicWithRotation() if shooting else Holonomic(),
            action_size=4 if shooting else 2,
            color=self.blue_color if blue else self.red_color,
            alpha=1,
        )
        agent.ball_within_angle = torch.zeros(world.batch_dim, device=world.device, dtype=torch.bool)
        agent.ball_within_range = torch.zeros(world.batch_dim, device=world.device, dtype=torch.bool)
        agent.shoot_force = torch.zeros(world.batch_dim, 2, device=world.device)
        world.add_agent(agent)
        return agent
