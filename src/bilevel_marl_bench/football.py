"""VMAS football with configurable player roles for both teams."""

import math

import torch
from vmas.scenarios.football import Scenario
from vmas.simulator.core import Agent, Sphere
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from vmas.simulator.utils import Color

# Offsets from VMAS's standard player: (radius, u_multiplier, max_speed, shoot_multiplier).
# Values from VMAS's physically different roles. Fix: upstream adds the goalkeeper's
# +0.2 shot boost to the rotation multiplier instead of the shoot multiplier.
ROLES = {
    "gk": (0.01, -0.05, -0.1, 0.2),
    "def": (0.0, 0.0, 0.0, 0.0),
    "fwd": (-0.005, 0.1, 0.05, -0.2),
}

# Spawn zones: each half split into 3 zones by depth from the centre line (fraction of the half).
# Goalkeepers spawn in the D instead: a semicircle in front of goal, same radius as the centre circle.
ZONES = {"fwd": (0, 1 / 3), "def": (1 / 3, 2 / 3)}


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

    def reset_agents(self, env_index=None):
        if self.spawn_in_formation:
            return super().reset_agents(env_index)
        n = 1 if env_index is not None else self.world.batch_dim
        dev = self.world.device
        half = self.pitch_length / 2 - self.agent_size  # centre line to goal line

        def rand():
            return torch.rand(n, device=dev)

        for agents, roles, sign in (
            (self.blue_agents, self.blue_roles, -1),
            (self.red_agents, self.red_roles, 1),
        ):
            for agent, role in zip(agents, roles):
                if role == "gk":  # uniform in the D, kept off the goal line by the keeper's radius
                    rad = agent.shape.radius
                    r = (self.goal_size / 2 - rad) * rand().sqrt()
                    t = (rand() - 0.5) * torch.pi
                    depth = half - rad - r * t.cos()
                    y = r * t.sin()
                else:  # uniform in the role's zone
                    lo, hi = ZONES[role]
                    depth = half * (lo + (hi - lo) * rand())
                    y = (rand() - 0.5) * (self.pitch_width - 2 * (self.agent_size + agent.shape.radius))
                agent.set_pos(torch.stack([sign * depth, y], dim=-1), batch_index=env_index)
                if sign == 1:
                    agent.set_rot(torch.tensor([torch.pi], device=dev), batch_index=env_index)

    def extra_render(self, env_index=0):
        from vmas.simulator import rendering

        # The original draws shooting fans for blue only (draw them for every shooting player)
        # and numbers for blue only (hide them; role labels are drawn below instead).
        blue, red = self.blue_agents, self.red_agents
        self.blue_agents = [a for a in blue + red if a.action_script is None]
        self.red_agents = blue + red
        try:
            geoms = super().extra_render(env_index)
        finally:
            self.blue_agents, self.red_agents = blue, red

        # Role labels, positioned like VMAS's number labels.
        w, h = self.viewer_size
        rx, ry = (w / h, 1) if w > h else (1, h / w)
        for agent in blue + red:
            x, y = agent.state.pos[env_index].tolist()
            geoms.append(
                rendering.TextLine(
                    text=agent.role.upper(),
                    font_size=15,
                    x=x * w / (self.viewer_zoom**2 * rx * 2) + w / 2,
                    y=y * h / (self.viewer_zoom**2 * ry * 2) + h / 2,
                )
            )

        # The D: a half ring in front of each goal, styled like the centre circle.
        r_out, r_in = self.goal_size / 2, self.goal_size / 2 - 0.02
        x0 = self.pitch_length / 2 - self.agent_size
        angles = [-math.pi / 2 + math.pi * i / 30 for i in range(31)]
        d_geoms = []
        for sign in (-1, 1):
            for a, b in zip(angles, angles[1:]):
                pts = [(r_out, a), (r_out, b), (r_in, b), (r_in, a)]
                quad = rendering.make_polygon(
                    [(sign * (x0 - r * math.cos(t)), r * math.sin(t)) for r, t in pts],
                    draw_border=False,
                )
                quad.set_color(*Color.WHITE.value)
                d_geoms.append(quad)
        # Insert right after the pitch background, so fans and labels are drawn on top.
        n_background = len(self.background_entities[0 if self._render_field else 3 :])
        geoms[n_background:n_background] = d_geoms
        return geoms

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
        agent.role = role
        world.add_agent(agent)
        return agent
