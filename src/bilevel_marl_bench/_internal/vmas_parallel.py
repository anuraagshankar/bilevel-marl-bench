"""PettingZoo ParallelEnv adapter for a single (non-vectorized) VMAS scenario."""

import vmas
from pettingzoo import ParallelEnv


class VmasParallelEnv(ParallelEnv):
    metadata = {"name": "vmas", "render_modes": ["human", "rgb_array"]}

    def __init__(self, scenario, title="VMAS", max_steps=None, render_mode=None, **kwargs):
        self.title = title
        self.render_mode = render_mode
        self._env = vmas.make_env(
            scenario,
            num_envs=1,
            wrapper="gymnasium",
            wrapper_kwargs={"render_mode": render_mode or "rgb_array"},
            dict_spaces=True,
            terminated_truncated=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.possible_agents = [a.name for a in self._env.unwrapped.agents]
        self.observation_spaces = dict(self._env.observation_space.spaces)
        self.action_spaces = dict(self._env.action_space.spaces)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        return self._env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, rewards, term, trunc, infos = self._env.step(
            [actions[a] for a in self.possible_agents]
        )
        terminations = {a: term for a in self.agents}
        truncations = {a: trunc for a in self.agents}
        if term or trunc:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return obs, rewards, terminations, truncations, infos

    def render(self):
        frame = self._env.render()
        if self.render_mode == "human":  # VMAS creates the window on first render, untitled
            self._env.unwrapped.viewer.window.set_caption(self.title)
        return frame

    def close(self):
        self._env.close()
