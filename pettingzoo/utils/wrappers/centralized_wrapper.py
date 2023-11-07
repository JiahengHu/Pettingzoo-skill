"""
Convert mpe to gym environment
"""

import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np
import gymnasium

class CentralizedWrapper(gym.Env):
	def __init__(self, env):
		self._env = env

		dict_act_space = env.action_spaces
		low_action_range = []
		high_action_range = []

		for val in dict_act_space.values():
			assert isinstance(val, gymnasium.spaces.Box)
			low_action_range.append(val.low)
			high_action_range.append(val.high)
		low_action_range = np.concatenate(low_action_range)
		high_action_range = np.concatenate(high_action_range)

		self.action_space = spaces.Box(
			low=low_action_range, high=high_action_range, shape=low_action_range.shape, dtype=np.float32)
		self.observation_space = env.state_space
		self.agent_name = self._env.possible_agents[0]
		assert self._env.unwrapped.local_ratio == 0, "local_ratio must be 0"

	def reset(self):
		observations, infos = self._env.reset()
		return self._env.state()

	def step(self, action):
		# Loop through each agent and assign action
		# We assume each agent has the same action space
		actions = np.split(action, len(self._env.agents))
		actions = {agent:act  for agent, act in zip(self._env.agents, actions)}
		observations, rewards, terminations, truncations, infos = self._env.step(actions)

		done = terminations[self.agent_name] or truncations[self.agent_name]
		rewards = rewards[self.agent_name]
		return self._env.state(), rewards, done, infos

	def __getattr__(self, name):
		return getattr(self._env, name)