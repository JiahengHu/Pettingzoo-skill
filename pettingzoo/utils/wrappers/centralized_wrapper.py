"""
Convert mpe to gym environment
"""

import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np
import gymnasium
import torch

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

	def render(self, mode='human'):
		return self._env.render()


	# THIS IS A HACK
	def plot_prediction_net(self, agent, cfg, step=0, device="cuda", anti=False, SHOW=False):
		assert agent.domain == "particle"
		N = cfg.env.particle.N
		# So we want to test: for each vector, what are the predicted skill
		possible_vectors = [
			[0.0],
			[0.2],
			[0.35],
			[0.5],
			[0.7],
		]
		prediction = []
		for vec in possible_vectors:
			with torch.no_grad():
				test_obs = torch.tensor(vec*N, device=device, dtype=torch.float32)
				if anti:
					predicted_z = torch.softmax(agent.anti_diayn(None, torch.tensor(test_obs, device=device)).
												reshape(cfg.agent.skill_channel, -1),
												dim=-1)
				else:
					predicted_z = torch.softmax(agent.diayn(None, torch.tensor(test_obs, device=device)).
												reshape(cfg.agent.skill_channel, -1),
												dim=-1)
			prediction.append(predicted_z.cpu().numpy())

		apd = f"_step_{step}"
		if anti:
			apd += "_anti"

		text = np.array2string(np.array(prediction), precision=2, suppress_small=True)
		with open(f"pred{apd}.txt", "w") as text_file:
			text_file.write(text)

	def __getattr__(self, name):
		return getattr(self._env, name)