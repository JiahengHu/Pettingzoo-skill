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

	def reset(self, seed=None):
		observations, infos = self._env.reset(seed)
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
		vectors = np.linspace(0.05, 1.8, 11)
		possible_vectors = [[v] for v in vectors]
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

class DownstreamCentralizedWrapper(CentralizedWrapper):
	"""
	Centralized wrapper that is responsible for downstream tasks
	Takes in a list of landmark ids that are used to generate reward
	"""
	def __init__(self, env, landmark_id, N, factorize):
		self._env = env
		self.N = N
		self.factorize = factorize
		self.distance_threshold = 0.6
		# We want to have binary indicator for each episode / each timestep
		# close or far from the landmark
		self.landmark_id = landmark_id

		self.initialize_parameters()
		self.initialize_action_space()
		self.initialize_state_space()

	def initialize_parameters(self):
		self.cycle_step = 50
		self.agent_name = self._env.possible_agents[0]
		assert self._env.unwrapped.local_ratio == 0, "local_ratio must be 0"

	def initialize_action_space(self):
		dict_act_space = self._env.action_spaces
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

	def initialize_state_space(self):
		state_dim = self.N * 8 + 1 # We have an additional indicator variable, plus time counter
		self.observation_space = spaces.Box(
			low=-np.float32(np.inf),
			high=+np.float32(np.inf),
			shape=(state_dim,),
			dtype=np.float32,
		)

	def step(self, action):
		self.step_count += 1.0
		# Loop through each agent and assign action
		# We assume each agent has the same action space
		actions = np.split(action, len(self._env.agents))
		actions = {agent:act  for agent, act in zip(self._env.agents, actions)}
		observations, rewards, terminations, truncations, infos = self._env.step(actions)

		done = terminations[self.agent_name] or truncations[self.agent_name]

		state = self._env.state()
		reward = self.get_reward(state)

		if self.step_count % self.cycle_step == 0:
			self.ds_state_update()

		return state, reward, done, infos

	def reset(self, seed=None):
		self._env.reset(seed)
		self.step_count = 0.0
		self.downstream_reset()
		return self._env.state()

	def get_reward(self, state):
		dist_list = state[:self.N]
		reward = np.zeros_like(self.landmark_id, dtype=np.float32)
		for idx, ids in enumerate(self.landmark_id):
			binary = self.binary_indicator[ids]
			dist = dist_list[ids]
			if binary == 0:
				if dist < self.distance_threshold:
					reward[idx] += 1
				else:
					reward[idx] -= 1
			else:
				if dist > self.distance_threshold:
					reward[idx] += 1
				else:
					reward[idx] -= 1
		if not self.factorize:
			reward = np.sum(reward)
		return reward

	def ds_state_update(self):
		self.binary_indicator = np.random.randint(2, size=10)

	def downstream_reset(self):
		self.binary_indicator = np.random.randint(2, size=10)

	# Defines additional states needed for the upper policy
	def get_additional_states(self):
		return np.concatenate([self.binary_indicator, [self.step_count / self.cycle_step]])


# Skip agent 5 and 8
class SequentialDSWrapper(DownstreamCentralizedWrapper):
	def __init__(self, env, N, agent_sequence=[0, 1, 2]):
		self._env = env
		self.N = N
		self.distance_threshold = 0.6

		self.agent_sequence = agent_sequence

		self.initialize_parameters()
		self.initialize_action_space()
		self.initialize_state_space()

	def initialize_state_space(self):
		state_dim = self.N * 8 + 1
		self.observation_space = spaces.Box(
			low=-np.float32(np.inf),
			high=+np.float32(np.inf),
			shape=(state_dim,),
			dtype=np.float32,
		)

	def get_reward(self, state):
		if self.progress_idx == len(self.agent_sequence):
			reward = 10
		else:
			dist_list = state[:self.N]
			reward = 0
			for idx in range(self.N):
				if idx in [5, 8]:
					continue
				binary = self.curren_idx[idx]
				dist = dist_list[idx]
				if binary == 0:
					if dist > self.distance_threshold:
						reward += 0
					else:
						reward -= 1
				else:
					if dist < self.distance_threshold:
						reward += 0
						self.charge_counter += 1
					else:
						reward -= 1
		return reward

	def ds_state_update(self):
		if self.progress_idx < len(self.agent_sequence) and self.charge_counter > 30:
			# switch to next target
			self.progress_idx += 1
			self.charge_counter = 0
			self.curren_idx = np.zeros(self.N)
			if self.progress_idx < len(self.agent_sequence):
				self.curren_idx[self.agent_sequence[self.progress_idx]] = 1


	def downstream_reset(self):
		self.progress_idx = 0
		self.charge_counter = 0
		self.curren_idx = np.zeros(self.N)
		self.curren_idx[self.agent_sequence[self.progress_idx]] = 1

	# Defines additional states needed for the upper policy
	def get_additional_states(self):
		return np.concatenate([self.curren_idx, [self.step_count / self.cycle_step]])