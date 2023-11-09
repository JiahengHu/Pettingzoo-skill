# noqa: D212, D415
"""
Modified from simple_spread.py
"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym import spaces


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_heterogenous_v3"

        state_dim = N * 9
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

    def observe(self, agent):
        return None

    def state(self):
        return self.scenario.get_state(self.world).astype(np.float32)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class clipWorld(World):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        super().step()
        for agent in self.agents:
            agent.state.p_pos = np.clip(
                agent.state.p_pos, -1, 1
            )  # clip position

class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = clipWorld()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.color = np.array([0.75, 0.75, 0.75])
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([1, 0, 0]) * i / len(world.landmarks)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.color = np.array([0, 0, 1]) * i / len(world.landmarks)

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # TODO: What if we want to fix the initial state?
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # In this env, we do not make use of local reward

        rew = 0
        # if agent.collide:
        #     for a in world.agents:
        #         rew -= 1.0 * (self.is_collision(a, agent) and a != agent)
        return rew

    def global_reward(self, world):
        rew = 0
        # for idx, lm in enumerate(world.landmarks):
        #     dists = [
        #         np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
        #         for a in world.agents
        #     ]
        #     rew -= min(dists)
        return rew

    def get_state(self, world):
        pos = []
        vel = []

        for agt in world.agents:
            vel.append(agt.state.p_vel)
            pos.append(agt.state.p_pos)

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)

        diayn_states = []
        thresholds = [0.3, 0.6]
        for idx, lm in enumerate(world.landmarks):
            other_idx = (idx + 1) % len(world.landmarks)
            ag_idx_list = [idx, other_idx] # We cam change the association here

            dists = [
                np.sqrt(np.sum(np.square(world.agents[a_i].state.p_pos - lm.state.p_pos)))
                for a_i in ag_idx_list
            ]
            m_dist = min(dists)
            if m_dist < thresholds[0]:
                diayn_states.append([1, 0, 0])
            elif m_dist < thresholds[1]:
                diayn_states.append([0, 1, 0])
            else:
                diayn_states.append([0, 0, 1])

        return np.concatenate(diayn_states + pos + vel + entity_pos)

    def observation(self, agent, world):
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + other_pos + comm
        )
