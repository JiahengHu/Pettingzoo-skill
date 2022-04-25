import sys
from copy import deepcopy

import numpy as np
import pytest
from gym import spaces

from pettingzoo.utils import wrappers

from .all_modules import all_environments


def box_action(env, agents):
    boxable = True
    for agent in agents:
        boxable = boxable and isinstance(env.action_space(agent), spaces.Box)
    return boxable


def box_observation(env, agents):
    boxable = True
    for agent in agents:
        boxable = boxable and isinstance(env.observation_space(agent), spaces.Box)
    return boxable


def discrete_observation(env, agents):
    is_discrete = True
    for agent in agents:
        is_discrete = is_discrete and (
            isinstance(env.observation_space(agent), spaces.Discrete)
        )
    return is_discrete


@pytest.mark.parametrize(("name", "env_module"), list(all_environments.items()))
def test_unwrapped(_, env_module):

    env = env_module.env()
    base_env = env.unwrapped

    env.reset()
    agents = env.agents

    if discrete_observation(env, agents):
        env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.BaseWrapper(env)
    env = wrappers.CaptureStdoutWrapper(env)
    if box_observation(env, agents) and box_action(env, agents):
        env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, 1.0)

    assert env.unwrapped == base_env, "Unwrapped Test: unequal envs"
