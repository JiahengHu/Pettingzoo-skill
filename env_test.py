from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper

parallel_env = simple_heterogenous_v3.parallel_env(
    render_mode='human',
    max_cycles=1000,
    continuous_actions=True,
    local_ratio=0
)

parallel_env = CentralizedWrapper(parallel_env)

observations, infos = parallel_env.reset()
dones = False
counter = 0
while not dones:
    counter += 1
    actions = parallel_env.action_space.sample()
    observations, rewards, dones, infos = parallel_env.step(actions)
    print(observations)

parallel_env.close()
