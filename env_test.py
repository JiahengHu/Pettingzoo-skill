import numpy as np
from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper

parallel_env = simple_heterogenous_v3.parallel_env(
    N=10,
    render_mode='rgb_array', # 'human' or 'rgb_array'
    max_cycles=1000,
    continuous_actions=True,
    local_ratio=0
)

parallel_env = CentralizedWrapper(parallel_env)

collect_obs = True
if collect_obs:
    obss = []
    imgs = []
    num_data = 1000000
    for i in range(num_data):
        observations = parallel_env.reset()
        img = parallel_env.render()
        obss.append(observations)
        imgs.append(img)

        vis = False
        if vis:
            import ipdb; ipdb.set_trace()
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        # N * 7: np.concatenate([diayn_states] + agent_stats + entity_pos); agent stats: speed + pos

        if i % 10000 == 0:
            print(i / num_data * 100, '%')
            obss = np.array(obss)
            imgs = np.array(imgs)
            np.save(f'data/obs{i}.npy', obss)
            np.save(f'data/img{i}.npy', imgs)
            obss = []
            imgs = []

else:
    observations = parallel_env.reset()
    dones = False
    counter = 0
    while not dones:
        counter += 1
        actions = parallel_env.action_space.sample()
        observations, rewards, dones, infos = parallel_env.step(actions)
        img = parallel_env.render()

parallel_env.close()
