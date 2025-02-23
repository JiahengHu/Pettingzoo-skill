import numpy as np
from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper, DownstreamCentralizedWrapper
import h5py
import minari

def img_encoder(img):
    return np.zeros(50)

parallel_env = simple_heterogenous_v3.parallel_env(
    N=10,
    render_mode='rgb_array', # 'human' or 'rgb_array'， rgb_array is just for getting obs
    max_cycles=1000,
    continuous_actions=True,
    local_ratio=0,
    # img_encoder=img_encoder,
)

parallel_env = CentralizedWrapper(parallel_env)
# pa2 = DownstreamCentralizedWrapper(parallel_env, [1], 10, False)

collect_obs = True
use_minari = True

if use_minari:
    dataset = None
    from minari import DataCollector
    env = DataCollector(parallel_env)
    for episode_id in range(1000):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # <- use your policy here
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if (episode_id + 1) % 10 == 0:
            # Update local Minari dataset every 10 episodes.
            # This works as a checkpoint to not lose the already collected data
            if dataset is None:
                dataset = env.create_dataset("data/test-v0")
            else:
                env.add_to_dataset(dataset)

elif collect_obs:
    obss_list = []
    imgs_list = []
    actions_list = []
    num_eps = 1000

    current_eps_count = 0
    while current_eps_count < num_eps:
        obss = []
        imgs = []
        actions = []

        observations = parallel_env.reset()
        dones = False
        img = parallel_env.render()
        obss.append(observations)
        imgs.append(img)
        while not dones:
            actions = parallel_env.action_space.sample()
            observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
            dones = terminations or truncations
            img = parallel_env.render()
            obss.append(observations)
            imgs.append(img)
            # import ipdb; ipdb.set_trace()
        actions_list.append(actions)
        imgs_list.append(imgs)
        obss_list.append(obss)
        current_eps_count += 1

        # TODO: use minari; change to gymnasium
        if current_eps_count % 10 == 0:
            print(current_eps_count / num_eps * 100, '%')
            obss_list = np.array(obss_list)
            imgs_list = np.array(imgs_list)
            actions_list = np.array(actions_list)
            np.save(f'data/obs{current_eps_count}.npy', obss_list)
            np.save(f'data/img{current_eps_count}.npy', imgs_list)
            np.save(f'data/actions{current_eps_count}.npy', actions_list)
            import ipdb; ipdb.set_trace()
            with h5py.File(f'data/img{current_eps_count}.h5', 'w') as hf:
                hf.create_dataset("img", data=imgs_list)

            obss_list = []
            imgs_list = []
            actions_list = []

else:
    observations = parallel_env.reset()
    dones = False
    counter = 0
    while not dones:
        counter += 1
        actions = parallel_env.action_space.sample()
        # import ipdb; ipdb.set_trace()

        observations, rewards, dones, infos = parallel_env.step(actions)
        img = parallel_env.render()
        print(observations)
        print(parallel_env.state_space)

parallel_env.close()
