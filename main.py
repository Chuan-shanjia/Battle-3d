from env.battle2d import Combat
from models.ippo import PPO
import time

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    grid_size = (15, 15)
    n_agent = 2

    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = 100000
    hidden_dim = 64
    gamma = 0.99
    lmbda = 0.97
    eps = 0.2


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = Combat(grid_shape=grid_size, n_agents=n_agent, n_opponents=n_agent)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    # 两个智能体共享同一个策略
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps,
                gamma, device)
    # agent.actor.load_state_dict(torch.load("outdir/models/actor_9.pth"), strict=False)
    # agent.critic.load_state_dict(torch.load("outdir/models/critic_9.pth"), strict=False)

    win_list = []
    start = time.time()
    for j in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % j) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                transition_dict = [[] for _ in range(n_agent)]
                for i  in range(n_agent):
                    transition_dict[i] = {
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'rewards': [],
                        'dones': []
                    }

                s = env.reset()
                terminal = False
                while not terminal:
                    a = [[] for _ in range(n_agent)]
                    for i in range(n_agent):
                        a[i] = agent.take_action(s[i])

                    next_s, r, done, info = env.step(a)

                    for i in range(n_agent):
                        transition_dict[i]['states'].append(s[i])
                        transition_dict[i]['actions'].append(a[i])
                        transition_dict[i]['next_states'].append(next_s[i])
                        transition_dict[i]['rewards'].append(
                            r[i] + 100 if info['win'] else r[i] - 0.1)
                        transition_dict[i]['dones'].append(False)

                    s = next_s
                    terminal = all(done)
                win_list.append(1 if info["win"] else 0)

                for i in range(n_agent):
                    agent.update(transition_dict[i])

                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * j + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(win_list[-100:])
                    })

                pbar.update(1)

        agent.save(j)

        # agent.save(j)
    end = time.time()
    training_time = end - start
    print('time cost:',training_time,'s')

    win_array = np.array(win_list)
    # 每100条轨迹取一次平均
    win_array = np.mean(win_array.reshape(-1, 100), axis=1)

    episodes_list = np.arange(win_array.shape[0]) * 100
    plt.plot(episodes_list, win_array)
    plt.xlabel('Episodes')
    plt.ylabel('Win rate')
    plt.title('IPPO on Combat')
    plt.show()
