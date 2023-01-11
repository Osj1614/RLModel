import gym
from rl.algo.sac import SAC
import os
import torch as th
import time
import numpy as np
from gym.spaces import Box
from rl.module.sac_networks import SACNet, TransSACNet, TransSACNet2, TransSACNet3
from rl.module.general import DummyExtractor, TaskExtractor
import metaworld
from rl.metaenv_wrapper import MetaWrapper

import torch.utils.tensorboard

#, critic_coef=1, ent_coef=0.01, pred_coef=0.0, gamma=0.99, epsilon=0.2, lamda=0.95, policy_epochs=4

alg_args = {
    'name' : 'transsac_v3_2',
    'nenvs' : 10,
    'n_steps' : 500,
    'gradient_steps' : 500,
    'gamma' : 0.99,
    'capacity' : 500000,
    'train_steps' : 20e6,
    'batch_size' : 1024,
    'train_start' : 1500,
    'lr' : 3e-4,
    'target_polyak' : 0.005,

}


def train(model, env, log_interval=0.01, save_interval=0.1):
    prevtime = time.time()
    saves = 1
    
    next_log = log_interval
    next_save = save_interval
    while next_log <= model.progress:
        next_log += log_interval
    while next_save <= model.progress:
        next_save += save_interval


    while os.path.isdir(f'logs/{model.name}/valid/{saves}'):
        saves += 1

    while model.progress < 1:
        model.train_epoch(env)
        if model.progress >= next_log:
            next_log += log_interval
            avg = np.mean(env.reward_sums)
            success = np.mean(env.success_list)
            
            print(f"Average score:\t{round(avg,3)}")
            print(f"Success rate:\t{round(success,3)}")
            print(f"progress:\t{round(model.progress * 100, 2)}%")
            currtime = time.time()
            time_passed = currtime - prevtime
            print(f"elapsed time:\t{round(time_passed, 3)} second")
            print(f"time left:\t{round(time_passed*(1-model.progress)/log_interval/3600, 3)} hour")
            prevtime = currtime
            model.write_log('Average_score', avg)
            model.write_log('Success_rate', success)
            print('-----------------------------------------------------------')


if __name__ == '__main__':
    device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')
    istrain = True
    
    mt10 = metaworld.MT10()

    envs = MetaWrapper(mt10.train_classes, mt10.train_tasks, nenvs=alg_args['nenvs'], env_sample='uniform_random')
    envs.reset()

    extractor = TaskExtractor(envs.observation_space.shape[0], envs.n_task)
    network = TransSACNet3(extractor, envs.action_space.shape[0])

    model = SAC(device, network, **alg_args)
    if istrain:
        train(model, envs, save_interval=0.1)
    else:
        #env = Monitor(env, f'./logs/{model.name}/video', force=True)
        state = env.reset()
        done = False
        while not done:
            action = model.model.get_action(th.as_tensor(state[None], dtype=th.float32).to(device)) 

            state, reward, done, info = env.step(action[0])

        env.close()