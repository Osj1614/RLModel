import gym
from rl.algo.ppo import PPO
import os
import torch as th
import time
import numpy as np
from gym.spaces import Box

from rl.module.actor_critic import ActorCriticNetwork
from rl.module.general import DummyExtractor

from rl.vecenv import vecenv
import metaworld
from rl.metaenv_wrapper import MetaWrapper

import torch.utils.tensorboard

#, critic_coef=1, ent_coef=0.01, pred_coef=0.0, gamma=0.99, epsilon=0.2, lamda=0.95, policy_epochs=4

alg_args = {
    'name' : 'lunar',
    'nenvs' : 16,
    'nsteps' : 256,
    'gamma' : 0.99,
    'train_steps' : 1e6,
    'minibatches' : 256,
    'lr' : 3e-4,
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
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    istrain = True
    
    env_name = 'LunarLanderContinuous-v2'

    env = gym.make(env_name)

    extractor = DummyExtractor(env.observation_space.shape[0])


    if isinstance(env.action_space, Box):
        action_type = 'continuous'
        network = ActorCriticNetwork(action_type, env.action_space.shape[0], extractor)
    else:
        action_type = 'discrete'
        network = ActorCriticNetwork(action_type, env.action_space.n, extractor)

    

    envs = vecenv(alg_args['nenvs'], env_name, {})

    model = PPO(device, network, **alg_args)
    
    if istrain:
        train(model, envs, save_interval=1)
    else:
        #env = Monitor(env, f'./logs/{model.name}/video', force=True)
        state = env.reset()
        done = False
        while not done:
            action = model.model.get_action(th.as_tensor(state[None], dtype=th.float32).to(device))
            state, reward, done, info = env.step(action[0])

        env.close()
        