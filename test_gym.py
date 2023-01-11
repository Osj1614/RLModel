import gym
from rl.ppo import PPO
from rl.maml import MAML
import os
import torch as th
import time
import numpy as np
from gym.spaces import Box
import rl.models
from rl.vecenv import vecenv

#, critic_coef=1, ent_coef=0.01, pred_coef=0.0, gamma=0.99, epsilon=0.2, lamda=0.95, policy_epochs=4


alg_args = {
    'name' : 'lunar',
    'nenvs' : 32,
    'nsteps' : 128,
    'train_steps' : 2e6,
    'lr' : 1e-4,
    'minibatches' : 16,
    'critic_coef' : 0.5,
    'ent_coef' : 0.00,
    'gamma' : 0.99,
    'epsilon' : 0.3,
    'lamda' : 0.95,
    'policy_epochs' : 10
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
            avg = np.mean(env.recent_scores)
            print(f"Average score:\t{round(avg,3)}")
            print(f"progress:\t{round(model.progress * 100, 2)}%")
            currtime = time.time()
            time_passed = currtime - prevtime
            print(f"elapsed time:\t{round(time_passed, 3)} second")
            print(f"time left:\t{round(time_passed*(1-model.progress)/log_interval/3600, 3)} hour")
            prevtime = currtime
            model.write_log('Average_score', avg)
            print('-----------------------------------------------------------')


if __name__ == '__main__':
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    istrain = True
    
    env_name = 'LunarLanderContinuous-v2'

    env = gym.make(env_name)

    extractor = rl.models.DummyExtractor(env.observation_space.shape[0])


    if isinstance(env.action_space, Box):
        action_type = 'continuous'
        network = rl.models.ACNet(action_type, env.action_space.shape[0], extractor)
    else:
        action_type = 'discrete'
        network = rl.models.ACNet(action_type, env.action_space.n, extractor)

    

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