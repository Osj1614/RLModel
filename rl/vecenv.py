import numpy as np
import gym
import random
from collections import deque
from threading import Thread

class vecenv:
    def __init__(self, nenvs, env_name, env_args):
        self.nenvs = nenvs
        self.envs = [gym.make(env_name, **env_args) for _ in range(self.nenvs)]
        self.recent_scores = deque(maxlen=100)
        self.reset()


    def reset(self):
        self.states = []
        self.total_reward = [0 for _ in range(self.nenvs)]
        for i in range(self.nenvs):
            s, _ = self.envs[i].reset()
            self.states.append(s)
        return np.array(self.states)


    def step(self, actions):
        rewards = [None for _ in range(self.nenvs)]
        dones = [None for _ in range(self.nenvs)]

        def step_per_env(i):
            ns, reward, done, _, _ = self.envs[i].step(actions[i])
            self.states[i] = ns
            self.total_reward[i] += reward
            rewards[i] = reward
            dones[i] = done
            if done:
                self.recent_scores.append(self.total_reward[i])
                self.total_reward[i] = 0
                self.states[i], _ = self.envs[i].reset()
        
        threads = []
        for i in range(self.nenvs):
            th = Thread(target=step_per_env, args=(i,))
            th.start()
            threads.append(th)
        
        for th in threads:
            th.join()

        return np.array(self.states), np.array(rewards), np.array(dones), []

    def close(self):
        for env in self.envs:
            env.close()