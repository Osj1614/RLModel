import random
import numpy as np
from gym.spaces import Box
from collections import deque

class MetaWrapper:
    def __init__(self, env_classes, env_parametric_tasks, nenvs, task_embedding=True, env_sample='uniform_random', epi_max=500):
        self.nenvs = nenvs
        self.epi_max = epi_max
        self.env_sample = env_sample
        self.task_embedding = task_embedding
        self.normalize_input = True
        self.n_task = len(env_classes)
        
        self.task_embeddings = np.eye(self.n_task, dtype=np.float32)
        
        self.task_envs = []
        self.task_parametric_tasks = []
        
        while len(self.task_envs) < self.nenvs:
            for name, env_cls in env_classes.items():
                self.task_envs.append(env_cls())
                self.task_parametric_tasks.append([task for task in env_parametric_tasks if task.env_name == name])
        
        self.envs = [None for _ in range(nenvs)]
        self.env_steps = [0 for _ in range(nenvs)]
        self.states = []
        self.env_idx = 0
        self.rew_list = [0 for _ in range(nenvs)]
        self.reward_sums = deque(maxlen=500)
        self.success_list = deque(maxlen=500)

        self.action_space = self.task_envs[0].action_space
        if self.task_embedding:
            obs_space = self.task_envs[0].observation_space
            low = np.concatenate((obs_space.low, np.zeros(self.n_task)))
            high = np.concatenate((obs_space.high, np.ones(self.n_task)))
            self.observation_space = Box(low=low, high=high)
        else:
            self.observation_space = self.task_envs[0].observation_space

    def reset_env_task(self, i):
        if self.env_sample == 'uniform_random':
            self.envs[i] = -1
            env_idx = np.random.choice([i for i in range(self.n_task) if i not in self.envs])
            self.envs[i] = env_idx
        elif self.env_sample == 'round_robin':
            self.envs[i] = self.env_idx
            self.env_idx = (self.env_idx + 1) % self.nenvs
        
        rand_parametric_task = np.random.randint(len(self.task_parametric_tasks[self.envs[i]]))
        self.task_envs[self.envs[i]].set_task(self.task_parametric_tasks[self.envs[i]][rand_parametric_task])
        self.env_steps[i] = 0
        return self.task_envs[self.envs[i]].reset()


    def reset(self):
        self.states = []
        for i in range(self.nenvs):
            self.states.append(self.reset_env_task(i))

            if self.task_embedding:
                self.states[i] = np.concatenate((self.states[i], self.task_embeddings[self.envs[i]]))

        return np.array(self.states)
    

    def step(self, actions):
        rewards = [None for _ in range(self.nenvs)]
        dones = [None for _ in range(self.nenvs)]
        infos = [None for _ in range(self.nenvs)]

        for i in range(self.nenvs):
            env = self.envs[i]
            self.states[i], rewards[i], dones[i], infos[i] = self.task_envs[env].step(actions[i])
            self.rew_list[i] += rewards[i]
            self.env_steps[i] += 1
            dones[i] = self.env_steps[i] >= 500
            if dones[i]:
                self.states[i] = self.reset_env_task(i)
                self.reward_sums.append(self.rew_list[i])
                self.success_list.append(infos[i]['success'])
                self.rew_list[i] = 0

            if self.task_embedding:
                self.states[i] = np.concatenate((self.states[i], self.task_embeddings[env]))
                
        
        return np.array(self.states), np.array(rewards), np.array(dones, dtype=np.float32), infos

