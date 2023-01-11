import numpy as np
import torch
from collections import deque


class ReplayMemory:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    

    def append(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state)) ###
    

    def sample(self, sample_size:int=32):
        states, actions, rewards, next_states, dones = list(), list(), list(), list(), list()
        
        sample_indices = np.random.choice(len(self.buffer), size=sample_size, replace=False)

        for index in sample_indices:
            sample = self.buffer[index]
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            dones.append(sample[3])
            next_states.append(sample[4])

        states, actions, rewards, next_states, dones = \
            (torch.from_numpy(np.array(datas)).float().to(self.device) for datas in (states, actions, rewards, next_states, dones))

        return states, actions, rewards, next_states, dones


class DREP_ReplayMemory:

    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.cur_traj = list()
        self.device = device
    

    def append(self, state, action, reward, goal, next_state, done):
        self.cur_traj.append((state, action, reward, goal, next_state, done))

        if done:
            self.buffer.append(self.cur_traj.copy()) 
            self.cur_traj.clear()


    def sample(self, samples):
        states, actions, rewards, goals, relabel_goals, next_states, dones = list(), list(), list(), list(), list(), list(), list()
        
        sample_indices = np.random.choice(len(self.buffer), size=samples, replace=True)

        for index in sample_indices:
            sample_traj = self.buffer[index]

            t = np.random.randint(0,len(sample_traj)-1)
            buffer = sample_traj[t]
            relabel_goal = sample_traj[np.random.randint(t, len(sample_traj))][0]

            states.append(buffer[0])
            actions.append(buffer[1])
            rewards.append(buffer[2])
            goals.append(buffer[3])
            relabel_goals.append(relabel_goal)
            next_states.append(buffer[4])
            dones.append(buffer[5])
   
        states, actions, rewards, goals, relabel_goals, next_states, dones = \
            (torch.from_numpy(np.array(datas)).float().to(self.device) for datas in (states, actions, rewards, goals, relabel_goals, next_states, dones))
        return states, actions, rewards, goals, relabel_goals, next_states, dones
