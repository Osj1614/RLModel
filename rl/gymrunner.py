import numpy as np
import gym
import torch as th
from select import select
from multiprocessing.connection import wait
from gym.spaces.box import Box
import time

class GymRunner:
    def __init__(self, device, model, env_name):
        self.model = model
        self.nenvs = model.nenvs
        self.update_interval = model.nsteps
        self.device = device
        self.envs = [gym.make(env_name) for _ in range(self.nenvs)]
        
        self.states = list()

        self.total_reward = [0 for _ in range(self.nenvs)]
        for i in range(self.nenvs):
            s = self.envs[i].reset()
            self.states.append(s)

        self.avg = 0
        self.high = -1000000
        self.cnt = 0

    def get_avg_high(self):
        avg = self.avg / (self.cnt+1e-8)
        self.avg = 0
        #self.high = -1000000
        self.cnt = 0
        return avg

    def run_steps(self):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]
        v_lst = [list() for _ in range(self.nenvs)]
        action_prob_lst = [list() for _ in range(self.nenvs)]

        #prev = time.time()
        for step in range(self.update_interval):
            #cur = time.time()
            #print(f'Step time: {cur - prev}')
            #prev = cur
            with th.no_grad():
                action, action_prob, values = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))

            for i in range(self.nenvs):
                if isinstance(self.envs[i].action_space, Box):
                    act = np.clip(action[i], self.envs[i].action_space.low, self.envs[i].action_space.high)
                else:
                    act = action[i][0]
                ns, reward, done, _ = self.envs[i].step(act)
                done = 1 if done else 0
                self.total_reward[i] += reward
                s_lst[i].append(self.states[i])
                a_lst[i].append(action[i])
                r_lst[i].append(reward)
                action_prob_lst[i].append(action_prob[i])
                done_lst[i].append(done)
                v_lst[i].append(values[i])
                if done:
                    self.avg += self.total_reward[i]
                    self.cnt += 1
                    self.total_reward[i] = 0
                    self.states[i] = self.envs[i].reset()
                else:
                    self.states[i] = ns


        _, _, values = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))
        for i in range(self.nenvs):
            v_lst[i].append(values[i])

        batches = {
            'states':np.array(s_lst),
            'actions':np.array(a_lst),
            'rewards':np.array(r_lst),
            'dones':np.array(done_lst),
            'action_probs':np.array(action_prob_lst),
            'values':np.array(v_lst)
        }
        
        return batches

    def run_steps_dqn(self):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]

        #prev = time.time()
        for step in range(self.update_interval):
            #cur = time.time()
            #print(f'Step time: {cur - prev}')
            #prev = cur
            with th.no_grad():
                action = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))

            for i in range(self.nenvs):
                act = action[i]
                ns, reward, done, _ = self.envs[i].step(act)
                self.total_reward[i] += reward
                s_lst[i].append(self.states[i])
                a_lst[i].append(action[i])
                r_lst[i].append(reward)
                done_lst[i].append(done)
                if done:
                    self.avg += self.total_reward[i]
                    self.cnt += 1
                    self.total_reward[i] = 0
                    self.states[i] = self.envs[i].reset()
                else:
                    self.states[i] = ns

        batches = {
            'states':np.array(s_lst),
            'actions':np.array(a_lst),
            'rewards':np.array(r_lst),
            'dones':np.array(done_lst),
        }
        
        return batches
