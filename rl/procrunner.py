import numpy as np
import gym
import torch as th
from select import select
from threading import Thread
from multiprocessing.connection import wait
from gym.spaces.box import Box
import time

class ProcRunner:
    def __init__(self, device, model, env_name, env_args):
        self.model = model
        self.nenvs = model.nenvs
        self.update_interval = model.nsteps
        self.device = device
        self.is_pri = False
        self.envs = [gym.make(env_name, **env_args) for _ in range(self.nenvs)]
        self.action_space = self.envs[0].action_space
        self.states = list()
        self.targets = list()
        self.total_reward = [0 for _ in range(self.nenvs)]
        for i in range(self.nenvs):
            s, targ = self.envs[i].reset()
            self.states.append(s)
            self.targets.append(targ)

        self.avg = 0
        self.high = -1000000
        self.cnt = 0

    def get_avg_high(self):
        avg = self.avg / (self.cnt+1e-8)
        high = self.high
        self.avg = 0
        self.high = -1000000
        self.cnt = 0
        return avg, high

    def run_steps(self):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]
        v_lst = [list() for _ in range(self.nenvs)]
        action_prob_lst = [list() for _ in range(self.nenvs)]
        target_lst = [list() for _ in range(self.nenvs)]

        prev = time.time()
        
        for step in range(self.update_interval):
            
            cur = time.time()
            #print(f'Step time: {cur - prev}')
            prev = cur
        
            with th.no_grad():
                action, action_prob, values, preds = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))
            
            pipes = []
            alive_pipes = []

            for i in range(self.nenvs):
                target_lst[i].append(self.targets[i])
                if isinstance(self.action_space, Box):
                    act = np.clip(action[i], self.action_space.low, self.action_space.high)
                else:
                    act = action[i][0]
                self.envs[i].send_action(act, preds[i], values[i])
                pipes.append(self.envs[i].pipe)
                alive_pipes.append(self.envs[i].pipe)
            
            while alive_pipes:
                for pipe in wait(alive_pipes):
                    i = pipes.index(pipe)
                    
                    ns, reward, done, self.targets[i] = self.envs[i].recieve()
            
                    if np.max(ns) > 10:
                        print(f'WARNING: state too large {np.max(ns)}')
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
                        self.envs[i].set_progress(self.model.progress)
                        self.states[i], self.targets[i] = self.envs[i].reset()
                    else:
                        self.states[i] = ns

                    alive_pipes.remove(pipe)

        _, _, values, _ = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))
        for i in range(self.nenvs):
            v_lst[i].append(values[i])

        batches = {
            'states':np.array(s_lst),
            'actions':np.array(a_lst),
            'rewards':np.array(r_lst),
            'dones':np.array(done_lst),
            'action_probs':np.array(action_prob_lst),
            'values':np.array(v_lst),
            'targets':np.array(target_lst)
        }
        
        return batches

    def run_steps_nopipe(self):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]
        v_lst = [list() for _ in range(self.nenvs)]
        action_prob_lst = [list() for _ in range(self.nenvs)]
        target_lst = [list() for _ in range(self.nenvs)]

        prev = time.time()
        
        for step in range(self.update_interval):
            
            cur = time.time()
            #print(f'Step time: {cur - prev}')
            prev = cur
        
            with th.no_grad():
                action, action_prob, values, preds = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))


            for i in range(self.nenvs):
                target_lst[i].append(self.targets[i])
                if isinstance(self.action_space, Box):
                    act = np.clip(action[i], self.action_space.low, self.action_space.high)
                else:
                    act = action[i][0]
                self.envs[i].send_action(act, preds[i], values[i])
                ns, reward, done, self.targets[i] = self.envs[i].recieve()
        
                if np.max(ns) > 10:
                    print(f'WARNING: state too large {np.max(ns)}')
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
                    self.envs[i].set_progress(self.model.progress)
                    self.states[i], self.targets[i] = self.envs[i].reset()
                else:
                    self.states[i] = ns

        _, _, values, _ = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))
        for i in range(self.nenvs):
            v_lst[i].append(values[i])

        batches = {
            'states':np.array(s_lst),
            'actions':np.array(a_lst),
            'rewards':np.array(r_lst),
            'dones':np.array(done_lst),
            'action_probs':np.array(action_prob_lst),
            'values':np.array(v_lst),
            'targets':np.array(target_lst)
        }
        
        return batches

    def run_steps_dqn(self):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]

        prev = time.time()
        
        for step in range(self.update_interval):
            
            cur = time.time()
            #print(f'Step time: {cur - prev}')
            prev = cur
        
            with th.no_grad():
                action = self.model.model.get_action(th.as_tensor(self.states, dtype=th.float32).to(self.device))
            
            for i in range(self.nenvs):
                act = action[i]
                ns, reward, done, _ = self.envs[i].step(act)
        
                if np.max(ns) > 10:
                    print(f'WARNING: state too large {np.max(ns)}')
                self.total_reward[i] += reward
                s_lst[i].append(self.states[i])
                a_lst[i].append(action[i])
                r_lst[i].append(reward)
                done_lst[i].append(done)
                if done:
                    self.avg += self.total_reward[i]
                    self.cnt += 1
                    self.total_reward[i] = 0
                    self.envs[i].set_progress(self.model.progress)
                    self.states[i], self.targets[i] = self.envs[i].reset()
                else:
                    self.states[i] = ns

        batches = {
            'states':np.array(s_lst),
            'actions':np.array(a_lst),
            'rewards':np.array(r_lst),
            'dones':np.array(done_lst)
        }
        
        return batches

    def close(self):
        for env in self.envs:
            env.close()
