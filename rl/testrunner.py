import numpy as np
import gym
import torch as th
from gym.spaces.box import Box
from threading import Thread, Lock
from l2env import calc_bps_stats, mean_std_text

class TestRunner:
    def __init__(self, agent, env_name, env_args, deterministic=False, date_divide=1, repeat=2):
        self.agent = agent
        self.device = agent.device
        self.model = agent.model
        self.nenvs = agent.nenvs
        self.date_divide = date_divide
        self.repeat = repeat
        self.counter = 0
        env_args['sample_method'] = 'test'
        self.save_dir = env_args['save_dir']
        print(self.save_dir)
        self.envs = [gym.make(env_name, **env_args) for _ in range(self.nenvs)]
        self.data_cnt = len(self.envs[0].data_lst)
        self.action_space = self.envs[0].action_space
        self.max_time = self.envs[0].max_time
        self.deterministic = deterministic
        self.states = [None for _ in range(self.nenvs)]

    def set_savedir(self, save_dir):
        for env in self.envs:
            env.set_savedir(save_dir)

    def save_test_avg(self, test_name):
        inst_bps = {}
        inst_twap_bps = {}
        inst_filled = {}
        inst_reward = {}

        for env in self.envs:
            for inst in env.inst_data_cnt:
                if inst not in inst_bps:
                    inst_bps[inst] = []
                    inst_twap_bps[inst] = []
                    inst_filled[inst] = []
                    inst_reward[inst] = []
                inst_bps[inst].extend(env.tot_bps[inst])
                inst_twap_bps[inst].extend(env.tot_twap_bps[inst])
                inst_filled[inst].extend(env.tot_filled[inst])
                inst_reward[inst].extend(env.tot_reward[inst])

        tot_bps = []
        tot_twap_bps = []
        tot_filled = []
        tot_reward = []
        for inst in inst_reward:
            logfile = open(f"{self.save_dir}/{inst}_avg.txt", "a")
            bps_mean, bps_std = calc_bps_stats(inst_bps[inst], inst_filled[inst])
            twap_bps_mean, twap_bps_std = calc_bps_stats(inst_twap_bps[inst], inst_filled[inst])
            
            logfile.write(mean_std_text(bps_mean, bps_std))
            for dicts in (inst_filled, inst_reward):
                results = np.array(dicts[inst])
                logfile.write(mean_std_text(results.mean(), results.std()))
            logfile.write(mean_std_text(twap_bps_mean, twap_bps_std))
            logfile.write('\n')
            logfile.close()

            tot_bps.extend(inst_bps[inst])
            tot_twap_bps.extend(inst_twap_bps[inst])
            tot_filled.extend(inst_filled[inst])
            tot_reward.extend(inst_reward[inst])

        logfile = open(f"{self.save_dir}/avg.txt", "a")
        bps_mean, bps_std = calc_bps_stats(tot_bps, tot_filled)
        twap_bps_mean, twap_bps_std = calc_bps_stats(tot_twap_bps, tot_filled)
        logfile.write(mean_std_text(bps_mean, bps_std))
        for arrs in (tot_filled, tot_reward):
            results = np.array(arrs)
            logfile.write(mean_std_text(results.mean(), results.std()))
        logfile.write(mean_std_text(twap_bps_mean, twap_bps_std))
        logfile.write('\n')
        logfile.close()

        self.agent.write_log(test_name, np.array(tot_reward).mean())

        cnts = [0 for _ in range(22)]
        tot = len(tot_reward)
        for i in range(tot):
            v = int(tot_reward[i])
            if v < -100:
                cnts[0] += 1
            elif v >= 100:
                cnts[-1] += 1
            else:
                cnts[11 + v // 10] += 1

        with open(f'{self.save_dir}/table.txt', 'a') as fil:
            for i in range(22):
                fil.write(f'{round(cnts[i]/tot*100, 2)}%\t')
            fil.write('\n\n')

    def reset(self):
        self.test_index = 0
        self.alive = list()
        
        for i in range(self.nenvs):
            self.envs[i].reset_averages()
            self.set_index(i)
            self.alive.append(True)

    def set_index(self, i):
        if self.test_index < self.data_cnt:
            kwargs = {}
            kwargs['index'] = self.test_index
            if self.date_divide != 0:
                kwargs['start_step'] = int((self.counter%self.date_divide) * self.max_time / self.date_divide)
                kwargs['end_step'] = int((self.counter%self.date_divide+1) * self.max_time / self.date_divide)
                self.counter += 1
                if self.counter >= self.date_divide*self.repeat:
                    self.test_index += 1
                    self.counter = 0
            else:
                self.counter += 1
                if self.counter > self.repeat:
                    self.test_index += 1
                    self.counter = 0

            self.states[i] = self.envs[i].reset(kwargs=kwargs) 
        else: 
            self.alive[i] = False

    def run_test(self, test_name='test', log=True):
        print('starting test')
        self.reset()
        tot_rew = 0
        epi_cnt = 0
        while True in self.alive:
            states = th.as_tensor(np.array(self.states), dtype=th.float32).to(self.device)
            action = self.model.get_action(states, deterministic=self.deterministic)
            self.agent.log_losses(states)
            for i in range(self.nenvs):
                if self.alive[i]:
                    if isinstance(self.action_space, Box):
                        act = np.clip(action[i], self.action_space.low, self.action_space.high)
                    else:
                        act = action[i]
                    self.envs[i].send_action(act)

            for i in range(self.nenvs):
                if self.alive[i]:
                    ns, r, done, _ = self.envs[i].recieve()
                    tot_rew += r
                    if done:
                        epi_cnt += 1
                        self.set_index(i)
                    else:
                        self.states[i] = ns
        if log:
            self.save_test_avg(test_name)
        self.agent.flush_log_losses()

        return tot_rew/epi_cnt

    def close(self):
        for env in self.envs:
            env.close()
