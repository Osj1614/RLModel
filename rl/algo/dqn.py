import numpy as np
import os
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from rl.running_std import RewardNormalizer
from torch.optim import Adam
from rl.replaymemory import ReplayMemory
import copy

from torch.utils.tensorboard import SummaryWriter

class DQN:
    def __init__(self, device, model, name, lr=5e-5, train_steps=100e6, n_step=1, train_ratio=4.0, batch_size=512, gamma=0.99, target_update=1e4, priority_weight=0.4, priority_exponent=0.6, capacity=2e5):
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model)
        
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.name = name
        self.device = device

        self.optimizer = Adam(model.parameters(), lr=lr)
        self.memory = ReplayMemory(device, gamma, n_step, priority_weight, priority_exponent, int(capacity), self.model.extractor.in_shape)

        self.num_quantiles = self.model.num_quantiles
        self.n_step = n_step
        self.gamma = gamma
        self.nenvs = 1
        self.batch_size = batch_size
        self.target_update = target_update
        self.curr_step = 0
        self.progress = 0
        self.train_steps = train_steps
        self.nsteps = int(self.batch_size // train_ratio)

        self.target_update_timer = 0
        self.train_ratio = train_ratio

        #self.taus = (th.arange(0.5, model.num_quantiles+0.5) / model.num_quantiles).to(device)
        self.clip_edges = model.num_quantiles * 0.1
        self.taus = (th.arange(self.clip_edges+0.5, model.num_quantiles+self.clip_edges+0.5) / (model.num_quantiles+self.clip_edges*2)).to(device)

        self.rew_norm = RewardNormalizer(1, cliprew=100, gamma=gamma)

        self.losses = ['q_loss', 'avg_q']
        self.writer = SummaryWriter(f'logs/{name}')
        self.writer.add_graph(self.model, input_to_model=th.zeros((1,)+self.model.extractor.in_shape).to(device))

        self.reward_log = 0
        self.cnt_log = 0


    def load_model(self, save_path):
        if os.path.isfile(save_path):
            checkpoint = th.load(save_path)
            self.curr_step = checkpoint['curr_step']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.rew_norm.ret_rms.load_state_dict(checkpoint['ret_rms'])
            self.update_target()
            self.progress = self.curr_step / self.train_steps
            print("Model loaded")
        else:
            print("No model is found")

    def save_model(self, save_path):
        save_dir = save_path[:save_path.rfind("/")]
        os.makedirs(save_dir, exist_ok=True)
        th.save({
            'curr_step':self.curr_step,
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'ret_rms' : self.rew_norm.ret_rms.state_dict()
        }, save_path)

    def write_log(self, tag, value):
        if not np.isfinite(value):
            print(f'{tag} is NaN. {value}')
        self.writer.add_scalar(tag, value, global_step=self.curr_step)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_q_target(self, next_s_lst, reward_lst, done_lst):
        q_targ_best = self.target_model.get_q_sa(next_s_lst, self.model.get_best_action(next_s_lst)) # batch, action, atom
        return reward_lst[:, None] + self.gamma ** self.n_step * (1.0 - done_lst[:, None]) * q_targ_best
    
    def huber_loss(self, diff, k=0.0):
        if k == 0:
            return diff.abs()
        return th.where(diff.abs() <= k, 0.5 * diff.pow(2), k * (diff.abs() - 0.5 * k))

    def train_epoch(self, env):
        states = env.states
        for step in range(self.nsteps):
            with th.no_grad():
                action = self.model.get_action(th.as_tensor(states, dtype=th.float32).to(self.device))
            next_states, rewards, dones, _ = env.step(action)
            #rewards = rewards / 30
            #norm_rewards = self.rew_norm(rewards[None, :], dones[None, :])
            self.memory.append(states[0], action[0], rewards[0], dones[0])
            states = next_states

            self.reward_log += rewards[0]
            self.cnt_log += dones[0]
        
        self.train_batches()

    def run_trains(self, s_lst, a_lst, reward_lst, next_s_lst, done_lst, importance_weights):
        with th.no_grad():
            self.model.reset_noise()
            self.target_model.reset_noise()
            targets = self.get_q_target(next_s_lst, reward_lst, done_lst)
        
        q_values = self.model.get_q_sa(s_lst, a_lst)
        
        diff = targets[:,:,None] - q_values[:,None,:]
        huber_loss = self.huber_loss(diff, 1.0)
        td_error = (abs(self.taus - (diff.detach() < 0).float()) * huber_loss).sum(dim=-1).mean(dim=-1)

        q_loss = th.mean(td_error * importance_weights)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        return {'td_error' : td_error.detach().cpu().numpy(), 'q_loss' : q_loss.item(), 'avg_q' : q_values.mean().item()}

    def train_batches(self):
        self.curr_step += self.nsteps
        
        if self.curr_step < self.batch_size*20:
            return

        self.target_update_timer += self.nsteps
        
        idxs, *datas = self.memory.sample(self.batch_size)
        losses = self.run_trains(*datas)
        self.memory.update_priorities(idxs, losses['td_error'])

        if self.target_update_timer >= self.target_update:
            self.update_target()
            self.target_update_timer = 0

            for loss_name in self.losses:
                self.write_log(loss_name, losses[loss_name])
                
            self.write_log('avg_score', self.reward_log / self.cnt_log)
            self.reward_log = 0
            self.cnt_log = 0
            
        self.progress = self.curr_step / self.train_steps
        