import numpy as np
import os
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from ..replaymemory import DREP_ReplayMemory
from ..module.drep_networks import DREPNet
import copy

from torch.utils.tensorboard import SummaryWriter

class DREP_SAC:
    def __init__(self, device, model, name, lr=3e-4, train_steps=100e6, n_steps=1, gradient_steps=1, batch_size=256, \
                gamma=0.99, target_polyak=0.005, capacity=1e6, nenvs=1, train_start=1e4):
        self.model = model.to(device)
        self.target_critic = copy.deepcopy(model)
        
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.name = name
        self.device = device

        self.optimizer = Adam(model.parameters(), lr=lr)
        self.memory = DREP_ReplayMemory(int(capacity), device)

        self.n_steps = n_steps
        self.gamma = gamma
        self.nenvs = nenvs
        self.batch_size = batch_size
        self.target_polyak = target_polyak
        self.gradient_steps = gradient_steps


        self.curr_step = 0
        self.progress = 0
        self.train_steps = train_steps
        self.target_entropy = -self.model.action_size
        self.latent_dim = self.model.extractor.out_size


        self.writer = SummaryWriter(f'logs/{name}')
        #self.writer.add_graph(self.model, input_to_model=th.zeros((1,)+self.model.extractor.in_shape).to(device))

        self.train_start = max(train_start, batch_size)

        self.logging_timer = 0
        self.log_update = 256
        self.reward_log = 0
        self.cnt_log = 0
        self.cur_goal = th.randn(size=(self.nenvs, self.latent_dim), device=self.device)*10

        self.cumulative_loss = {}


    def load_model(self, save_path):
        if os.path.isfile(save_path):
            checkpoint = th.load(save_path)
            self.curr_step = checkpoint['curr_step']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.target_critic = copy.deepcopy(self.model)
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
        }, save_path)


    def write_log(self, tag, value):
        if not np.isfinite(value):
            print(f'{tag} is NaN. {value}')
        self.writer.add_scalar(tag, value, global_step=self.curr_step)


    def update_target(self):
        for param_target, param in zip(self.target_critic.parameters(), self.model.parameters()):
            param_target.copy_(param_target.data * (1.0 - self.target_polyak) + param.data * self.target_polyak)


    def train_epoch(self, env):
        states = env.states

        if self.curr_step >= self.train_start:
            for step in range(self.n_steps):
                with th.no_grad():
                    t_state = th.as_tensor(states, dtype=th.float32).to(self.device)
                    action = self.model.get_action(t_state, self.cur_goal)
                next_states, rewards, dones, _ = env.step(action)
                rewards = rewards.astype(dtype=np.float32)

                for i in range(self.nenvs):
                    self.memory.append(states[i], action[i], rewards[i], self.cur_goal[i].cpu().numpy(), next_states[i], dones[i])
                    if dones[i]:
                        self.cur_goal[i] = th.randn(size=(self.latent_dim,), device=self.device)*10
                states = next_states

                self.reward_log += np.sum(rewards)
                self.cnt_log += np.sum(dones)
                
            for _ in range(self.gradient_steps):
                self.train_batches()
        else:
            action = np.array([env.envs[0].action_space.sample()])
            next_states, rewards, dones, _ = env.step(action)
            rewards = rewards.astype(dtype=np.float32)

            for i in range(self.nenvs):
                self.memory.append(states[i], action[i], rewards[i], self.cur_goal[i].cpu().numpy(), next_states[i], dones[i])
                if dones[i]:
                    self.cur_goal[i] = th.randn(size=(self.latent_dim,), device=self.device)*10

        self.curr_step += self.n_steps * self.nenvs



        if self.logging_timer >= self.log_update:

            for loss_name in self.cumulative_loss:
                self.write_log(loss_name, self.cumulative_loss[loss_name] / self.logging_timer)
                self.cumulative_loss[loss_name] = 0
            
            self.write_log('step', self.curr_step)
            if self.cnt_log != 0:
                self.write_log('avg_score', self.reward_log / self.cnt_log)
                self.reward_log = 0
                self.cnt_log = 0
            
            self.logging_timer = 0
            
        self.progress = self.curr_step / self.train_steps

    def run_trains(self, states, actions, rewards, goals, relabel_goals, next_states, dones):
        alpha = self.model.log_alpha.detach().exp()
        half = goals.shape[0] // 2
        #mixed_goals = th.cat([goals[:half], relabel_goals[half:]], dim=0)
        #ind = th.randperm(half, device=self.device)
        #relabel_goals[:half] = relabel_goals[ind]
        model_goals = self.model.extractor(relabel_goals)
        with th.no_grad():
            target_goals = self.target_critic.extractor(relabel_goals)
            
            goal_rewards = th.mean(th.abs(target_goals - self.target_critic.extractor(states)), dim=-1) \
                            - th.mean(th.abs(target_goals - self.target_critic.extractor(next_states)), dim=-1)

        
            next_actions, next_logprobs, _ = self.model(next_states, target_goals)
            next_logprobs = next_logprobs.unsqueeze(-1)
            q_targ = self.target_critic.get_q(next_states, target_goals, next_actions) - alpha * next_logprobs
            target = goal_rewards[:, None] + self.gamma * (1.0 - dones[:, None]) * q_targ
        
        q1s, q2s = self.model.get_q(states, model_goals, actions, sep=True)
        q_loss = F.mse_loss(q1s, target) + F.mse_loss(q2s, target)


        _, act_logprob, q_value = self.model(states, model_goals)
        act_logprob = act_logprob.unsqueeze(-1)
        policy_loss = (alpha*act_logprob - q_value).mean()


        alpha_loss = -(self.model.log_alpha * (act_logprob + self.target_entropy).detach()).mean()
        
        scale_loss = th.square(th.sum(th.abs(self.model.extractor(states) - self.model.extractor(next_states)), dim=-1)-0.1).mean()

        tot_loss = q_loss + policy_loss + alpha_loss + scale_loss * 10

        self.optimizer.zero_grad()
        tot_loss.backward()
        self.optimizer.step()

        self.update_target()

        return {'q_loss' : q_loss.item(), 
        'avg_q' : q_value.mean().item(), 
        'policy_loss' : policy_loss.item(), 
        'alpha_loss' : alpha_loss.item(), 
        'log_alpha' : self.model.log_alpha.item(),
        'scale_loss' : scale_loss.item(),
        'goal_reward_mean' : goal_rewards.mean().item()}

    def train_batches(self):
        datas = self.memory.sample(self.batch_size)
        losses = self.run_trains(*datas)

        for loss in losses:
            if loss not in self.cumulative_loss:
                self.cumulative_loss[loss] = 0
            self.cumulative_loss[loss] += losses[loss]

        self.logging_timer += 1

        