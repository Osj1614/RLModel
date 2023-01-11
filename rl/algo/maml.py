from torch import distributions
from torch.optim import Adam, SGD
import numpy as np
import os
import torch as th
import torch.nn.functional as F
from .running_std import RewardNormalizer
import copy
import torchopt

from torch.utils.tensorboard import SummaryWriter

def clip_grad(grads):
    max_norm = 0.3
    norm_type = 2

    device = grads[0].device

    total_norm = th.norm(th.stack([th.norm(grad.detach(), norm_type).to(device) for grad in grads]), norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = th.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.detach().mul_(clip_coef_clamped.to(device))
    return grads


class MAML:
    def __init__(self, device, model, env, name='', nenvs=1, nsteps=512, train_steps=100e6, lr=1e-3, minibatches=1, critic_coef=0.5, ent_coef=0.01,\
         gamma=0.99, epsilon=0.2, lamda=0.95, policy_epochs=4, inner_epochs=4, boot_epochs=4, ntasks=8, use_bmg=True):
        self.model = model.to(device)
        self.lr = lr
        self.outer_opt = torchopt.Adam(self.model.parameters(), lr=self.lr, eps=1e-4, weight_decay=1e-7)
        if use_bmg:
            self.inner_lr = 1e-3
            self.boot_epochs = boot_epochs

            #clip_optim = torchopt.combine.chain(torchopt.clip.clip_grad_norm(max_norm=1.0), torchopt.adam(lr=self.inner_lr, eps=1e-4, weight_decay=1e-7, use_accelerated_op=True, moment_requires_grad=True))
            #self.inner_opt = torchopt.MetaOptimizer(self.model, clip_optim)
            self.inner_opt = torchopt.MetaAdam(self.model, lr=self.inner_lr, eps=1e-4, weight_decay=1e-7, use_accelerated_op=True, moment_requires_grad=True)
            self.inner_last_opt = torchopt.MetaSGD(self.model, lr=1e-2, moment_requires_grad=False)
        else:
            self.inner_lr = 0.01
            self.boot_epochs = 0
            #clip_optim = torchopt.combine.chain(torchopt.clip.clip_grad_norm(max_norm=1.0), torchopt.sgd(lr=self.inner_lr, weight_decay=1e-7, moment_requires_grad=True))
            #self.inner_opt = torchopt.MetaOptimizer(self.model, clip_optim)
            self.inner_opt = torchopt.MetaSGD(self.model, lr=self.inner_lr, weight_decay=1e-7, moment_requires_grad=True)
            

        self.name = name
        self.device = device
        self.train_steps = train_steps
        self.obs_shape = self.model.extractor.in_shape
        self.use_bmg = use_bmg
    
        self.nenvs = nenvs
        self.nsteps = nsteps
        self.ntask = ntasks
        self.minibatches = minibatches
        self.critic_coef = critic_coef
        self.ent_coef = ent_coef
        self.boot_value_coef = 0.25
        self.gamma = gamma
        self.epsilon = epsilon
        self.lamda = lamda
        self.policy_epochs = policy_epochs
        self.inner_epochs = inner_epochs
        self.train_interval = 20
        self.test_interval = 5

        self.update_cnt = 0
        self.train_steps = int(train_steps)
        self.curr_step = 0
        self.update_steps = self.nenvs*self.nsteps
        self.progress = 0

        self.rew_norm = RewardNormalizer(nenvs, cliprew=10, gamma=gamma)

        self.policy_losses = ['Actor_loss', 'Critic_loss', 'Entropy', 'Clipfrac', 'Approx_kl']
        self.writer = SummaryWriter(f'logs/{name}')
        self.writer.add_graph(self.model, input_to_model=th.zeros((1,)+self.obs_shape).to(device))

        self.reward_log = 0
        self.cnt_log = 0
        
        self.insts = env.targ_insts
        self.dates = []
        self.train_dates = []
        for data in l2env.data_lst:
            if not data[0] in self.dates:
                self.dates.append(data[0])
                
        for data in env.data_lst:
            if not data[0] in self.train_dates:
                self.train_dates.append(data[0])

        self.cur_test_date = 0
        self.cur_inst = 0

    def load_model(self, save_path):
        if os.path.isfile(save_path):
            checkpoint = th.load(save_path)
            self.curr_step = checkpoint['curr_step']
            self.model.load_state_dict(checkpoint['model'])
            self.outer_opt.load_state_dict(checkpoint['outer_opt'])
            self.rew_norm.ret_rms.load_state_dict(checkpoint['ret_rms'])
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
            'outer_opt' : self.outer_opt.state_dict(),
            'ret_rms' : self.rew_norm.ret_rms.state_dict()
        }, save_path)
    

    def train_epoch(self, vecenv):
        tot_loss = 0
        for k in range(self.ntask):

            net_state = torchopt.extract_state_dict(self.model)
            optim_state = torchopt.extract_state_dict(self.inner_opt)


            self.cur_test_date = np.random.randint(self.train_interval, len(self.train_dates))
            self.cur_inst = np.random.randint(0, len(self.insts))
            train_dates = f'{self.train_dates[self.cur_test_date-self.train_interval]}~{self.train_dates[self.cur_test_date-1]}'
            
            for env in vecenv.envs:
                env.set_id(train_dates, self.insts[self.cur_inst])
            vecenv.reset(mode='inner')


            for _ in range(self.inner_epochs):
                self.run_env(vecenv, train=True)
            

            test_date = f'{self.dates[self.cur_test_date]}~{self.dates[self.cur_test_date]}'
            for env in vecenv.envs:
                env.set_id(test_date, self.insts[self.cur_inst])
            vecenv.reset(mode='outer')

            if self.use_bmg:
                k_net_state = torchopt.extract_state_dict(self.model)

                for _ in range(self.boot_epochs-1):
                    self.run_env(vecenv, train=True)
                last_loss, batches = self.run_env(vecenv, train=False)
                self.inner_last_opt.step(last_loss)

                states = th.from_numpy(batches['states']).float().view(-1, *self.obs_shape).to(self.device)
                with th.no_grad():
                    boot_value, boot_policy = self.model(states)
                    boot_policy = self.model.get_dist(boot_policy)
                
                torchopt.recover_state_dict(self.model, k_net_state)

                orig_value, orig_policy = self.model(states)
                orig_policy = self.model.get_dist(orig_policy)
                
                value_matching = th.square(orig_value-boot_value).mean()
                policy_matching = distributions.kl_divergence(boot_policy, orig_policy).sum(axis=-1).mean()
                
                self.write_log('outer/value_matching_loss', value_matching.detach().item())
                self.write_log('outer/policy_matching_loss', policy_matching.detach().item())

                boot_loss = value_matching*self.boot_value_coef + policy_matching
                self.write_log('outer/boot_loss', boot_loss.detach().item())
                tot_loss += boot_loss
            else:
                test_loss, _ = self.run_env(vecenv, train=False)
                tot_loss += test_loss

            torchopt.recover_state_dict(self.model, net_state)
            torchopt.recover_state_dict(self.inner_opt, optim_state)
        
        tot_loss /= self.ntask
        self.outer_opt.zero_grad()
        tot_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.outer_opt.step()
        
        self.curr_step += self.update_steps * (self.inner_epochs+self.boot_epochs) * self.ntask
        self.progress = self.curr_step / self.train_steps


    def get_date_model(self, vecenv, inst, date):        
        net_state = torchopt.extract_state_dict(self.model)
        optim_state = torchopt.extract_state_dict(self.inner_opt)

        self.cur_test_date = self.dates.index(date)
        train_dates = f'{self.dates[self.cur_test_date-self.train_interval]}~{self.dates[self.cur_test_date-1]}'

        for env in vecenv.envs:
            env.set_id(train_dates, inst)
        vecenv.reset(mode='inner')
        for _ in range(self.inner_epochs):
            self.run_env(vecenv, train=True)
        
        yield self.model

        torchopt.recover_state_dict(self.model, net_state)
        torchopt.recover_state_dict(self.inner_opt, optim_state)


    def run_env(self, env, train=False):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]
        v_lst = [list() for _ in range(self.nenvs)]
        action_prob_lst = [list() for _ in range(self.nenvs)]
        target_lst = [list() for _ in range(self.nenvs)]

        states = env.states.copy()
        for _ in range(self.nsteps):
            action, action_prob, values = self.model.get_action_value(th.as_tensor(np.array(states), dtype=th.float32).to(self.device))
            
            next_states, rewards, dones, _ = env.step(np.squeeze(action, -1))
            for i in range(self.nenvs):
                s_lst[i].append(states[i])
                a_lst[i].append(action[i])
                r_lst[i].append(rewards[i])
                action_prob_lst[i].append(action_prob[i])
                done_lst[i].append(dones[i])
                v_lst[i].append(values[i])
            
                self.reward_log += rewards[i]
                self.cnt_log += dones[i]
            
            states = next_states
        
        _, _, values = self.model.get_action_value(th.as_tensor(np.array(states), dtype=th.float32).to(self.device))
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
        return self.train_batches(batches, train=train), batches

    def batch_loss(self, s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, train=False):
        loss_sum = np.zeros(len(self.policy_losses))
        size = self.update_steps
        minibatch_size = size // self.minibatches
        order = np.arange(size)
        tot_loss = 0

        for _ in range(self.policy_epochs):
            np.random.shuffle(order)
            for i in range(0, size, minibatch_size):
                end = i + minibatch_size
                ind = order[i:end]
                slices = (arr[ind].to(self.device) for arr in (s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst))
                loss, losses_log = self.get_loss(*slices)
                loss_sum += losses_log
                if train:
                    self.inner_opt.step(loss)
                else:
                    tot_loss += loss

            if not train:
                break
        if train:
            loss_sum /= self.minibatches*self.policy_epochs
        else:
            loss_sum /= self.minibatches
            tot_loss /= self.minibatches

        train_log = 'inner' if train else 'outer'
        for i in range(len(self.policy_losses)):
            self.write_log(f'{train_log}/{self.policy_losses[i]}', loss_sum[i])
        
        return tot_loss

    def get_loss(self, states, actions, returns, advantages, action_probs):
        value, policy = self.model(states)
        policy = self.model.get_dist(policy)
        cur_prob = policy.log_prob(actions).sum(axis=-1)

        ratio = th.exp(cur_prob - action_probs)
        noclip_gain = ratio * advantages
        clipped_gain = th.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        actor_loss = -th.min(noclip_gain, clipped_gain).mean()
        entropy_bonus = policy.entropy().sum(axis=-1).mean()
        critic_loss = F.mse_loss(value, returns)
        policy_loss = actor_loss - entropy_bonus * self.ent_coef + critic_loss * self.critic_coef

        with th.no_grad():
            clipfrac = (th.abs(ratio - 1) > self.epsilon).float().mean()
            approxkl = 0.5 * (th.log(ratio) ** 2).mean()

        return policy_loss, np.array([actor_loss.item(), critic_loss.item(), entropy_bonus.item(), clipfrac.item(), approxkl.item()])

    def train_batches(self, batches, train=False):
        train_log = 'inner' if train else 'outer'
        s_lsts = th.from_numpy(batches['states']).float().view(self.update_steps, *self.obs_shape)
        a_lsts = th.from_numpy(batches['actions']).float().view(self.update_steps, -1)
        r_lsts = batches['rewards']
        self.write_log(f'{train_log}/Avg_reward', np.average(r_lsts))
        done_lsts = batches['dones']
        #r_lsts /= 30
        r_lsts = self.rew_norm(r_lsts, done_lsts)
        action_prob_lsts = th.from_numpy(batches['action_probs']).float().view(self.update_steps)
        value_lsts = batches['values']
        #target_lsts = th.from_numpy(batches['targets']).float().view(self.update_steps, -1)
        
        advantage_lsts = self.calc_gae(r_lsts, value_lsts, done_lsts)
        value_lsts = value_lsts[:, :-1]

        self.write_log(f'{train_log}/Value', value_lsts.mean())
        self.write_log(f'{train_log}/Advantage', advantage_lsts.mean())
        self.write_log(f'{train_log}/Avg_score', self.reward_log / self.cnt_log)
        self.reward_log = 0
        self.cnt_log = 0
        
        returns_lsts = value_lsts + advantage_lsts
        advantage_lsts = (advantage_lsts - advantage_lsts.mean()) / (advantage_lsts.std() + 1e-8)
        
        returns_lsts = th.from_numpy(returns_lsts).float().view(self.update_steps)
        advantage_lsts = th.from_numpy(advantage_lsts).float().view(self.update_steps)

        return self.batch_loss(s_lsts, a_lsts, returns_lsts, advantage_lsts, action_prob_lsts, train=train)


    def calc_gae(self, r_lst, value_lst, done_lst, gamma=None):
        if gamma == None:
            gamma = self.gamma
        cur = np.zeros([self.nenvs])
        advantage_lst = np.zeros([self.nenvs, self.nsteps])
        for i in reversed(range(self.nsteps)):
            delta = r_lst[:,i] + gamma * value_lst[:,i+1] * (1-done_lst[:,i]) - value_lst[:,i]
            advantage_lst[:,i] = cur = self.lamda * gamma * (1-done_lst[:,i]) * cur + delta
        return advantage_lst

    def write_log(self, tag, value):
        if not np.isfinite(value):
            print(f'{tag} is NaN. {value}')
        self.writer.add_scalar(tag, value, global_step=self.curr_step)

    def log_losses(self, states):
        pass

    def flush_log_losses(self):
        pass
