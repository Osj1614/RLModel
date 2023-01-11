from torch import distributions
from torch.optim import Adam
import numpy as np
import os
import torch as th
import torch.nn.functional as F
from rl.running_std import RewardNormalizer

from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(self, device, model, name='', nenvs=32, nsteps=256, train_steps=200e6, lr=2.5e-4, minibatches=8, critic_coef=1, ent_coef=0.01, pred_coef=0.0,\
         gamma=0.99, epsilon=0.2, lamda=0.95, policy_epochs=4):
        self.model = model.to(device)
        self.optimizer_policy = Adam(model.parameters(), lr=lr)
        self.name = name
        self.device = device
        self.train_steps = train_steps
        self.obs_shape = self.model.extractor.in_shape
    
        self.nenvs = nenvs
        self.nsteps = nsteps
        self.minibatches = minibatches
        self.critic_coef = critic_coef
        self.ent_coef = ent_coef
        self.pred_coef = pred_coef
        self.gamma = gamma
        self.epsilon = epsilon
        self.lamda = lamda
        self.policy_epochs = policy_epochs


        self.update_cnt = 0
        self.train_steps = int(train_steps)
        self.curr_step = 0
        self.update_steps = self.nenvs*self.nsteps
        self.progress = 0

        self.rew_norm = RewardNormalizer(nenvs, cliprew=100, gamma=gamma)

        self.policy_losses = ['Actor_loss', 'Critic_loss', 'Entropy', 'Clipfrac', 'Approx_kl']
        self.writer = SummaryWriter(f'logs/{name}')
        self.writer.add_graph(self.model, input_to_model=th.zeros((1,)+self.obs_shape).to(device))

        self.reward_log = 0
        self.cnt_log = 0

    def load_model(self, save_path):
        if os.path.isfile(save_path):
            checkpoint = th.load(save_path)
            self.curr_step = checkpoint['curr_step']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
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
            'optimizer_policy' : self.optimizer_policy.state_dict(),
            'ret_rms' : self.rew_norm.ret_rms.state_dict()
        }, save_path)

    def train_epoch(self, env):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]
        v_lst = [list() for _ in range(self.nenvs)]
        action_prob_lst = [list() for _ in range(self.nenvs)]
        target_lst = [list() for _ in range(self.nenvs)]

        states = env.states.copy()

        for step in range(self.nsteps):
            action, action_prob, values = self.model.get_action_value(th.as_tensor(np.array(states), dtype=th.float32).to(self.device))
            
            if action.shape[-1] == 1:
                next_states, rewards, dones, _ = env.step(np.squeeze(action, -1))
            else:
                next_states, rewards, dones, _ = env.step(action)


                
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
        self.train_batches(batches)

    def run_trains(self, s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst):
        loss_sum = np.zeros(len(self.policy_losses))
        size = self.update_steps
        minibatch_size = size // self.minibatches
        order = np.arange(size)
        
        for _ in range(self.policy_epochs):
            np.random.shuffle(order)
            for i in range(0, size, minibatch_size):
                end = i + minibatch_size
                ind = order[i:end]
                slices = (arr[ind].to(self.device) for arr in (s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst))
                loss_sum += self.train_policy_phase(*slices)
        '''
        else:
            envsperbatch = self.network.nenvs // self.network.minibatches
            envinds = np.arange(self.network.nenvs)
            flatinds = np.arange(self.network.nenvs * self.network.nsteps).reshape(self.network.nenvs, self.network.nsteps)
            np.random.shuffle(envinds)
            for i in range(0, self.network.nenvs, envsperbatch):
                end = i + envsperbatch
                mbenvinds = envinds[i:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in (s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, m_lst))
                hstates = hs_lst[mbenvinds]
                loss_sum += self.train_policy_phase(*slices)
        '''
        loss_sum /= self.minibatches*self.policy_epochs
        
        for i in range(len(self.policy_losses)):
            self.write_log(self.policy_losses[i], loss_sum[i])

    def train_policy_phase(self, states, actions, returns, advantages, action_probs):
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

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
        self.optimizer_policy.step()
        
        return np.array([actor_loss.item(), critic_loss.item(), entropy_bonus.item(), clipfrac.item(), approxkl.item()])

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

    def train_batches(self, batches):
        s_lsts = th.from_numpy(batches['states']).float().view(self.update_steps, *self.obs_shape)
        a_lsts = th.from_numpy(batches['actions']).float().view(self.update_steps, -1)
        r_lsts = batches['rewards']
        done_lsts = batches['dones']
        self.write_log("Avg_reward", np.average(r_lsts))
        #r_lsts /= 30
        r_lsts = self.rew_norm(r_lsts, done_lsts)
        action_prob_lsts = th.from_numpy(batches['action_probs']).float().view(self.update_steps)
        value_lsts = batches['values']
        #target_lsts = th.from_numpy(batches['targets']).float().view(self.update_steps, -1)
        
        advantage_lsts = self.calc_gae(r_lsts, value_lsts, done_lsts)
        value_lsts = value_lsts[:, :-1]
        self.write_log('Value', value_lsts.mean())
        self.write_log('Advantage', advantage_lsts.mean())

        self.write_log('avg_score', self.reward_log / self.cnt_log)
        self.reward_log = 0
        self.cnt_log = 0
        
        returns_lsts = value_lsts + advantage_lsts
        advantage_lsts = (advantage_lsts - advantage_lsts.mean()) / (advantage_lsts.std() + 1e-8)
        
        returns_lsts = th.from_numpy(returns_lsts).float().view(self.update_steps)
        advantage_lsts = th.from_numpy(advantage_lsts).float().view(self.update_steps)
        
        self.run_trains(s_lsts, a_lsts, returns_lsts, advantage_lsts, action_prob_lsts)

        self.curr_step += self.update_steps
        self.progress += self.update_steps / self.train_steps

    def log_losses(self, states):
        pass

    def flush_log_losses(self):
        pass