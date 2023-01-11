import math
import numpy as np
import torch as th
from torch import nn, Tensor
from torch.nn import functional as F

from .distributions import tanh_normal
from torch.distributions import Normal, Categorical, Gumbel



class ActorCriticNetwork(nn.Module):
    def __init__(self, action_type, action_size, extractor, act_fn=F.relu):
        super().__init__()
        self.action_size = action_size
        self.action_type = action_type
        self.extractor = extractor
        self.in_size = extractor.out_size

        self.act_fn = act_fn
        #self.shared_fc1 = nn.Linear(self.in_size, 128)
        #self.shared_fc2 = nn.Linear(128, 128)
        self.value_fc1 = nn.Linear(self.in_size, 256)
        self.value_fc2 = nn.Linear(256, 256)
        self.policy_fc1 = nn.Linear(self.in_size, 256)
        self.policy_fc2 = nn.Linear(256, 256)

        self.value_fc = nn.Linear(256, 1)
        self.policy_fc = nn.Linear(256, action_size)

        nn.init.xavier_normal_(self.policy_fc.weight, 0.01)

        if action_type == 'continuous':
            self.policy_std = nn.Parameter(th.ones(self.action_size) * 1, requires_grad=True)
        
    def forward(self, input, detach_value=False):
        latent = self.extractor(input)
        hidden = self.act_fn(self.policy_fc1(latent))

        hidden = self.act_fn(self.policy_fc2(hidden))
        policy = self.policy_fc(hidden)

        if detach_value:
            latent = latent.detach()

        hidden = self.act_fn(self.value_fc1(latent))

        hidden = self.act_fn(self.value_fc2(hidden))

        value = self.value_fc(hidden)
       
        return value.squeeze(dim=1), policy
    
    def get_dist(self, policy, detach_std=False):
        if self.action_type == 'discrete':
            policy = Categorical(logits=policy[:,None])
        elif self.action_type == 'continuous':
            if detach_std:
                policy = tanh_normal(policy, self.policy_std.detach().exp().repeat(policy.shape[0], 1))
            else:
                policy = tanh_normal(policy, self.policy_std.exp().repeat(policy.shape[0], 1))
        return policy 

    def get_deterministic_action(self, policy):
        if self.action_type == 'discrete':
            return th.argmax(policy, dim=-1, keepdim=True)
        elif self.action_type == 'continuous':
            return policy

    def get_action_value(self, input, deterministic=False):
        with th.no_grad():
            value, policy = self.forward(input)
            policy_dist = self.get_dist(policy)
            if deterministic:
                action = self.get_deterministic_action(policy)
            else:
                action = policy_dist.sample()
            action_prob = policy_dist.log_prob(action).sum(axis=-1)
        return action.cpu().numpy(), action_prob.cpu().numpy(), value.cpu().numpy()

    def get_action(self, input, deterministic=False):
        with th.no_grad():
            _, policy = self.forward(input)
            policy_dist = self.get_dist(policy)
            if deterministic:
                action = self.get_deterministic_action(policy)
            else:
                action = policy_dist.sample()
        
        if self.action_type == 'discrete':
            action = action.squeeze(-1)

        return action.cpu().numpy()



class ActorCriticNetwork2(nn.Module):
    def __init__(self, action_type, action_size, extractor, act_fn=F.relu):
        super().__init__()
        self.action_size = action_size
        self.action_type = action_type
        self.extractor = extractor
        self.in_size = extractor.out_size

        self.act_fn = act_fn
        #self.shared_fc1 = nn.Linear(self.in_size, 128)
        #self.shared_fc2 = nn.Linear(128, 128)
        self.value_fc1 = nn.Linear(self.in_size, 128)
        self.value_fc2 = nn.Linear(256, 256)
        self.policy_fc1 = nn.Linear(self.in_size, 128)
        self.policy_fc2 = nn.Linear(256, 256)

        self.value_fc = nn.Linear(512, 1)
        self.policy_fc = nn.Linear(512, action_size)

        nn.init.xavier_normal_(self.policy_fc.weight, 0.01)

        if action_type == 'continuous':
            self.policy_std = nn.Parameter(th.ones(self.action_size) * 1, requires_grad=True)
        
    def forward(self, input, detach_value=False):
        latent = self.extractor(input)
        tmp = self.policy_fc1(latent)
        hidden = self.act_fn(th.cat((tmp, -tmp), dim=-1))

        tmp = self.policy_fc2(hidden)
        hidden = self.act_fn(th.cat((tmp, -tmp), dim=-1))
        policy = self.policy_fc(hidden)

        if detach_value:
            latent = latent.detach()

        tmp = self.value_fc1(latent)
        hidden = self.act_fn(th.cat((tmp, -tmp), dim=-1))

        tmp = self.value_fc2(hidden)
        hidden = self.act_fn(th.cat((tmp, -tmp), dim=-1))

        value = self.value_fc(hidden)
       
        return value.squeeze(dim=1), policy
    
    def get_dist(self, policy, detach_std=False):
        if self.action_type == 'discrete':
            policy = Categorical(logits=policy[:,None])
        elif self.action_type == 'continuous':
            if detach_std:
                policy = tanh_normal(policy, self.policy_std.detach().exp().repeat(policy.shape[0], 1))
            else:
                policy = tanh_normal(policy, self.policy_std.exp().repeat(policy.shape[0], 1))
        return policy 

    def get_deterministic_action(self, policy):
        if self.action_type == 'discrete':
            return th.argmax(policy, dim=-1, keepdim=True)
        elif self.action_type == 'continuous':
            return policy

    def get_action_value(self, input, deterministic=False):
        with th.no_grad():
            value, policy = self.forward(input)
            policy_dist = self.get_dist(policy)
            if deterministic:
                action = self.get_deterministic_action(policy)
            else:
                action = policy_dist.sample()
            action_prob = policy_dist.log_prob(action).sum(axis=-1)
        return action.cpu().numpy(), action_prob.cpu().numpy(), value.cpu().numpy()

    def get_action(self, input, deterministic=False):
        with th.no_grad():
            _, policy = self.forward(input)
            policy_dist = self.get_dist(policy)
            if deterministic:
                action = self.get_deterministic_action(policy)
            else:
                action = policy_dist.sample()
        
        if self.action_type == 'discrete':
            action = action.squeeze(-1)

        return action.cpu().numpy()

