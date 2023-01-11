import math
import numpy as np
import torch as th
from torch import nn, Tensor
from torch.nn import functional as F

from .distributions import tanh_normal


LOG_WEIGHT = False
STD_MIN = -20
STD_MAX = 2


class DREPNet(nn.Module):
    def __init__(self, extractor, action_size, hidden_size):
        super().__init__()

        self.extractor = extractor

        self.action_size = action_size
        self.extractor = extractor
        self.in_size = extractor.out_size

        critic_in_size = extractor.out_size+action_size
        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size*2)
        )


        self.q1 = nn.Sequential(
            nn.Linear(critic_in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(critic_in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.log_alpha = nn.Parameter(th.tensor(-2.0, dtype=th.float32))


    def forward(self, state, goal):
        latent = F.normalize(goal - self.extractor(state), dim=-1)
        policy = self.actor(latent)
        policy_dist = self.get_dist(policy)
        
        act, act_pretanh = policy_dist.rsample(return_pretanh_value=True)
        act_prob = policy_dist.log_prob(act, act_pretanh).sum(axis=-1)

        for param in self.parameters():
            param.requires_grad = False

        value = self.get_q(state, goal, act)

        for param in self.parameters():
            param.requires_grad = True

        return act, act_prob, value


    def get_dist(self, policy):
        mean, log_std = policy.chunk(2, dim=-1)
        log_std = th.clamp(log_std, STD_MIN, STD_MAX)
        policy = tanh_normal(mean, log_std.exp())
        return policy 


    def get_action(self, state, goal):
        with th.no_grad():
            latent = goal - self.extractor(state)
            policy = self.actor(latent)
            policy_dist = self.get_dist(policy)
            action = policy_dist.sample()
        return action.cpu().numpy()


    def get_q(self, state, goal, action, sep=False):
        latent = goal - self.extractor(state)
        input = th.cat((latent, action), dim=-1)
        if sep:
            return self.q1(input), self.q2(input)
        return th.min(self.q1(input), self.q2(input))
