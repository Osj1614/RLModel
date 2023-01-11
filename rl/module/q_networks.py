import math
import numpy as np
import torch as th
from torch import nn, Tensor
from torch.nn import functional as F

from .distributions import tanh_normal
from torch.distributions import Normal, Categorical, Gumbel


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.w_mu = nn.Parameter(th.Tensor(out_features, in_features))
        self.w_sigma = nn.Parameter(th.Tensor(out_features, in_features))
        self.register_buffer('w_eps', th.Tensor(out_features, in_features))

        self.b_mu = nn.Parameter(th.Tensor(out_features))
        self.b_sigma = nn.Parameter(th.Tensor(out_features))
        self.register_buffer('b_eps', th.Tensor(out_features))
        
        mu_range = 1 / math.sqrt(self.in_features)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.w_sigma.data.fill_(self.std_init * mu_range)
        self.b_mu.data.uniform_(-mu_range, mu_range)
        self.b_sigma.data.fill_(self.std_init * mu_range)

    
    def reset_noise(self):
        epsilon_in = self.factorized_gaussian_noise(self.in_features)
        epsilon_out = self.factorized_gaussian_noise(self.out_features)
        self.w_eps.copy_(epsilon_out.ger(epsilon_in))
        self.b_eps.copy_(epsilon_out)

    def factorized_gaussian_noise(self, size):
        x = th.randn(size, device=self.w_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x, deterministic=False):
        if deterministic:
            return F.linear(x, self.w_mu, self.b_mu)
        return F.linear(x, self.w_mu+self.w_sigma*self.w_eps, self.b_mu+self.b_sigma*self.b_eps)


class QNet(nn.Module):
    def __init__(self, action_size, extractor, num_quantiles=100, act_fn=F.relu):
        super().__init__()
        self.extractor = extractor

        self.action_size = action_size
        self.in_size = extractor.out_size
        self.num_quantiles = num_quantiles

        self.act_fn = act_fn
        self.adv_fc1 = NoisyLinear(self.in_size, 256)
        self.value_fc1 = NoisyLinear(self.in_size, 256)
        self.adv_fc2 = NoisyLinear(256, action_size*num_quantiles)
        self.value_fc2 = NoisyLinear(256, num_quantiles)

    def forward(self, input, deterministic=False):
        hidden = self.extractor(input)
        adv_hidden = self.act_fn(self.adv_fc1(hidden, deterministic))
        value_hidden = self.act_fn(self.value_fc1(hidden, deterministic))
        advantages = self.adv_fc2(adv_hidden, deterministic).view(-1, self.action_size, self.num_quantiles)
        values = self.value_fc2(value_hidden, deterministic).view(-1, 1, self.num_quantiles)

        q_values = values + advantages - th.mean(advantages, dim=1, keepdim=True)
        return q_values

    def reset_noise(self):
        self.adv_fc1.reset_noise()
        self.value_fc1.reset_noise()
        self.adv_fc2.reset_noise()
        self.value_fc2.reset_noise()

    def get_avg_q(self, q_values):
        return th.mean(q_values, dim=-1)

    def get_best_action(self, input, deterministic=False):
        q_values = self.forward(input, deterministic)
        return th.argmax(self.get_avg_q(q_values), dim=1)
    
    def get_action(self, input, deterministic=False):
        return self.get_best_action(input, deterministic).cpu().numpy()

    def get_q_sa(self, input, action):
        q_values = self.forward(input)
        return th.sum(q_values * F.one_hot(action, self.action_size)[:, :, None], dim=1)


class VQNet(nn.Module):
    def __init__(self, feat_dim, groups, entries, code_dim, out_size):
        super().__init__()
        self.feat_dim = feat_dim
        self.groups = groups
        self.entries = entries
        self.out_size = out_size
        self.weight_proj = nn.Linear(feat_dim, groups*entries)
        self.codebooks = nn.ModuleList([nn.Linear(entries, code_dim, bias=False) for _ in range(groups)])
        self.out_fc = nn.Linear(code_dim*groups, out_size)

        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

    def forward(self, input):
        logits = self.weight_proj(input)
        one_hots = logit_to(logits, 'gumbel_softmax', self.entries, flatten=False, temperature=0.1)
        codes = []
        for g in range(self.groups):
            codes.append(self.codebooks[g](one_hots.select(dim=-2, index=g)))
        full_code = th.cat(codes, dim=-1)
        probs = th.mean(logit_to(logits, 'prob', self.entries, flatten=False), dim=(0, 1))
        perplexity = th.exp(-th.sum(probs * th.log(probs+1e-7), dim=-1)) / self.entries
        return self.out_fc(full_code), perplexity

