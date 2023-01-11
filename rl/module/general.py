from typing import Optional, Any

import math
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical, Gumbel


def logit_to(latent, mode, classes, flatten=True, temperature=0.1):
    batch_size = latent.shape[:-1]
    out_size = latent.shape[-1]
    logits = latent.view(batch_size+(out_size//classes, classes))
    if mode == 'logprob':
        ret = F.log_softmax(logits, dim=-1)
    elif mode == 'prob':
        ret = F.softmax(logits, dim=-1)
    elif mode == 'sample':
        dist = Categorical(logits=logits/temperature)
        prob_latent = dist.probs
        ret = F.one_hot(dist.sample(), num_classes=classes) + prob_latent - prob_latent.detach()
    elif mode == 'gumbel_softmax':
        ret = F.gumbel_softmax(logits, tau=temperature, hard=True)
    elif mode == 'deterministic':
        prob_latent = F.softmax(logits, dim=-1)
        argmax = th.argmax(prob_latent, dim=-1)
        one_hot = th.zeros_like(prob_latent).scatter_(-1, argmax.unsqueeze(-1), 1.)
        ret = one_hot + prob_latent - prob_latent.detach()

    if flatten:
        return ret.view(batch_size+(out_size,))
    else:
        return ret



class DummyExtractor(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = (in_shape,)
        self.out_size = in_shape
    
    def forward(self, x):
        return x


class MLPExtractor(nn.Module):
    def __init__(self, in_shape, out_size, hidden=64):
        super().__init__()
        if in_shape is tuple:
            in_shape = in_shape[0]
        self.in_size = in_shape
        self.out_size = out_size
        self.encoder = nn.Sequential(
            nn.Linear(self.in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.out_size)
        )
    
    def forward(self, x):
        return self.encoder(x)


class TaskExtractor(nn.Module):
    def __init__(self, in_size, n_task):
        super().__init__()
        self.in_shape = (in_size-n_task,)
        self.out_size = in_size-n_task
        self.n_task = n_task
    
    def forward(self, x):
        return th.split(x, (self.out_size, self.n_task), dim=1)


class MultiLinear(nn.Module):
    def __init__(self, in_size, out_size, heads):
        super().__init__()
        self.weight = nn.Parameter(th.zeros((heads, out_size, in_size)))
        self.bias = nn.Parameter(th.zeros(heads, out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # x : b, h, in, 1
    # weight : ?, h, out, in
    def forward(self, x):
        # b, h, out
        return th.matmul(self.weight, x.unsqueeze(-1)).squeeze(-1) + self.bias


        

class CnnEncoder(nn.Module):
    def __init__(self, feat_dim, out_size=128):
        super().__init__()
        self.act_fn = F.relu
        self.stride = 24
        self.receptive_field = 30
        self.conv1 = nn.Conv1d(feat_dim, 256, 6, 3)
        self.conv2 = nn.Conv1d(256, 256, 3, 2)
        self.conv3 = nn.Conv1d(256, 256, 2, 2)
        self.conv4 = nn.Conv1d(256, out_size, 2, 2)

    def forward(self, x:th.Tensor):
        hidden = self.act_fn(self.conv1(x))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        return self.conv4(hidden)

