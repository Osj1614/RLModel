import math
import numpy as np
import torch as th
from torch import nn, Tensor
from torch.nn import functional as F

from .distributions import tanh_normal
from .general import MultiLinear
from torch.distributions import Normal, Categorical, Gumbel


LOG_WEIGHT = False
STD_MIN = -20
STD_MAX = 2


class SACNet(nn.Module):
    def __init__(self, extractor, action_size):
        super().__init__()

        self.extractor = extractor

        self.action_size = action_size
        self.extractor = extractor
        self.in_size = extractor.out_size

        critic_in_size = extractor.out_size+action_size
        self.actor = nn.Sequential(
            nn.Linear(self.in_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, action_size*2)
        )


        self.q1 = nn.Sequential(
            nn.Linear(critic_in_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(critic_in_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        )

        self.log_alpha = nn.Parameter(th.tensor(-2.0, dtype=th.float32))


    def forward(self, state):
        latent = self.extractor(state)
        policy = self.actor(latent)
        policy_dist = self.get_dist(policy)
        
        act, act_pretanh = policy_dist.rsample(return_pretanh_value=True)
        act_prob = policy_dist.log_prob(act, act_pretanh).sum(axis=-1)

        for param in self.parameters():
            param.requires_grad = False

        value = self.get_q(latent, act)

        for param in self.parameters():
            param.requires_grad = True

        return act, act_prob, value


    def get_dist(self, policy):
        mean, log_std = policy.chunk(2, dim=-1)
        log_std = th.clamp(log_std, STD_MIN, STD_MAX)
        policy = tanh_normal(mean, log_std.exp())
        return policy 


    def get_action(self, state):
        with th.no_grad():
            latent = self.extractor(state)
            policy = self.actor(latent)
            policy_dist = self.get_dist(policy)
            action = policy_dist.sample()
        return action.cpu().numpy()


    def get_q(self, state, action, sep=False):
        latent = self.extractor(state)
        input = th.cat((latent, action), dim=-1)

        if sep:
            return self.q1(input), self.q2(input)
        return th.min(self.q1(input), self.q2(input))


class TransSACNet(nn.Module):
    def __init__(self, extractor, action_size):
        super().__init__()

        self.extractor = extractor
        self.action_size = action_size
        self.in_size = extractor.out_size

        d_model = 64
        self.state_embedding = MultiLinear(1, d_model, self.in_size)
        self.action_embedding = MultiLinear(1, d_model, self.action_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=d_model, dropout=0.0, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, 3)
        self.action_token_embedding = nn.Parameter(data=th.zeros((action_size, d_model)))
        self.q_token_embedding = nn.Parameter(data=th.zeros((2, d_model)))

        bound = 1 / math.sqrt(d_model)
        nn.init.uniform_(self.action_token_embedding, -bound, bound)
        nn.init.uniform_(self.q_token_embedding, -bound, bound)


        self.q1 = nn.Linear(d_model, 1)
        self.q2 = nn.Linear(d_model, 1)
        self.actor = MultiLinear(d_model, 2, action_size)

        self.log_alpha = nn.Parameter(th.tensor(-2.0, dtype=th.float32))


    def forward(self, state):
        batch_size = state.shape[0]
        latent = self.extractor(state)
        
        state_emb = self.state_embedding(latent.unsqueeze(dim=2))
        action_tokens = self.action_token_embedding.expand(batch_size, -1, -1)
        trans_latent = self.trans(th.cat((action_tokens, state_emb), dim=1))

        policy = self.actor(trans_latent[:, :self.action_size, :])
        policy_dist = self.get_dist(policy)
        
        act, act_pretanh = policy_dist.rsample(return_pretanh_value=True)
        act_prob = policy_dist.log_prob(act, act_pretanh).sum(axis=-1)


        for param in self.parameters():
            param.requires_grad = False

        value = self.get_q(state, act)

        for param in self.parameters():
            param.requires_grad = True
        

        return act, act_prob, value


    def get_dist(self, policy):
        mean, log_std = policy[..., 0], policy[..., 1]
        log_std = th.clamp(log_std, STD_MIN, STD_MAX)
        policy = tanh_normal(mean, log_std.exp())
        return policy 


    def get_action(self, state):
        batch_size = state.shape[0]
        with th.no_grad():
            latent = self.extractor(state)
            state_emb = self.state_embedding(latent.unsqueeze(dim=2))
            action_tokens = self.action_token_embedding.expand(batch_size, -1, -1)
            trans_latent = self.trans(th.cat((action_tokens, state_emb), dim=1))

            policy = self.actor(trans_latent[:, :self.action_size, :])
            policy_dist = self.get_dist(policy)
            action = policy_dist.sample()
        return action.cpu().numpy()


    def get_q(self, state, action, sep=False):
        batch_size = state.shape[0]
        latent = self.extractor(state)

        state_emb = self.state_embedding(latent.unsqueeze(dim=2))
        action_emb = self.action_embedding(action.unsqueeze(dim=2))
        q_tokens = self.q_token_embedding.expand(batch_size, -1, -1)

        tokens = th.cat((q_tokens, state_emb, action_emb), dim=1)
        trans_latent = self.trans(tokens)

        q1 = self.q1(trans_latent[:, 0, :])
        q2 = self.q2(trans_latent[:, 1, :])

        if sep:
            return q1, q2
        return th.min(q1, q2)


class TransSACNet2(nn.Module):
    def __init__(self, extractor, action_size):
        super().__init__()

        self.extractor = extractor
        self.action_size = action_size
        self.in_size = extractor.out_size

        
        self.d_model = 128
        self.n_state_token = 16
        self.state_embedding = nn.Sequential(
            nn.Linear(self.in_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model*self.n_state_token)
        )
        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model),
        )
        self.action_embedding2 = nn.Sequential(
            nn.Linear(self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(self.d_model, 4, dim_feedforward=self.d_model*4, dropout=0.0, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, 3)
        self.action_token_embedding = nn.Parameter(data=th.zeros((action_size, self.d_model)))

        bound = 1 / math.sqrt(self.d_model)
        nn.init.uniform_(self.action_token_embedding, -bound, bound)


        self.q1 = nn.Linear(self.d_model, 1)
        self.q2 = nn.Linear(self.d_model, 1)
        self.actor = MultiLinear(self.d_model, 2, action_size)

        self.log_alpha = nn.Parameter(th.tensor(-2.0, dtype=th.float32))


    def forward(self, state):
        batch_size = state.shape[0]
        latent = self.extractor(state)
        
        state_emb = self.state_embedding(latent).view(batch_size, self.n_state_token, self.d_model)
        action_tokens = self.action_token_embedding.expand(batch_size, -1, -1)
        trans_latent = self.trans(th.cat((action_tokens, state_emb), dim=1))

        policy = self.actor(trans_latent[:, :self.action_size, :])
        policy_dist = self.get_dist(policy)
        
        act, act_pretanh = policy_dist.rsample(return_pretanh_value=True)
        act_prob = policy_dist.log_prob(act, act_pretanh).sum(axis=-1)


        for param in self.parameters():
            param.requires_grad = False

        value = self.get_q(state, act)

        for param in self.parameters():
            param.requires_grad = True
        

        return act, act_prob, value


    def get_dist(self, policy):
        mean, log_std = policy[..., 0], policy[..., 1]
        log_std = th.clamp(log_std, STD_MIN, STD_MAX)
        policy = tanh_normal(mean, log_std.exp())
        return policy 


    def get_action(self, state):
        batch_size = state.shape[0]
        with th.no_grad():
            latent = self.extractor(state)
            
            state_emb = self.state_embedding(latent).view(batch_size, self.n_state_token, self.d_model)
            action_tokens = self.action_token_embedding.expand(batch_size, -1, -1)
            trans_latent = self.trans(th.cat((action_tokens, state_emb), dim=1))


            policy = self.actor(trans_latent[:, :self.action_size, :])
            policy_dist = self.get_dist(policy)
            action = policy_dist.sample()
        return action.cpu().numpy()


    def get_q(self, state, action, sep=False):
        batch_size = state.shape[0]
        latent = self.extractor(state)

        state_emb = self.state_embedding(latent).view(batch_size, self.n_state_token, self.d_model)
        action_emb1 = self.action_embedding(action).unsqueeze(dim=1)
        action_emb2 = self.action_embedding2(action).unsqueeze(dim=1)

        tokens = th.cat((action_emb1, action_emb2, state_emb), dim=1)
        trans_latent = self.trans(tokens)

        q1 = self.q1(trans_latent[:, 0, :])
        q2 = self.q2(trans_latent[:, 1, :])

        if sep:
            return q1, q2
        return th.min(q1, q2)


class TransSACNet3(nn.Module):
    def __init__(self, extractor, action_size):
        super().__init__()

        self.extractor = extractor
        self.action_size = action_size
        self.in_size = extractor.out_size
        self.n_task = extractor.n_task
        
        self.d_model = 128
        self.n_state_token = 16

        self.state_embedding = nn.Sequential(
            nn.Linear(self.in_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model*self.n_state_token)
        )


        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model),
        )
        
        self.action_embedding2 = nn.Sequential(
            nn.Linear(self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(self.d_model, 4, dim_feedforward=self.d_model*2, dropout=0.0, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, 2)
        self.task_token_embedding = nn.Linear(self.n_task, self.d_model)

        
        decoder_layer = nn.TransformerDecoderLayer(self.d_model, 4, dim_feedforward=self.d_model*2, dropout=0.0, batch_first=True)
        self.q_trans = nn.TransformerDecoder(decoder_layer, 2)


        self.q1 = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size*2),
        )
        

        self.log_alpha = nn.Parameter(th.tensor(-2.0, dtype=th.float32))


    def forward(self, state):
        batch_size = state.shape[0]
        latent, task_onehot = self.extractor(state)

        state_emb = self.state_embedding(latent).view(batch_size, self.n_state_token, self.d_model)
        task_token = self.task_token_embedding(task_onehot).unsqueeze(dim=1)
        trans_tokens = self.trans(th.cat((task_token, state_emb), dim=1))

        policy = self.actor(trans_tokens[:, 0, :])
        policy_dist = self.get_dist(policy)
        
        act, act_pretanh = policy_dist.rsample(return_pretanh_value=True)
        act_prob = policy_dist.log_prob(act, act_pretanh).sum(axis=-1)


        for param in self.parameters():
            param.requires_grad = False

        value = self.get_q(state, act, trans_tokens=trans_tokens)

        for param in self.parameters():
            param.requires_grad = True
        

        return act, act_prob, value


    def get_dist(self, policy):
        mean, log_std = policy.chunk(2, dim=-1)
        log_std = th.clamp(log_std, STD_MIN, STD_MAX)
        policy = tanh_normal(mean, log_std.exp())
        return policy 


    def get_action(self, state):
        batch_size = state.shape[0]
        with th.no_grad():
            latent, task_onehot = self.extractor(state)

            state_emb = self.state_embedding(latent).view(batch_size, self.n_state_token, self.d_model)
            task_token = self.task_token_embedding(task_onehot).unsqueeze(dim=1)
            trans_tokens = self.trans(th.cat((task_token, state_emb), dim=1))

            policy = self.actor(trans_tokens[:, 0, :])
            policy_dist = self.get_dist(policy)
            action = policy_dist.sample()
        return action.cpu().numpy()


    def get_q(self, state, action, sep=False, trans_tokens=None):
        batch_size = state.shape[0]

        if trans_tokens == None:
            latent, task_onehot = self.extractor(state)

            state_emb = self.state_embedding(latent).view(batch_size, self.n_state_token, self.d_model)
            task_token = self.task_token_embedding(task_onehot).unsqueeze(dim=1)
            trans_tokens = self.trans(th.cat((task_token, state_emb), dim=1))

        action_emb1 = self.action_embedding(action).unsqueeze(dim=1)
        action_emb2 = self.action_embedding2(action).unsqueeze(dim=1)

        tokens = th.cat((action_emb1, action_emb2), dim=1)
        q_trans_tokens = self.q_trans(tokens, trans_tokens)

        q1 = self.q1(q_trans_tokens[:, 0, :])
        q2 = self.q2(q_trans_tokens[:, 1, :])

        if sep:
            return q1, q2
        return th.min(q1, q2)