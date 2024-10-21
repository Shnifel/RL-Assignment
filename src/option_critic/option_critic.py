# Code adapted from : https://github.com/lweitkamp/option-critic-pytorch/

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli, Normal

from math import exp
import numpy as np

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                action_dim,  # Continuous action space (num_actions -> action_dim)
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.action_dim = action_dim  # Continuous action dimension
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.Q            = nn.Linear(64, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)  # Option-Termination
        self.options_mean = nn.Parameter(torch.zeros(num_options, 64, action_dim))
        self.options_log_std = nn.Parameter(torch.zeros(num_options, action_dim))  # Log of standard deviation

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = torch.bernoulli(termination)
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        # Get mean and log_std for the selected option
        mean = state.data @ self.options_mean[option]  # Mean of action distribution
        log_std = self.options_log_std[option].clamp(-20, 2)  # Log std deviation, clamped for numerical stability
        std = log_std.exp()  # Convert log std to std

        # Sample from Normal distribution
        action_dist = Normal(mean, std)
        action = action_dist.sample()
        logp = action_dist.log_prob(action).sum(-1)  # Sum log probs for multi-dimensional actions
        entropy = action_dist.entropy().sum(-1)  # Sum entropy over action dimensions

        return action, logp, entropy

    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q      = model.get_Q(states)
    
    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime      = model_prime.get_Q(next_states_prime) # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states            = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err

def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args):
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss