# individual network settings for each actor + critic pair
# see networkforall for details

from .NetWork import Network
from components import hard_update, device
from torch.optim import Adam, SGD
import random
import torch
import numpy as np
from components import OrnsteinUhlenbeckProcess, LinearSchedule, get_maddpg_cfg_defaults

hyper_parameter = get_maddpg_cfg_defaults().HYPER_PARAMETER.clone()


class MADDPGPolicy:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic,
                 hidden_out_critic, lr_actor=1.0e-4, lr_critic=1.0e-4, seed=0):
        super(MADDPGPolicy, self).__init__()
        random.seed(seed)
        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True, seed=seed).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True, seed=seed).to(device)

        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, seed=seed).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, seed=seed).to(device)

        self.random_process = OrnsteinUhlenbeckProcess(size=(hyper_parameter.ACTION_SPACE,), std=LinearSchedule(0.2))

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)

        # self.actor_optimizer = SGD(self.actor.parameters(), lr=lr_actor, momentum=0.9)
        # self.critic_optimizer = SGD(self.critic.parameters(), lr=lr_critic, momentum=0.9)

    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = state.to(device)
        # calculate action values
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            noise_val = self.random_process.sample() * noise_weight
            action += noise_val
        return np.clip(action, -1, 1)

    def reset(self):
        self.random_process.reset_states()