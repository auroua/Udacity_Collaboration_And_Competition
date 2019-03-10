# individual network settings for each actor + critic pair
# see networkforall for details

from .NetWork import Network
from components import hard_update, OrnsteinUhlenbeckProcess, device, LinearSchedule
from torch.optim import Adam


class MADDPGPolicy:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic,
                 hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(MADDPGPolicy, self).__init__()
        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, obs):
        obs = obs.to(device)
        action = self.actor(obs)
        return action

    def target_act(self, obs):
        obs = obs.to(device)
        action = self.target_actor(obs)
        return action

    def critic_val(self, obs):
        obs = obs.to(device)
        critic_val = self.critic(obs)
        return critic_val

    def target_critic_val(self, obs):
        obs = obs.to(device)
        target_critic_val = self.target_critic(obs)
        return target_critic_val