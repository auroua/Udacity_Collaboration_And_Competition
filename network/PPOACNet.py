import torch
import torch.nn as nn
import torch.nn.functional as F
from components import get_ppo_ac_cfg_defaults, device, layer_init

hyper_parameter = get_ppo_ac_cfg_defaults().HYPER_PARAMETER.clone()
model_parameter = get_ppo_ac_cfg_defaults().MODEL_PARAMETER.clone()


class ActorNet(nn.Module):
    def __init__(self, gate=torch.tanh):
        super(ActorNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(hyper_parameter.STATE_SPACE, model_parameter.H1))
        self.layer2 = layer_init(nn.Linear(model_parameter.H1, model_parameter.H2))
        self.gate = gate

    def forward(self, inputs):
        x = self.gate(self.layer1(inputs))
        x = self.gate(self.layer2(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, gate=torch.tanh):
        super(CriticNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(hyper_parameter.STATE_SPACE, model_parameter.H1))
        self.layer2 = layer_init(nn.Linear(model_parameter.H1, model_parameter.H2))
        self.gate = gate

    def forward(self, inputs):
        x = self.gate(self.layer1(inputs))
        x = self.gate(self.layer2(x))
        return x


class PPOACNet(nn.Module):
    def __init__(self):
        super(PPOACNet, self).__init__()
        self.actor_body = ActorNet()
        self.critic_body = CriticNet()
        self.fc_actor = layer_init(nn.Linear(model_parameter.H2, hyper_parameter.ACTION_SPACE), 1e-3)
        self.fc_critic = layer_init(nn.Linear(model_parameter.H2, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros(hyper_parameter.ACTION_SPACE))
        self.to(device)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_actor.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.total_params = self.actor_params + self.critic_params
        self.total_params.append(self.std)

    def forward(self, x, action=None):
        if action is None:
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)
        actor_out = self.actor_body(x)
        critic_out = self.critic_body(x)
        mean = torch.tanh(self.fc_actor(actor_out))
        v = self.fc_critic(critic_out)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        prob = torch.exp(log_prob)
        entropy = dist.entropy()
        return {
            'a': action,
            'log_pi_a': log_prob,
            'prob': prob,
            'mean': mean,
            'ent': entropy,
            'v': v
        }
