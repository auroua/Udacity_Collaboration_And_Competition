import torch
import torch.nn as nn
import torch.nn.functional as F
from components import layer_init, get_ppo_ac_cfg_defaults, device, tensor

hyper_parameter = get_ppo_ac_cfg_defaults().HYPER_PARAMETER.clone()
model_parameter = get_ppo_ac_cfg_defaults().MODEL_PARAMETER.clone()


class ActorNet(nn.Module):
    def __init__(self, gate=F.relu):
        super(ActorNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(hyper_parameter.STATE_SPACE, model_parameter.H1))
        self.layer2 = layer_init(nn.Linear(model_parameter.H1, model_parameter.H2))
        self.gate = gate

    def forward(self, inputs):
        x = self.gate(self.layer1(inputs))
        x = self.gate(self.layer2(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, gate=F.relu):
        super(CriticNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(hyper_parameter.STATE_SPACE, model_parameter.H1))
        self.layer2 = layer_init(nn.Linear(model_parameter.H1 + hyper_parameter.ACTION_SPACE, model_parameter.H2))
        self.gate = gate

    def forward(self, inputs, action):
        x = self.gate(self.layer1(inputs))
        x = self.gate(self.layer2(torch.cat((x, action), 1)))
        return x


class DDPGNet(nn.Module):
    def __init__(self, actor_opt_fn, critic_opt_fn):
        super(DDPGNet, self).__init__()
        self.actor_body = ActorNet()
        self.critic_body = CriticNet()
        self.fc_actor = layer_init(nn.Linear(model_parameter.H2, hyper_parameter.ACTION_SPACE), 1e-3)
        self.fc_critic = layer_init(nn.Linear(model_parameter.H2, 1), 1e-3)
        self.to(device)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_actor.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.total_params = self.actor_params + self.critic_params

        if actor_opt_fn is not None:
            self.actor_opt = actor_opt_fn(self.actor_params)
        if critic_opt_fn is not None:
            self.critic_opt = critic_opt_fn(self.critic_params)

    def forward(self, x, action=None):
        actor_out = torch.tanh(self.fc_actor(self.actor_body(x)))
        return actor_out

    def critic(self, x, a):
        return self.fc_critic(self.critic_body(x, a))
