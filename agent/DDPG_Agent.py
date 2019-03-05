import torch
from network import DDPGNet
from components import device, get_ddpg_cfg_defaults, Replay, OrnsteinUhlenbeckProcess, LinearSchedule, \
    Task, to_np, tensor
import numpy as np


hyper_parameter = get_ddpg_cfg_defaults().HYPER_PARAMETER.clone()


class DDPGAgent:
    def __init__(self, env_path):
        self.policy_net = DDPGNet(actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
                                  critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))
        self.target_net = DDPGNet(actor_opt_fn=None, critic_opt_fn=None)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.task = Task('Reacher', 20, env_path)
        self.replay = Replay(memory_size=int(1e6), batch_size=64)
        self.random_process = OrnsteinUhlenbeckProcess(size=(hyper_parameter.ACTION_SPACE, ),
                                                       std=LinearSchedule(0.2))
        self.state = None
        self.online_rewards = np.zeros(hyper_parameter.AGENTS_NUM)
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - hyper_parameter.target_network_mix) +
                               param * hyper_parameter.target_network_mix)

    def step(self):
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset().vector_observations
        x_input = tensor(self.state).to(device)
        action = self.policy_net(x_input)
        action = to_np(action)
        action += self.random_process.sample()
        next_states, rewards, terminals = self.task.step(action)
        # next_states = np.squeeze(next_states, 0)
        # rewards = np.squeeze(rewards, 0)
        # terminals = np.squeeze(terminals, 0)

        self.episode_reward += rewards[0]
        self.replay.feed([self.state, action, rewards, next_states, terminals.astype(np.uint8)])
        if terminals[0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.random_process.reset_states()
        self.state = next_states
        self.total_steps += 1

        if self.replay.size() >= hyper_parameter.min_memory_size:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = states.squeeze(1)
            actions = actions.squeeze(1)
            rewards = tensor(rewards)
            next_states = next_states.squeeze(1)
            terminals = tensor(terminals)

            next_states = tensor(next_states)
            next_states = next_states.to(device)

            a_next = self.target_net(next_states)
            q_next = self.target_net.critic(next_states, a_next)

            q_next = hyper_parameter.GAMMA * q_next * (1 - terminals)
            q_next.add_(rewards)
            q_next = q_next.detach()

            states = tensor(states)
            states = states.to(device)
            q = self.policy_net.critic(states, tensor(actions))
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.policy_net.zero_grad()
            critic_loss.backward()
            self.policy_net.critic_opt.step()

            action = self.policy_net(states)
            policy_loss = -self.policy_net.critic(states.detach(), action).mean()

            self.policy_net.zero_grad()
            policy_loss.backward()
            self.policy_net.actor_opt.step()

            self.soft_update(self.target_net, self.policy_net)
