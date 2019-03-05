import torch
from network import PPOACNet
from components import get_ppo_ac_cfg_defaults, Task, Storage, tensor, to_np, random_sample
import numpy as np

hyper_parameter = get_ppo_ac_cfg_defaults().HYPER_PARAMETER.clone()
train_parameter = get_ppo_ac_cfg_defaults().TRAIN_PARAMETER.clone()


class PPOACAgent:
    def __init__(self, env_path):
        super(PPOACAgent, self).__init__()
        self.policy = PPOACNet()
        self.task = Task('Reacher', 20, env_path)
        self.states = self.task.reset().vector_observations
        self.rollout_length = hyper_parameter.ROLLOUT_LENGTH
        self.online_rewards = np.zeros(hyper_parameter.AGENTS_NUM)
        self.episode_rewards = []
        self.total_steps = 0
        # self.optimizer = torch.optim.RMSprop(self.policy.total_params, lr=0.0007)
        self.optimizer = torch.optim.Adam(self.policy.total_params, lr=1e-4, eps=1e-5)

    def step(self):
        storage = Storage(self.rollout_length)
        states = self.states
        for _ in range(self.rollout_length):
            prediction = self.policy(states)
            next_states, rewards, terminals = self.task.step(to_np(prediction['a']))
            self.online_rewards += rewards
            for i, terminal in enumerate(terminals):
                if terminal:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})

            states = next_states
        self.states = states
        prediction = self.policy(states)
        storage.add(prediction)
        storage.placeholder()

        returns = prediction['v'].detach()
        advantages = tensor(np.zeros((hyper_parameter.AGENTS_NUM, 1)))
        for i in reversed(range(self.rollout_length)):
            returns = storage.r[i] + hyper_parameter.GAMMA * storage.m[i] * returns
            if not hyper_parameter.USE_GAE:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + hyper_parameter.GAMMA * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * hyper_parameter.GAE_TAU * hyper_parameter.GAMMA * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(hyper_parameter.SURROGATE):
            sampler = random_sample(np.arange(states.size(0)), hyper_parameter.BATCHSIZE)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.policy(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - hyper_parameter.CLIP_VAL,
                                          1.0 + hyper_parameter.CLIP_VAL) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - hyper_parameter.entropy_weight * prediction['ent'].mean()
                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.total_params, train_parameter.Gradient_Clip)
                self.optimizer.step()

        steps = self.rollout_length * hyper_parameter.AGENTS_NUM
        self.total_steps += steps

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def close(self):
        self.task.close()


