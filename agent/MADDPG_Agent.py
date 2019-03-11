# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from network import MADDPGPolicy
import torch
from components import tensor, get_maddpg_cfg_defaults, Task, OrnsteinUhlenbeckProcess, LinearSchedule, to_np, \
    ReplayBuffer, soft_update
import numpy as np
from collections import deque


hyper_parameter = get_maddpg_cfg_defaults().HYPER_PARAMETER.clone()


class MADDPGAgent:
    def __init__(self, env_path, writer):
        super(MADDPGAgent, self).__init__()
        # critic input = obs_full + actions = 24+2+2=28
        self.maddpg_agent = [MADDPGPolicy(24, 128, 64, 2, 52, 256, 128, seed=123),
                             MADDPGPolicy(24, 128, 64, 2, 52, 256, 128, seed=321)]
        self.state = None
        self.task = Task('Tennis', 1, env_path)
        self.random_process = OrnsteinUhlenbeckProcess(size=(hyper_parameter.ACTION_SPACE,), std=LinearSchedule(1000.0))
        self.episode_reward_agent1 = 0
        self.episode_reward_agent2 = 0
        self.episode_rewards = []
        self.episode_score_window = deque(maxlen=100)
        self.total_steps = 0
        self.replay = ReplayBuffer(action_size=hyper_parameter.ACTION_SPACE,
                                   buffer_size=hyper_parameter.REPLAY_BUFFER_SIZE,
                                   batch_size=hyper_parameter.BATCHSIZE,
                                   seed=112233)
        self.mse_loss = torch.nn.MSELoss()
        self.writer = writer
        self.summary_step = 0

    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        actions = [to_np(agent.act(obs)) + self.random_process.sample() for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def step(self):
        if self.state is None:
            self.random_process.reset_states()
            env_info = self.task.reset()
            self.state = env_info.vector_observations
        action = np.array(self.act(tensor(self.state)))
        next_states, rewards, terminals = self.task.step(action)
        self.episode_reward_agent1 += rewards[0]
        self.episode_reward_agent2 += rewards[1]
        self.replay.add(self.state, action, rewards, next_states, terminals.astype(np.uint8))
        self.summary_step += 1
        if np.any(terminals):
            max_rewards = max(self.episode_reward_agent1, self.episode_reward_agent2)
            self.episode_rewards.append(max_rewards)
            self.episode_score_window.append(max_rewards)
            if hyper_parameter.SAVE_INTERVAL and not self.total_steps % hyper_parameter.SAVE_INTERVAL \
                    and len(self.episode_rewards) and max(self.episode_rewards) > 0.5:
                self.save('saved_models/model-%s-%s.pth' % (self.__class__.__name__, str(len(self.episode_rewards) + 1)))
            if hyper_parameter.LOG_INTERVAL and not self.total_steps % hyper_parameter.LOG_INTERVAL and \
                    len(self.episode_score_window) == 100:
                none_zero = np.count_nonzero(self.episode_score_window)
                avg = np.mean(self.episode_score_window)
                # print(rewards[-100:])
                print('total steps %d, returns %.2f/%.2f/%.2f/%.2f (num/nonzero/max/avg_100)' % (
                    self.total_steps, len(self.episode_score_window), none_zero, np.max(self.episode_score_window), avg))

            self.episode_reward_agent1 = 0
            self.episode_reward_agent2 = 0
            self.random_process.reset_states()
            env_info = self.task.reset()
            self.total_steps += 1
            self.state = env_info.vector_observations


        else:
            self.state = next_states
        if len(self.replay) >= hyper_parameter.min_memory_size:
            experiences = self.replay.sample_maddpg()
            states1, states2, actions1, actions2, rewards1, rewards2, \
            next_states1, next_states2, dones1, dones2 = experiences
            next_action_agent1 = self.maddpg_agent[0].target_act(tensor(next_states1))
            np_next_action_agent1 = to_np(next_action_agent1)
            next_action_agent2 = self.maddpg_agent[1].target_act(tensor(next_states2))
            np_next_action_agent2 = to_np(next_action_agent2)
            next_critic_input_agent1 = np.hstack((next_states1, next_states2,
                                                  np_next_action_agent1, np_next_action_agent2))
            next_critic_input_agent2 = np.hstack((next_states2, next_states1,
                                                  np_next_action_agent2, np_next_action_agent1))
            with torch.no_grad():
                q_next_agent1 = self.maddpg_agent[0].target_critic_val(tensor(next_critic_input_agent1))
                q_next_agent2 = self.maddpg_agent[1].target_critic_val(tensor(next_critic_input_agent2))
            q_y_agent1 = tensor(rewards1) + hyper_parameter.GAMMA*q_next_agent1*(1 - tensor(dones1))
            q_y_agent2 = tensor(rewards2) + hyper_parameter.GAMMA*q_next_agent2*(1 - tensor(dones2))

            critic_input1 = np.hstack((states1, states2, actions1, actions2))
            critic_input2 = np.hstack((states2, states1, actions2, actions1))
            q_current_agent1 = self.maddpg_agent[0].critic_val(tensor(critic_input1))
            q_current_agent2 = self.maddpg_agent[1].critic_val(tensor(critic_input2))

            # agent critic loss
            critic_loss1 = self.mse_loss(q_y_agent1, q_current_agent1)
            # update agent1 critic network
            self.maddpg_agent[0].critic_optimizer.zero_grad()
            critic_loss1.backward()
            self.maddpg_agent[0].critic_optimizer.step()

            state1_tensor = tensor(states1)
            state2_tensor = tensor(states2)

            action_agent1 = self.maddpg_agent[0].act(state1_tensor)
            action_agent2 = self.maddpg_agent[1].act(state2_tensor)

            agent1_input = torch.cat((state1_tensor, state2_tensor, action_agent1, action_agent2.detach()), 1)
            agent2_input = torch.cat((state2_tensor, state1_tensor, action_agent2, action_agent1.detach()), 1)

            actor_agent1_loss = -self.maddpg_agent[0].critic_val(agent1_input).mean()
            actor_agent2_loss = -self.maddpg_agent[1].critic_val(agent2_input).mean()
            # update agent1 actor network
            self.maddpg_agent[0].actor_optimizer.zero_grad()
            actor_agent1_loss.backward()
            self.maddpg_agent[0].actor_optimizer.step()

            # update agent2 critic network
            critic_loss2 = self.mse_loss(q_y_agent2, q_current_agent2)
            self.maddpg_agent[1].critic_optimizer.zero_grad()
            critic_loss2.backward()
            self.maddpg_agent[1].critic_optimizer.step()
            # update agent2 actor network
            self.maddpg_agent[1].actor_optimizer.zero_grad()
            actor_agent2_loss.backward()
            self.maddpg_agent[1].actor_optimizer.step()

            soft_update(self.maddpg_agent[0].target_actor, self.maddpg_agent[0].actor,
                        hyper_parameter.target_network_mix)
            soft_update(self.maddpg_agent[1].target_actor, self.maddpg_agent[1].actor,
                        hyper_parameter.target_network_mix)
            soft_update(self.maddpg_agent[0].target_critic, self.maddpg_agent[0].critic,
                        hyper_parameter.target_network_mix)
            soft_update(self.maddpg_agent[1].target_critic, self.maddpg_agent[1].critic,
                        hyper_parameter.target_network_mix)

            agent1_critic_loss = critic_loss1.cpu().detach().item()
            agent2_critic_loss = critic_loss2.cpu().detach().item()
            agent1_actor_loss = actor_agent1_loss.cpu().detach().item()
            agent2_actor_loss = actor_agent2_loss.cpu().detach().item()
            # add summary writer
            self.writer.add_scalar('agent1/critic_loss', agent1_critic_loss, self.summary_step)
            self.writer.add_scalar('agent2/critic_loss', agent2_critic_loss, self.summary_step)
            self.writer.add_scalar('agent1/actor_loss', agent1_actor_loss, self.summary_step)
            self.writer.add_scalar('agent2/actor_loss', agent2_actor_loss, self.summary_step)

    def save(self, filename):
        torch.save({
            'agent1_actor': self.maddpg_agent[0].actor.state_dict(),
            'agent1_critic': self.maddpg_agent[0].critic.state_dict(),
            'agent2_actor': self.maddpg_agent[1].actor.state_dict(),
            'agent2_critic': self.maddpg_agent[1].critic.state_dict()
        }, filename)

    def close(self):
        self.task.close()
