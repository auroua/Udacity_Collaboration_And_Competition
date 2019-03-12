# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from network import MADDPGPolicy
import torch
from components import tensor, get_maddpg_cfg_defaults, Task, to_np, soft_update, ReplayBuffer
import numpy as np
from collections import deque


hyper_parameter = get_maddpg_cfg_defaults().HYPER_PARAMETER.clone()
model_parameter = get_maddpg_cfg_defaults().MODEL_PARAMETER.clone()


class MADDPGAgent:
    def __init__(self, env_path, writer):
        super(MADDPGAgent, self).__init__()
        # critic input = obs_full + actions = 24+2+2=28
        self.agent1 = MADDPGPolicy(24, model_parameter.H1, model_parameter.H2, 2, 52, model_parameter.H1,
                                   model_parameter.H2, seed=123)
        self.agent2 = MADDPGPolicy(24, model_parameter.H1, model_parameter.H2, 2, 52, model_parameter.H1,
                                   model_parameter.H2, seed=321)
        self.agents = [self.agent1, self.agent2]
        self.state = None
        self.task = Task('Tennis', 1, env_path)
        self.episode_reward_agent1 = 0
        self.episode_reward_agent2 = 0
        self.episode_rewards = []
        self.avg_results = []
        self.episode_score_window = deque(maxlen=100)
        self.total_steps = 0
        self.replay = ReplayBuffer(action_size=hyper_parameter.ACTION_SPACE,
                                   buffer_size=hyper_parameter.REPLAY_BUFFER_SIZE,
                                   batch_size=hyper_parameter.BATCHSIZE,
                                   seed=112233)
        self.mse_loss = torch.nn.MSELoss()
        self.writer = writer
        self.summary_step = 0
        self.flag = False

    def step(self):
        if self.state is None:
            self.agent1.reset()
            self.agent2.reset()
            env_info = self.task.reset()
            self.state = env_info.vector_observations
        if self.total_steps < 1200:
            action = np.vstack(tuple([agent.act(tensor(self.state[idx::2, :]))
                                      for idx, agent in enumerate([self.agent1, self.agent2])]))
        elif self.total_steps < 1200*1.75 and np.random.randint(1, 10) <=5:
            action = np.vstack(tuple([agent.act(tensor(self.state[idx::2, :]))
                                      for idx, agent in enumerate([self.agent1, self.agent2])]))
        else:
            action = np.vstack(tuple(
                [agent.act(tensor(self.state[idx::2, :]), add_noise=False)
                 for idx, agent in enumerate([self.agent1, self.agent2])]))
        next_states, rewards, terminals = self.task.step(action)
        self.episode_reward_agent1 += rewards[0]
        self.episode_reward_agent2 += rewards[1]
        self.replay.add(self.state, action, rewards, next_states, terminals.astype(np.uint8))
        self.summary_step += 1
        if np.any(terminals):
            max_rewards = max(self.episode_reward_agent1, self.episode_reward_agent2)

            self.episode_rewards.append(max_rewards)
            self.episode_score_window.append(max_rewards)
            none_zero = np.count_nonzero(self.episode_score_window)
            self.episode_reward_agent1 = 0
            self.episode_reward_agent2 = 0
            self.agent1.reset()
            self.agent2.reset()
            env_info = self.task.reset()
            self.total_steps += 1
            self.state = env_info.vector_observations
            if self.flag:
                avg = np.mean(self.episode_score_window)
                self.avg_results.append(avg)
                print('episodes %d, returns %.2f/%.2f/%.2f/%.2f (num/nonzero/max/avg)' % (
                    self.total_steps, len(self.episode_score_window), none_zero, np.max(self.episode_score_window), avg))
                if hyper_parameter.SAVE_INTERVAL and not self.total_steps % hyper_parameter.SAVE_INTERVAL \
                        and len(self.episode_rewards) and avg >= 0.5:
                    self.save('saved_models/model-%s-%s.pth' % (self.__class__.__name__, str(len(self.episode_rewards) + 1)))
        else:
            self.state = next_states

        if len(self.replay) >= hyper_parameter.min_memory_size and \
                self.summary_step % hyper_parameter.UPDATE_FREQ == 0:
            self.flag = True
            experiences_ag1 = self.replay.sample_maddpg()
            experiences_ag2 = self.replay.sample_maddpg()
            actor_total_loss = []
            critic_total_loss = []
            for i, experience in enumerate([experiences_ag1, experiences_ag2]):
                # update agent1 critic
                states1, states2, actions1, actions2, rewards1, rewards2, \
                next_states1, next_states2, dones1, dones2 = experience
                curr_agnet = self.agents[i]
                opp_agent = self.agents[1-i]
                next_action_agent1 = curr_agnet.target_actor(tensor(next_states1))
                np_next_action_agent1 = to_np(next_action_agent1)
                next_action_agent2 = opp_agent.target_actor(tensor(next_states2))
                np_next_action_agent2 = to_np(next_action_agent2)
                next_critic_input_agent1 = np.hstack((next_states1, next_states2,
                                                      np_next_action_agent1, np_next_action_agent2))
                with torch.no_grad():
                    q_next_agent1 = curr_agnet.target_critic(tensor(next_critic_input_agent1))
                    q_y_agent1 = tensor(rewards1) + hyper_parameter.GAMMA*q_next_agent1*(1 - tensor(dones1))
                critic_input1 = np.hstack((states1, states2, actions1, actions2))
                q_current_agent1 = curr_agnet.critic(tensor(critic_input1))
                critic_loss1 = self.mse_loss(q_y_agent1, q_current_agent1)
                # update agent1 critic network
                curr_agnet.critic_optimizer.zero_grad()
                critic_loss1.backward()
                curr_agnet.critic_optimizer.step()

                state1_tensor = tensor(states1)
                state2_tensor = tensor(states2)
                action_agent1 = curr_agnet.actor(state1_tensor)
                action_agent2 = opp_agent.actor(state2_tensor).detach()
                agent1_input = torch.cat((state1_tensor, state2_tensor, action_agent1, action_agent2), 1)
                actor_agent1_loss = -curr_agnet.critic(agent1_input).mean()

                curr_agnet.actor_optimizer.zero_grad()
                actor_agent1_loss.backward()
                curr_agnet.actor_optimizer.step()

                soft_update(curr_agnet.target_actor, curr_agnet.actor, hyper_parameter.target_network_mix)
                soft_update(curr_agnet.target_critic, curr_agnet.critic, hyper_parameter.target_network_mix)

                actor_total_loss.append(actor_agent1_loss)
                critic_total_loss.append(critic_loss1)

            # soft_update(self.agent1.target_actor, self.agent1.actor, hyper_parameter.target_network_mix)
            # soft_update(self.agent1.target_critic, self.agent1.critic, hyper_parameter.target_network_mix)
            # soft_update(self.agent2.target_actor, self.agent2.actor, hyper_parameter.target_network_mix)
            # soft_update(self.agent2.target_critic, self.agent2.critic, hyper_parameter.target_network_mix)

            agent1_critic_loss = critic_total_loss[0].cpu().detach().item()
            agent2_critic_loss = critic_total_loss[1].cpu().detach().item()
            agent1_actor_loss = actor_total_loss[0].cpu().detach().item()
            agent2_actor_loss = actor_total_loss[1].cpu().detach().item()
            # add summary writer
            self.writer.add_scalar('agent1/critic_loss', agent1_critic_loss, self.summary_step)
            self.writer.add_scalar('agent2/critic_loss', agent2_critic_loss, self.summary_step)
            self.writer.add_scalar('agent1/actor_loss', agent1_actor_loss, self.summary_step)
            self.writer.add_scalar('agent2/actor_loss', agent2_actor_loss, self.summary_step)

    def save(self, filename):
        torch.save({
            'agent1_actor': self.agent1.actor.state_dict(),
            'agent1_critic': self.agent1.critic.state_dict(),
            'agent2_actor': self.agent2.actor.state_dict(),
            'agent2_critic': self.agent2.critic.state_dict()
        }, filename)

    def close(self):
        self.task.close()
