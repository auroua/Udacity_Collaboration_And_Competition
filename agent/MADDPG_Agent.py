# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from network import MADDPGPolicy
import torch
from components import device, tensor, get_maddpg_cfg_defaults, Task, OrnsteinUhlenbeckProcess, LinearSchedule, to_np, \
    ReplayBuffer
import numpy as np

hyper_parameter = get_maddpg_cfg_defaults().HYPER_PARAMETER.clone()


class MADDPGAgent:
    def __init__(self, env_path, tau=0.02):
        super(MADDPGAgent, self).__init__()
        # critic input = obs_full + actions = 24+2+2=28
        self.maddpg_agent = [MADDPGPolicy(24, 64, 32, 2, 52, 64, 32),
                             MADDPGPolicy(24, 64, 32, 2, 52, 64, 32)]
        self.discount_factor = hyper_parameter.GAMMA
        self.state = None
        self.task = Task('Tennis', 20, env_path)
        self.random_process = OrnsteinUhlenbeckProcess(size=(hyper_parameter.ACTION_SPACE, ), std=LinearSchedule(0.2))
        self.episode_reward_agent1 = 0
        self.episode_reward_agent2 = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.replay = ReplayBuffer(action_size=hyper_parameter.ACTION_SPACE,
                                   buffer_size=hyper_parameter.REPLAY_BUFFER_SIZE,
                                   batch_size=hyper_parameter.BATCHSIZE,
                                   seed=112233)

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        actions = [to_np(agent.act(obs)) + self.random_process.sample() for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs) for ddpg_agent, obs in
                          zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def step(self):
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset().vector_observations
        action = np.array(self.act(tensor(self.state)))
        next_states, rewards, terminals = self.task.step(action)
        self.episode_reward_agent1 += rewards[0]
        self.episode_reward_agent2 += rewards[1]
        self.replay.add(self.state, action, rewards, next_states, terminals.astype(np.uint8))
        if np.any(terminals):
            self.episode_rewards.append(max(self.episode_reward_agent1, self.episode_reward_agent2))
            self.episode_reward_agent1 = 0
            self.episode_reward_agent2 = 0
            self.random_process.reset_states()
            self.task.reset()
        self.state = next_states
        self.total_steps += 1

        if len(self.replay) >= hyper_parameter.min_memory_size:
            experiences = self.replay.sample_maddpg()
            states1, states2, actions1, actions2, rewards1, rewards2, \
            next_states1, next_states2, dones1, dones2 = experiences

            next_action_agent1 = self.maddpg_agent[0].target_act(tensor(next_states1))
            next_action_agent2 = self.maddpg_agent[1].target_act(tensor(next_states2))

            np_next_action_agent1 = to_np(next_action_agent1)
            np_next_action_agent2 = to_np(next_action_agent2)
            next_critic_input = np.hstack((next_states1, next_states2, np_next_action_agent1, np_next_action_agent2))
            with torch.no_grad():
                q_next_agent1 = self.maddpg_agent[0].target_critic_val(tensor(next_critic_input))
                q_next_agent2 = self.maddpg_agent[1].target_critic_val(tensor(next_critic_input))
                q_y_agent1 = hyper_parameter.GAMMA*q_next_agent1 + tensor(rewards1)
                q_y_agent2 = hyper_parameter.GAMMA*q_next_agent2 + tensor(rewards2)

            critic_input = np.hstack((states1, states2, actions1, actions2))
            q_current_agent1 = self.maddpg_agent[0].critic_val(tensor(critic_input))
            q_current_agent2 = self.maddpg_agent[1].critic_val(tensor(critic_input))

            # agent critic loss
            critic_loss1 = torch.nn.MSELoss(q_y_agent1, q_current_agent1)
            critic_loss2 = torch.nn.MSELoss(q_y_agent2, q_current_agent2)

            # update agent1 critic network
            self.maddpg_agent[0].critic_optimizer.zero_grad()
            critic_loss1.backwards()
            self.maddpg_agent[0].critic_optimizer.step()

            # update agent2 critic network
            self.maddpg_agent[1].critic_optimizer.zero_grad()
            critic_loss2.backwards()
            self.maddpg_agent[1].critic_optimizer.step()







    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(tensor, samples)

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - hyper_parameter.target_network_mix) +
                               param * hyper_parameter.target_network_mix)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            self.soft_update(ddpg_agent.target_actor, ddpg_agent.actor)
            self.soft_update(ddpg_agent.target_critic, ddpg_agent.critic)
            
            
            




