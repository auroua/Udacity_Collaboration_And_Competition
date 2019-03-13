from components import get_maddpg_cfg_defaults, tensor
import torch
from network import MADDPGPolicy
from unityagents import UnityEnvironment
import numpy as np

AGENT_NAME = 'MADDPGAgent'
env_path = './Unity_Env/Tennis_Linux/Tennis.x86_64'
saved_model_path = './model-MADDPGAgent.pth'
hyper_parameter = get_maddpg_cfg_defaults().HYPER_PARAMETER.clone()
model_parameter = get_maddpg_cfg_defaults().MODEL_PARAMETER.clone()
ckpt = torch.load(saved_model_path)

if __name__ == '__main__':
    agent1 = MADDPGPolicy(24, model_parameter.H1, model_parameter.H2, 2, 52, model_parameter.H1,
                          model_parameter.H2, seed=0)
    agent1.actor.load_state_dict(ckpt['agent1_actor'])
    agent1.critic.load_state_dict(ckpt['agent1_critic'])
    agent2 = MADDPGPolicy(24, model_parameter.H1, model_parameter.H2, 2, 52, model_parameter.H1,
                          model_parameter.H2, seed=1)
    agent2.actor.load_state_dict(ckpt['agent2_actor'])
    agent2.critic.load_state_dict(ckpt['agent2_critic'])

    env = UnityEnvironment(file_name='./Unity_Env/Tennis_Linux/Tennis.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    for i in range(10):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            agent1_act = agent1.act(tensor(states[0::2, :]), noise_weight=False)
            agent2_act = agent2.act(tensor(states[1::2, :]), noise_weight=False)
            actions = np.vstack((agent1_act, agent2_act))
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):
                break
        print('Tennis Round {}, Total score (averaged over agents) this episode: {}'.format(i, scores))

    env.close()
