from unityagents import UnityEnvironment
import numpy as np
from network import PPOACNet, A2CNet
from components import get_ppo_ac_cfg_defaults, get_ac_cfg_defaults, tensor, to_np
import torch


POLICY_NAME = 'PPOACNet'
if POLICY_NAME == 'PPOACNet':
    hyper_parameter = get_ppo_ac_cfg_defaults().HYPER_PARAMETER.clone()
    policy = PPOACNet()
elif POLICY_NAME == 'A2CNet':
    hyper_parameter = get_ac_cfg_defaults().HYPER_PARAMETER.clone()
    policy = A2CNet()
else:
    raise ValueError('This agent does not support at present!')

# init policy
policy.load_state_dict(torch.load('./model-PPOACAgent-finish.pth'))


if __name__ == '__main__':
    env = UnityEnvironment(file_name='./Reacher_Env/multiple_agents/Reacher_Linux/Reacher.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        pred = policy(tensor(states))
        actions = to_np(pred['a'])
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()
