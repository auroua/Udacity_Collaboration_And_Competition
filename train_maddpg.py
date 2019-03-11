from components import  get_episodes_count, get_maddpg_cfg_defaults
import time
import numpy as np
import agent
from matplotlib import pyplot as plt

AGENT_NAME = 'MADDPGAgent'
env_path = './Unity_Env/Tennis_Linux/Tennis.x86_64'
hyper_parameter = get_maddpg_cfg_defaults().HYPER_PARAMETER.clone()


if __name__ == '__main__':
    agent = getattr(agent, AGENT_NAME)(env_path)
    agent1_reward = []
    agent2_reward = []
    total_mean_rewards = []
    agent_name = agent.__class__.__name__
    t0 = time.time()

    while True:
        if hyper_parameter.SAVE_INTERVAL and not agent.total_steps % hyper_parameter.SAVE_INTERVAL \
                and len(agent.episode_rewards) and max(agent.episode_rewards) > 0.5:
            agent.save('saved_models/model-%s-%s.pth' % (agent_name, str(len(total_mean_rewards)+1)))
        if hyper_parameter.LOG_INTERVAL and not agent.total_steps % hyper_parameter.LOG_INTERVAL and \
                len(agent.episode_rewards):
            rewards = agent.episode_rewards
            none_zero = np.count_nonzero(rewards)
            # mean_rewards = np.mean(rewards)
            # total_mean_rewards.append(mean_rewards)
            print('total steps %d, returns %.2f/%.2f/%.2f (num/nonzero/max), %.2f steps/s' % (
                agent.total_steps, len(rewards), none_zero, np.max(rewards),
                hyper_parameter.LOG_INTERVAL / (time.time() - t0)))
            t0 = time.time()
        if hyper_parameter.MAX_STEPS and agent.total_steps >= hyper_parameter.MAX_STEPS:
            agent.close()
            break
        if get_episodes_count(total_mean_rewards, 0.5) > 100:
            agent.close()
            agent.save('saved_models/model-%s-finish.pth' % agent_name)
            print('Reacher Environment solved!')
            break
        agent.step()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_mean_rewards)), total_mean_rewards)
    plt.ylabel('Mult-Agents Mean Score')
    plt.xlabel('Finished Episode #')
    plt.show()
