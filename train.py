from components import get_ppo_ac_cfg_defaults, get_episodes_count, get_ac_cfg_defaults
import time
import numpy as np
import agent
from matplotlib import pyplot as plt


env_path = './Reacher_Env/multiple_agents/Reacher_Linux/Reacher.x86_64'
# The agent type  ['A2CAgent', 'PPOACAgent']
AGENT_NAME = 'A2CAgent'
if AGENT_NAME == 'PPOACAgent':
    hyper_parameter = get_ppo_ac_cfg_defaults().HYPER_PARAMETER.clone()
elif AGENT_NAME == 'A2CAgent':
    hyper_parameter = get_ac_cfg_defaults().HYPER_PARAMETER.clone()
else:
    raise ValueError('This agent does not support at present!')


if __name__ == '__main__':
    agent = getattr(agent, AGENT_NAME)(env_path)
    total_mean_rewards = []
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if hyper_parameter.SAVE_INTERVAL and not agent.total_steps % hyper_parameter.SAVE_INTERVAL \
                and len(agent.episode_rewards) and max(agent.episode_rewards) > 30:
            agent.save('saved_models/model-%s-%s.pth' % (agent_name, str(len(total_mean_rewards)+1)))
        if hyper_parameter.LOG_INTERVAL and not agent.total_steps % hyper_parameter.LOG_INTERVAL and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            mean_rewards = np.mean(rewards)
            agent.episode_rewards = []
            total_mean_rewards.append(mean_rewards)
            print('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                len(total_mean_rewards), mean_rewards, np.median(rewards), np.min(rewards), np.max(rewards),
                hyper_parameter.LOG_INTERVAL / (time.time() - t0)))
            t0 = time.time()
        if hyper_parameter.MAX_STEPS and agent.total_steps >= hyper_parameter.MAX_STEPS:
            agent.close()
            break
        if get_episodes_count(total_mean_rewards, 30) > 100:
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
