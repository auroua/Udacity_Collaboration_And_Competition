from components import get_maddpg_cfg_defaults
import time
import numpy as np
import agent
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

AGENT_NAME = 'MADDPGAgent'
env_path = './Unity_Env/Tennis_Linux/Tennis.x86_64'
hyper_parameter = get_maddpg_cfg_defaults().HYPER_PARAMETER.clone()


if __name__ == '__main__':
    writer = SummaryWriter()
    agent = getattr(agent, AGENT_NAME)(env_path, writer)
    t0 = time.time()
    while True:
        if hyper_parameter.MAX_STEPS and agent.total_steps >= hyper_parameter.MAX_STEPS:
            agent.save('/home/aurora/workspaces_2019/Udacity_Collaboration_And_Competition/'
                       'saved_models/maddpg_agent_final.pth')
            agent.close()
            break
        agent.step()
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(agent.avg_results)), agent.avg_results)
    plt.ylabel('100 Episodes Mean Score')
    plt.xlabel('Episodes')
    plt.show()
