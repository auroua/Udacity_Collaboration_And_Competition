# This repo is for Udacity Deep Reinforcement Learning course Collaboration and Competition Project

#### Steps
1. install python
2. ```pip install pytorch``` reference [pytorch website](https://pytorch.org/)
3. ```pip install unityagents```
4. download **reacher env** from [Reacher_Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
5. Run ```components/env_tst.py``` to test the environment work all right.
6. Run ```train_maddpg.py``` to train the agent. This module is for the vector state space.
7. Run ```test_maddpg.py``` to test the agent interact with Env.
8. The default config file is  ```components/config_maddpg.py```. You can modify the default parameter value to retrain the agent.


#### Code Environments
* XUbuntu 18.04
* CUDA 10.0
* cudnn 7.4.1
* Python 3.6
* Pytorch 1.0
* yacs v0.1.5


#### Reacher Env
* num agents: 2
* action space: 2 continuous action.
* state space: 24 states
* [version] The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents)

#### TO-DO-LIST
* ~~MADDPG.~~

#### Project Architecture
* Package agent contains the MADDPG agent.
* Package components contains the config files for agent, envs and util functions.
* Package network contains the agent policy network.


#### References
1. [Udacity Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning)
2. [DeepRL](https://github.com/ShangtongZhang/DeepRL)
3. [Open AI MADDPG](https://github.com/openai/maddpg)
4. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
5. [Unity: A General Platform for Intelligent Agents](https://arxiv.org/pdf/1809.02627.pdf)