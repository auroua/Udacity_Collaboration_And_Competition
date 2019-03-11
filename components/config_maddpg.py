# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()


_C.HYPER_PARAMETER = CN()
# The EPSILON Value
_C.HYPER_PARAMETER.EPSILON = 0.2
# The beta parameter used to balance rewards and entropy
_C.HYPER_PARAMETER.BETA = 0.01
# Discount Factor
_C.HYPER_PARAMETER.GAMMA = 0.95
# The Agent Number
_C.HYPER_PARAMETER.AGENTS_NUM = 1
# Action Space Size
_C.HYPER_PARAMETER.ACTION_SPACE = 2
# Environment State Space Size
_C.HYPER_PARAMETER.STATE_SPACE = 24
# MAX Single Episodes length
_C.HYPER_PARAMETER.TMAX = 10000
# Model Save Interval
_C.HYPER_PARAMETER.SAVE_INTERVAL = 2000
# Max Iteration Stpes
_C.HYPER_PARAMETER.MAX_STEPS = int(2e7)
# Model log interval
_C.HYPER_PARAMETER.LOG_INTERVAL = 200
# Suggort clip value
_C.HYPER_PARAMETER.CLIP_VAL = 0.2
# Batch Size
_C.HYPER_PARAMETER.BATCHSIZE = 256
# entropy weight
_C.HYPER_PARAMETER.entropy_weight = 0.01
# Minimal Memory Size
_C.HYPER_PARAMETER.min_memory_size = 1024
# Target NetWork Mix
_C.HYPER_PARAMETER.target_network_mix = 1e-3
# Replay Buffer Size
_C.HYPER_PARAMETER.REPLAY_BUFFER_SIZE = 5e5

_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 400
_C.MODEL_PARAMETER.H2 = 300


_C.TRAIN_PARAMETER = CN()
# Training episodes
_C.TRAIN_PARAMETER.EPISODES = 1000
# Learning Rate
_C.TRAIN_PARAMETER.LR =3e-4
# MOMENTUM
_C.TRAIN_PARAMETER.MOMENTUM = 0.9
# The Optimizer used for training (SGD, Adam, ADABOUND)
_C.TRAIN_PARAMETER.OPTIMIZER = 'Adam'
# Grad clip
_C.TRAIN_PARAMETER.Gradient_Clip = 0.5
# entropy weight
_C.TRAIN_PARAMETER.entropy_weight = 0.01
# value loss weight
_C.TRAIN_PARAMETER.value_loss_weight = 1.0


def get_maddpg_cfg_defaults():
    return _C.clone()
