# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()


_C.HYPER_PARAMETER = CN()
# # The EPSILON Value
# _C.HYPER_PARAMETER.EPSILON = 0.2
# # The beta parameter used to balance rewards and entropy
# _C.HYPER_PARAMETER.BETA = 0.01
# Discount Factor
_C.HYPER_PARAMETER.GAMMA = 0.95
# The Agent Number
_C.HYPER_PARAMETER.AGENTS_NUM = 1
# Action Space Size
_C.HYPER_PARAMETER.ACTION_SPACE = 2
# Environment State Space Size
_C.HYPER_PARAMETER.STATE_SPACE = 24
# Model Save Interval
_C.HYPER_PARAMETER.SAVE_INTERVAL = 1000
# Max Iteration Stpes
_C.HYPER_PARAMETER.MAX_STEPS = int(16000)
# Model log interval
_C.HYPER_PARAMETER.LOG_INTERVAL = 100
# Batch Size
_C.HYPER_PARAMETER.BATCHSIZE = 512
# Minimal Memory Size
_C.HYPER_PARAMETER.min_memory_size = 4096
# Target NetWork Mix
_C.HYPER_PARAMETER.target_network_mix = 1e-3
# Replay Buffer Size
_C.HYPER_PARAMETER.REPLAY_BUFFER_SIZE = 5e5
# update actor and critic frequencies
_C.HYPER_PARAMETER.UPDATE_FREQ = 4

_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 400
_C.MODEL_PARAMETER.H2 = 300


def get_maddpg_cfg_defaults():
    return _C.clone()