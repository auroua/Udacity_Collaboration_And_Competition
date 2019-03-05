import numpy as np
from pathlib import Path
from unityagents import UnityEnvironment
from .config_ppo_ac import get_ppo_ac_cfg_defaults
from abc import ABC, abstractmethod
# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py


hyper_parameter = get_ppo_ac_cfg_defaults().clone().HYPER_PARAMETER


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_env(env_path, no_graphics=True):
    env = UnityEnvironment(file_name=env_path, no_graphics=no_graphics)
    # get the default brain
    brain_name = env.brain_names[0]
    return env, brain_name


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
                        num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, envs, brain_name):
        self.envs = envs
        VecEnv.__init__(self, 20, None, None)
        self.actions = None
        self.brain_name = brain_name

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        env_info = self.envs.step(self.actions)[self.brain_name]
        rewards = np.array(env_info.rewards)
        next_state = env_info.vector_observations
        dones = np.array(env_info.local_done)
        return next_state, np.array(rewards), np.array(dones)    # next_state shape [20, 33],   rewards shape []

    def reset(self):
        # reset the environment
        return self.envs.reset(train_mode=True)[self.brain_name]

    def close(self):
        self.envs.close()


class Task:
    def __init__(self,
                 name,
                 num_agents,
                 env_path,
                 log_dir=None):
        if log_dir is not None:
            mkdir(log_dir)
        envs, brainname = make_env(env_path)
        # if single_process:
        #     Wrapper = DummyVecEnv
        # else:
        #     Wrapper = SubprocVecEnv
        self.env = DummyVecEnv(envs, brainname)
        self.name = name
        self.num_agents = num_agents
        self.state_dim = hyper_parameter.STATE_SPACE
        self.action_dim = hyper_parameter.ACTION_SPACE

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        actions = np.clip(actions, -1.0, 1.0)
        return self.env.step(actions)

    def close(self):
        self.env.close()


class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)