from collections import namedtuple, deque
import random
import torch
import numpy as np
from components import device


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory.
           state: 2*24
           action: 2*2
           reward: 2
           done: 2
           critic input: agent i's state 24 + agent1 action 2 + agent2 action 2
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
                                       self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
                                 self.device)
        return states, actions, rewards, next_states, dones

    def sample_maddpg(self):
        """Randomly sample a batch of experiences from memory.
           state: 2*24
           action: 2*2
           reward: 2
           done: 2
           critic input: agent i's state 24 + agent1 action 2 + agent2 action 2
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states1 = np.vstack([e.state[0] for e in experiences if e is not None])
        states2 = np.vstack([e.state[1] for e in experiences if e is not None])
        actions1 = np.vstack([e.action[0] for e in experiences if e is not None])
        actions2 = np.vstack([e.action[1] for e in experiences if e is not None])
        rewards1 = np.vstack([e.reward[0] for e in experiences if e is not None])
        rewards2 = np.vstack([e.reward[1] for e in experiences if e is not None])
        next_states1 = np.vstack([e.next_state[0] for e in experiences if e is not None])
        next_states2 = np.vstack([e.next_state[1] for e in experiences if e is not None])
        dones1 = np.vstack([e.done[0] for e in experiences if e is not None]).astype(np.uint8)
        dones2 = np.vstack([e.done[1] for e in experiences if e is not None]).astype(np.uint8)
        return states1, states2, actions1, actions2, rewards1, rewards2, next_states1, next_states2, dones1, dones2

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
