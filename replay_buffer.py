import numpy as np


class ReplayBuffer:
    def __init__(self, input_dims):
        self.buf_index = 0
        self.max_memory = 1000000

        self.observations = np.zeros(shape=(self.max_memory, *input_dims))
        self.next_observations = np.zeros(shape=(self.max_memory, *input_dims))
        self.actions = np.zeros(self.max_memory, dtype=np.int32)
        self.rewards = np.zeros(self.max_memory)
        self.terminates = np.zeros(self.max_memory)

    def store_memory(
        self, observation, next_observation, action, reward, terminate
    ):
        """Use to store memory."""
        idx = self.buf_index % self.max_memory
        self.observations[idx] = observation
        self.next_observations[idx] = next_observation
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminates[idx] = 1 - terminate
        self.buf_index += 1

    def sample_buffer(self, sample_size):
        """Use to sample the buffer."""
        sampler = min(self.buf_index, self.max_memory)

        mask = np.random.choice(sampler, sample_size, replace=False)
        observations = self.observations[mask]
        next_observations = self.next_observations[mask]
        actions = self.actions[mask]
        rewards = self.rewards[mask]
        terminates = self.terminates[mask]

        return (observations, next_observations, actions, rewards, terminates)
