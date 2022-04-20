import time
import numpy as np
import gym
import os

from BC.infrastructure.utils import *

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.obs = None # np.ndarray
        self.acs = None 

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add(self, observations, actions):

        if self.obs is None:
            self.obs = np.array(observations[-self.max_size:])
            self.acs = np.array(actions[-self.max_size:])

        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]

    def sample_random_data(self, batch_size):
        assert self.obs.shape[0] == self.acs.shape[0]
        # return batch_size number of random entries from obs and actions
        indices = np.random.permutation(len(self))[:batch_size]
        return self.obs[indices], self.acs[indices]

    def sample_recent_data(self, batch_size=1):
        return self.obs[-batch_size:], self.acs[-batch_size:]
