import numpy as np
import time

from BC.MLP_policy import MLPPolicy
from BC.replay_buffer import ReplayBuffer
from BC.infrastructure.utils import *

class BCAgent:
    def __init__(self, env, agent_params):
        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicy(self.agent_params['ac_dim'],
                               self.agent_params['ob_dim'],
                               self.agent_params['n_layers'],
                               self.agent_params['hidden_size'],
                               self.agent_params['device'],
                            #    discrete = self.agent_params['discrete'],
                               lr = self.agent_params['learning_rate'],
                               ) 

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])
        self.replay_size = self.agent_params['replay_size']

    def train(self, new_obs_batch, new_action_batch, random=True):
        if len(self.replay_buffer) < self.replay_size:
            return 0

        else:
            old_obs_batch, old_action_batch = self.sample(self.replay_size, random)
            obs_batch = np.concatenate((old_obs_batch, new_obs_batch))
            action_batch = np.concatenate((old_action_batch, new_action_batch))

            loss  = self.actor.update(obs_batch, action_batch) 
            return loss
            
    def add_to_replay_buffer(self, states, actions):
        self.replay_buffer.add(states, actions)

    def sample(self, batch_size, random=True):
        if random:
            return self.replay_buffer.sample_random_data(batch_size)
        else:
            return self.replay_buffer.sample_recent_data(batch_size)
