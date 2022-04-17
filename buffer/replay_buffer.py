"""
the replay buffer here is from the openai baselines code
Edited based on paper of Hindsight Experience Replay

"""
import numpy as np


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory manachieved_goalement
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'achieved_g': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'desired_g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'rew': np.empty([self.size, self.T, self.env_params['reward']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_achieved_goal, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        idxs = self._get_storachieved_goale_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['achieved_goal'][idxs] = mb_achieved_goal
        self.buffers['desired_goal'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['achieved_goal_next'] = temp_buffers['achieved_goal'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storachieved_goale_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
