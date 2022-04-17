import numpy as np
import torch as th
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union


from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

class HerReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        # device: Union[th.device, str] = "cpu",
        replay_buffer: Optional[DictReplayBuffer] = None, # TODO: It is only for offline, online it is None
        max_episode_length: Optional[int] = None,
        # n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
        handle_timeout_termination: bool = True,
    ):
        # self.n_sampled_goal = n_sampled_goal

        # # compute ratio between HER replays and regular replays in percent for online HER sampling
        # self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))

        # for online sampling, it replaces the "classic" replay buffer completely
        if online_sampling:
            her_buffer_size = buffer_size 
        else:
            her_buffer_size = self.max_episode_length

        self.env = env
        self.buffer_size = her_buffer_size        

        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = self.buffer_size // self.max_episode_length
        self.current_idx = 0 # TODO: ??? Feel like idx for time steps in buffer

        # Counter to prevent overflow
        self.episode_steps = 0 # episode counter

        # Get shape of observation and goal (usually the same)
        self.obs_shape = get_obs_shape(self.env.observation_space.spaces["observation"])
        self.goal_shape = get_obs_shape(self.env.observation_space.spaces["achieved_goal"])

        # input dimensions for buffer initialization
        input_shape = {
            "observation": (self.env.num_envs,) + self.obs_shape,
            "achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "desired_goal": (self.env.num_envs,) + self.goal_shape,
            "action": (self.action_dim,),
            "reward": (1,),
            "next_obs": (self.env.num_envs,) + self.obs_shape,
            "next_achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "next_desired_goal": (self.env.num_envs,) + self.goal_shape,
            "done": (1,),
        }

        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        
        # # Store info dicts so it can be used to compute the reward (e.g. continuity cost)
        # self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)] #TODO??

        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    def _sample_offline(
        self,
        n_sampled_goal: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample function for offline sampling of HER transition,
        in that case, only one episode is used and transitions
        are added to the regular replay buffer.
        :param n_sampled_goal: Number of sampled goals for replay
        :return: at most(n_sampled_goal * episode_length) HER transitions.
        """
        # `maybe_vec_env=None` as we should store unnormalized transitions,
        # they will be normalized at sampling time
        return self._sample_transitions(
            batch_size=None,
            maybe_vec_env=None,
            online_sampling=False,
            n_sampled_goal=n_sampled_goal,
        )

    
    # called after finishing storing a complete episode (include many times add() to add transitions)
    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        # self.current_idx here means the idx of last transition in the current last epside in the buffer
        self.episode_lengths[self.pos] = self.current_idx 

        # Last episode is finally added in the buffer, we now update pointer 
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0


    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Add one transition in the buffer. (call store_episode after all transitions from one episode are added)"""
        if self.current_idx == 0 and self.full: 
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length) #TODO??

        # Remove termination signals due to timeout, so done really means reaching the goal
        if self.handle_timeout_termination:
            done_ = done * (1 - np.array([info.get("TimeLimit.truncated", False) for info in infos]))
        else:
            done_ = done

        self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        # self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"] # TODO ??
        # self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        # self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]

        # When doing offline sampling
        # Add real transition to normal replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(
                obs,
                next_obs,
                action,
                reward,
                done,
                infos,
            )

        self.info_buffer[self.pos].append(infos)

        # update current episode timestep idx pointer
        self.current_idx += 1

        # update current episode counter
        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode() # Mark finish add all transitions in this episode  !!!!!!!!!!!!!!
            if not self.online_sampling:
                # sample virtual transitions and store them in replay buffer
                self._sample_her_transitions()
                # clear storage for current episode
                # Whole purpose of offline sampling is to add some her_transitions to its regular replay buffer.
                # After that, current HerReplayBuffer is useless, thus reset()
                self.reset() 

            self.episode_steps = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.
        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        if self.replay_buffer is not None:
            return self.replay_buffer.sample(batch_size, env)
        return self._sample_transitions(batch_size, maybe_vec_env=env, online_sampling=True)  # pytype: disable=bad-return-type
    
    # offline only
    def _sample_her_transitions(self) -> None:
        """
        Sample additional goals and store new transitions in replay buffer
        when using offline sampling.
        """

        # Sample goals to create virtual transitions for the last episode.
        observations, next_observations, actions, rewards = self._sample_offline(n_sampled_goal=self.n_sampled_goal)

        # Store virtual transitions in the replay buffer, if available
        if len(observations) > 0:
            for i in range(len(observations["observation"])):
                self.replay_buffer.add(
                    {key: obs[i] for key, obs in observations.items()},
                    {key: next_obs[i] for key, next_obs in next_observations.items()},
                    actions[i],
                    rewards[i],
                    # We consider the transition as non-terminal
                    done=[False],
                    infos=[{}],
                )

    # offline only
    def _sample_offline(
        self,
        n_sampled_goal: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample function for offline sampling of HER transition,
        in that case, only one episode is used and transitions
        are added to the regular replay buffer.
        :param n_sampled_goal: Number of sampled goals for replay
        :return: at most(n_sampled_goal * episode_length) HER transitions.
        """
        # `maybe_vec_env=None` as we should store unnormalized transitions,
        # they will be normalized at sampling time
        return self._sample_transitions(
            batch_size=None,
            maybe_vec_env=None,
            online_sampling=False,
            n_sampled_goal=n_sampled_goal,
        )


    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
        online_sampling: bool,
        n_sampled_goal: Optional[int] = None,
    ) -> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        if online_sampling:
            # assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # # Do not sample the episode with index `self.pos` as the episode is invalid
            # if self.full:
            #     episode_indices = (
            #         np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
            #     ) % self.n_episodes_stored
            # else:
            #     episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            # # A subset of the transitions will be relabeled using HER algorithm
            # her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
            assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
            # Offline sampling: there is only one episode stored
            episode_length = self.episode_lengths[0]
            # we sample n_sampled_goal per timestep in the episode (only one is stored).
            episode_indices = np.tile(0, (episode_length * n_sampled_goal)) #TODO: Doesn't look like episode indices
            # we only sample virtual transitions
            # as real transitions are already stored in the replay buffer
            her_indices = np.arange(len(episode_indices)) #TODO: ??

        ep_lengths = self.episode_lengths[episode_indices]

        if online_sampling:
            # # Select which transitions to use
            # transitions_indices = np.random.randint(ep_lengths)
        else:
            if her_indices.size == 0:
                # Episode of one timestep, not enough for using the "future" strategy
                # no virtual transitions are created in that case
                return {}, {}, np.zeros(0), np.zeros(0)
            else:
                # Repeat every transition index n_sampled_goals times
                # to sample n_sampled_goal per timestep in the episode (only one is stored).
                # Now with the corrected episode length when using "future" strategy
                transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
                episode_indices = episode_indices[transitions_indices]
                her_indices = np.arange(len(episode_indices))

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            transitions["reward"][her_indices, 0] = self.env.env_method(
                "compute_reward",
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_achieved_goal"][her_indices, 0],
                # here we use the new desired goal
                transitions["desired_goal"][her_indices, 0],
                transitions["info"][her_indices, 0],
            )

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        if online_sampling:
            # next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

            # normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

            # return DictReplayBufferSamples(
            #     observations=normalized_obs,
            #     actions=self.to_torch(transitions["action"]),
            #     next_observations=next_obs,
            #     dones=self.to_torch(transitions["done"]),
            #     rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
            # )
        else:
            return observations, next_observations, transitions["action"], transitions["reward"]


    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)