from wall_env import PointEnv
from tf_agents.environments import wrappers
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

import numpy as np
import gym


class GoalConditionedPointWrapper(gym.Wrapper):
  """Wrapper that appends goal to state produced by environment."""
  
  def __init__(self, env, prob_constraint=0.8, min_dist=0, max_dist=4,
               threshold_distance=1.0):
    """Initialize the environment.

    Args:
      env: an environment.
      prob_constraint: (float) Probability that the distance constraint is
        followed after resetting.
      min_dist: (float) When the constraint is enforced, ensure the goal is at
        least this far from the initial state.
      max_dist: (float) When the constraint is enforced, ensure the goal is at
        most this far from the initial state.
      threshold_distance: (float) States are considered equivalent if they are
        at most this far away from one another.
    """
    self._threshold_distance = threshold_distance # tolerant distance from goal
    self._prob_constraint = prob_constraint
    self._min_dist = min_dist
    self._max_dist = max_dist
    super(GoalConditionedPointWrapper, self).__init__(env)
    self.observation_space = gym.spaces.Dict({
        'observation': env.observation_space,
        'goal': env.observation_space,
    })
  
  def _normalize_obs(self, obs): # TODO: ?? Why need this Normalization
    """Normalize observations."""
    return np.array([
        obs[0] / float(self.env._height),
        obs[1] / float(self.env._width)
    ])

  def reset(self):
    """env.reset and reset a random goal."""
    goal = None
    count = 0
    while goal is None:
      obs = self.env.reset()
      (obs, goal) = self._sample_goal(obs)
      count += 1
      if count > 1000:
        print('WARNING: Unable to find goal within constraints.')
    self._goal = goal
    return {'observation': self._normalize_obs(obs),
            'goal': self._normalize_obs(self._goal)}

  def step(self, action):
    obs, _, _, _ = self.env.step(action)
    rew = -1.0
    done = self._is_done(obs, self._goal)
    return {'observation': self._normalize_obs(obs),
            'goal': self._normalize_obs(self._goal)}, rew, done, {}

  def set_sample_goal_args(self, prob_constraint=None,
                           min_dist=None, max_dist=None):
    assert prob_constraint is not None
    assert min_dist is not None
    assert max_dist is not None
    assert min_dist >= 0
    assert max_dist >= min_dist
    self._prob_constraint = prob_constraint
    self._min_dist = min_dist
    self._max_dist = max_dist

  def _is_done(self, obs, goal):
    """Determines whether observation equals goal."""
    return np.linalg.norm(obs - goal) < self._threshold_distance

  def _sample_goal(self, obs):
    """Sampled a goal state."""
    if np.random.random() < self._prob_constraint:
      return self._sample_goal_constrained(obs, self._min_dist, self._max_dist)
    else:
      return self._sample_goal_unconstrained(obs)

  def _sample_goal_constrained(self, obs, min_dist, max_dist):
    """Samples a goal with dist min_dist <= d(obs, goal) <= max_dist.

    Args:
      obs: observation (without goal).
      min_dist: (int) minimum distance to goal.
      max_dist: (int) maximum distance to goal.
    Returns:
      obs: observation (without goal).
      goal: a goal state.
    """
    (i, j) = self.env._discretize_state(obs)
    mask = np.logical_and(self.env._apsp[i, j] >= min_dist,
                          self.env._apsp[i, j] <= max_dist)
    mask = np.logical_and(mask, self.env._walls == 0)
    candidate_states = np.where(mask)
    num_candidate_states = len(candidate_states[0])
    if num_candidate_states == 0:
      return (obs, None)
    goal_index = np.random.choice(num_candidate_states)
    goal = np.array([candidate_states[0][goal_index],
                     candidate_states[1][goal_index]],
                    dtype=np.float)
    goal += np.random.uniform(size=2)
    dist_to_goal = self.env._get_distance(obs, goal)
    assert min_dist <= dist_to_goal <= max_dist
    assert not self.env._is_blocked(goal)
    return (obs, goal)
    
  def _sample_goal_unconstrained(self, obs):
    """Samples a goal without any constraints.

    Args:
      obs: observation (without goal).
    Returns:
      obs: observation (without goal).
      goal: a goal state.
    """
    return (obs, self.env._sample_empty_state())
    
  @property
  def max_goal_dist(self):
    apsp = self.env._apsp
    return np.max(apsp[np.isfinite(apsp)])
    

class NonTerminatingTimeLimit(wrappers.PyEnvironmentBaseWrapper):
  """Resets the environment without setting done = True.

  Resets the environment if either these conditions holds:
    1. The base environment returns done = True
    2. The time limit is exceeded.
  """

  def __init__(self, env, duration):
    super(NonTerminatingTimeLimit, self).__init__(env)
    self._duration = duration
    self._step_count = None

  def _reset(self):
    self._step_count = 0
    return self._env.reset()

  @property
  def duration(self):
    return self._duration

  def _step(self, action):
    if self._step_count is None:
      return self.reset()

    ts = self._env.step(action)

    self._step_count += 1
    if self._step_count >= self._duration or ts.is_last():
      self._step_count = None

    return ts
  
def env_load_fn(environment_name,
         max_episode_steps=None,
         resize_factor=1,
         gym_env_wrappers=(GoalConditionedPointWrapper,),
         terminate_on_timeout=False):
  """Loads the selected environment and wraps it with the specified wrappers.

  Args:
    environment_name: Name for the environment to load.
    max_episode_steps: If None the max_episode_steps will be set to the default
      step limit defined in the environment's spec. No limit is applied if set
      to 0 or if there is no timestep_limit set in the environment's spec.
    gym_env_wrappers: Iterable with references to wrapper classes to use
      directly on the gym environment.
    terminate_on_timeout: Whether to set done = True when the max episode
      steps is reached.

  Returns:
    A PyEnvironmentBase instance.
  """
  gym_env = PointEnv(walls=environment_name,
                     resize_factor=resize_factor)
  
  for wrapper in gym_env_wrappers:
    gym_env = wrapper(gym_env)
  
  env = gym_wrapper.GymWrapper( # tf_agents.environments
      gym_env,
      discount=1.0,
      auto_reset=True,
  )

  if max_episode_steps > 0:
    if terminate_on_timeout:
      env = wrappers.TimeLimit(env, max_episode_steps)
    else:
      env = NonTerminatingTimeLimit(env, max_episode_steps)

  return tf_py_environment.TFPyEnvironment(env)
