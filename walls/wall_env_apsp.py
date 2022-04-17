import collections
import random

import gym
import numpy as np
import networkx as nx
from walls_cfg import WALLS, resize_walls


class PointEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self, walls=None, resize_factor=1,
               action_noise=1.0):
    """Initialize the point environment.

    Args:
      walls: (str) name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
      action_noise: (float) Standard deviation of noise to add to actions. Use 0
        to add no noise.
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    self._apsp = self._compute_apsp(self._walls)
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = action_noise
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.array([0.0, 0.0]),
        high=np.array([self._height, self._width]),
        dtype=np.float32)
    self.reset()

  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=np.float)
    state += np.random.uniform(size=2) # Note state in in continuous space
    assert not self._is_blocked(state)
    return state
      
  def reset(self):
    self.state = self._sample_empty_state()
    return self.state.copy()
  
  def _get_distance(self, obs, goal):
    """Compute the shortest path distance.
    Note: This distance is *not* used for training."""
    (i1, j1) = self._discretize_state(obs)
    (i2, j2) = self._discretize_state(goal)
    return self._apsp[i1, j1, i2, j2]

  def _discretize_state(self, state, resolution=1.0):
    (i, j) = np.floor(resolution * state).astype(np.int)
    # Round down to the nearest cell if at the boundary.
    if i == self._height:
      i -= 1
    if j == self._width:
      j -= 1
    return (i, j)
  
  def _is_blocked(self, state):
    """Check if (nearest discretized state) is an obstacle."""
    # state may not lie in any discretized states, due to random.uniform
    # if not self.observation_space.contains(state): # observation_space.contains not implemented 
    #   return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def step(self, action):
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)
    num_substeps = 10 # TODO: ?? Why do many substeps
    dt = 1.0 / num_substeps
    num_axis = len(action) # note actions are in many dimensions
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state

    done = False
    rew = -1.0 * np.linalg.norm(self.state) # TODO: ?? Why this reward 
    return self.state.copy(), rew, done, {}

  @property
  def walls(self):
    return self._walls

  def _compute_apsp(self, walls):
    """Compute All Pairs Shortest Path (APSP)."""
    (height, width) = walls.shape
    g = nx.Graph()
    # Add all the nodes
    for i in range(height):
      for j in range(width):
        if walls[i, j] == 0:
          g.add_node((i, j))

    # Add all the edges
    for i in range(height):
      for j in range(width):
        for di in [-1, 0, 1]:
          for dj in [-1, 0, 1]:
            if di == dj == 0: continue  # Don't add self loops
            if i + di < 0 or i + di > height - 1: continue  # No cell here
            if j + dj < 0 or j + dj > width - 1: continue  # No cell here
            if walls[i, j] == 1: continue  # Don't add edges to walls
            if walls[i + di, j + dj] == 1: continue  # Don't add edges to walls
            g.add_edge((i, j), (i + di, j + dj))

    # dist[i, j, k, l] is path from (i, j) -> (k, l)
    dist = np.full((height, width, height, width), np.float('inf'))
    for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
      for ((i2, j2), d) in dist_dict.items():
        dist[i1, j1, i2, j2] = d
    return dist