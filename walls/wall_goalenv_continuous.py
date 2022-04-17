import gym
import numpy as np
from collections import OrderedDict
from walls_cfg import WALLS, resize_walls

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class PointEnv(gym.GoalEnv):
	"""Abstract class for 2D navigation environments."""

	def __init__(self, 
				 walls=None, 
				 resize_factor=1,
				 action_noise=0.0,
				 distance_threshold=0.05,
				 reward_type = "sparse"):
		"""Initialize the point environment.
		Args:
			walls: (str) name of one of the maps defined above.
			resize_factor: (int) Scale the map by this factor.
			action_noise: (float) Standard deviation of noise to add to actions. Use 0
				to add no noise.
			distance_threshold: threashold for checking if goal is reached
			reward_type: "sparse": reward is -1 if goal not reached
						 "dense": reward depends on the distance to the goal
		"""
		if resize_factor > 1:
			self._walls = resize_walls(WALLS[walls], resize_factor)
		else:
			self._walls = WALLS[walls]
		(height, width) = self._walls.shape

		self._height = height
		self._width = width
		self._action_noise = action_noise # TODO: defult 0, can add noise later
		self.distance_threshold = distance_threshold
		self.reward_type = reward_type

		self.action_space = gym.spaces.Box(
			low=np.array([-1.0, -1.0]),
			high=np.array([1.0, 1.0]),
			dtype=np.float32) # action is continuous!

		self.observation_space = gym.spaces.Dict(
			{
				key: gym.spaces.Box(
					low=np.array([0.0, 0.0]),
					high=np.array([self._height, self._width]),
					dtype=np.float32)
				for key in ['observation', 'achieved_goal', 'desired_goal']
			}
		) # observation is discrete!

		self._state = None # np.ndarray, init in self.reset()
		self._goal = None # np.ndarray, init in self.reset()
		self.viewer = None
		self.reset()


	def _sample_empty_state(self):
		candidate_states = np.where(self._walls == 0)
		num_candidate_states = len(candidate_states[0])
		state_index = np.random.choice(num_candidate_states)
		# empty cell index
		state = np.array([candidate_states[0][state_index],
							 candidate_states[1][state_index]],
							 dtype=np.float32)
		# a random point in the sampled cell
		state += np.random.uniform(size=2) # Note state is in continuous space
		return state
			

	def reset(self):
		self._state = self._sample_empty_state()
		self._goal = self._sample_empty_state()
		obs = self._get_obs()
		return obs


	def compute_reward(self, achieved_goal, goal, info):
		# Format follows compute_reward from openai gym
		d = goal_distance(achieved_goal, goal)
		if self.reward_type == "sparse":
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return -d


	def step(self, action):
		"""Split action into num_substeps. Return the state before confronting obstacles."""
		assert self.action_space.contains(action)

		# get next state
		# Method 1: Don't move at all if take the action will hit obstacles

		# Method 2: Take the action in multiple steps until it hits obstacles
		num_substeps = 10 # hyperparameter to break the action into steps
		step = action / num_substeps
		new_state = self._state.copy()
		for _ in range(num_substeps):
			new_state += step
			if self._is_collide(new_state):
				new_state -= step
				break
		self._state = new_state

		done = goal_distance(self._state, self._goal) < self.distance_threshold
		obs = self._get_obs()
		info = {'is_success': int(done)}
		reward = self.compute_reward(obs["achieved_goal"], self._goal, info)

		return obs, reward, done, info


	def _is_collide(self, state):
		x, y = state # dimension is assumed 2
		hit_boundary, hit_obstacle = False, False
		hit_boundary = x < 0 or x >= self._width or y < 0 or y >= self._height
		if not hit_boundary:
			cell_y, cell_x = int(np.floor(y)), int(np.floor(x))
			hit_obstacle = self._walls[cell_y, cell_x] == 1
		return hit_obstacle or hit_boundary


	# def _is_terminal(self, state):
	# 	return state == self._goal


	def _state_to_xy(self, state):
		x, y = state % self._width, state // self._width
		return x, y

	def _xy_to_state(self, x, y):
		return y * self._width + x


	def _get_obs(self):
		obs = OrderedDict({
				'achieved_goal': self._state, # use .copy when it is numpy.array
				'desired_goal': self._goal,
				'observation': self._state
		})
		return obs


	def render(self, mode='human', close=False):
		u_size = 40 # cell size

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			m = 2 # gaps between two cells

			self.viewer = rendering.Viewer(self._width * u_size, self._height * u_size)

			for y in range(self._height):
				for x in range(self._width):
					# four vertices of a square block
					v = [(x * u_size + m, y * u_size + m),
						((x + 1 ) * u_size - m, y * u_size + m),
						((x + 1) * u_size - m, (y + 1) * u_size - m),
							(x * u_size + m, (y + 1 ) * u_size - m)]

					# build geom obj based on its vertices 
					rect = rendering.FilledPolygon(v)
					self.viewer.add_geom(rect)

					# set obstacle, goal, rest to be different colors
					if self.walls[y,x] == 1: # obstacle cells are with gray color
						rect.set_color(0.2,0.2,0.2)
					else:
						rect.set_color(0.8,0.8,0.8)

			# draw agent
			self.agent = rendering.make_circle(u_size/5, 30, True)
			self.agent.set_color(1.0, 1.0, 0.0)
			self.viewer.add_geom(self.agent)
			self.agent_trans = rendering.Transform()
			self.agent.add_attr(self.agent_trans)

			# draw goal region
			self.goal_region = rendering.make_circle(u_size/5, 30, True)
			self.goal_region.set_color(0.0, 1, 1)
			self.viewer.add_geom(self.goal_region)
			self.goal_region_trans = rendering.Transform()
			self.goal_region.add_attr(self.goal_region_trans)


		# update position of an agent
		sx, sy = self._state
		self.agent_trans.set_translation(sx * u_size, sy * u_size) 

		# update position of a goal
		gx, gy = self._goal
		self.goal_region_trans.set_translation(gx * u_size, gy * u_size) 

		return self.viewer.render(return_rgb_array = mode=='rgb_array')


	@property
	def walls(self):
		return self._walls
	

if __name__ == '__main__':
	from stable_baselines3.common.env_checker import check_env
	env = PointEnv("FourRooms")
	check_env(env, warn=True)
