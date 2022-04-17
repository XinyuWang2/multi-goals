import gym
import numpy as np
from walls_cfg import WALLS, resize_walls

# Fixed start and goal position
# FourRooms: resize_factor=1. start = (9,2), goal, (2, 9)
# Small: resize_factor=1. start = (0,0), goal, (3, 3)
# Spiral9x9: resize_factor=1. start = (0,0), goal, (4, 4)

s1, s2 = 0, 0
g1, g2 = 4, 4


class PointEnv(gym.Env):
	"""Abstract class for 2D navigation environments."""

	def __init__(self, walls=None, resize_factor=1,
							 action_noise=0.0):
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
		(height, width) = self._walls.shape
		self._height = height
		self._width = width
		self._action_noise = action_noise # TODO: defult 0, can add noise later
		self.action_space = gym.spaces.Discrete(4) # action is discrete!
		self.observation_space = gym.spaces.Discrete(self._height * self._width) # observation is discrete!
		self._state = None # int
		self._goal = None # int, goal is set in self.reset()
		self.viewer = None
		self.reset()


	def _sample_empty_state(self):
		candidate_states = np.where(self._walls == 0)
		num_candidate_states = len(candidate_states[0])
		state_index = np.random.choice(num_candidate_states)
		state_2d = np.array([candidate_states[0][state_index],
											candidate_states[1][state_index]])
		state = state_2d[0] * self._width + state_2d[1]
		return state
			

	def reset(self):
		self._state = s1 * self._width + s2  #self._sample_empty_state()
		self._goal = g1 * self._width + g2 #self._sample_empty_state()
		
		# obs = {
		# 		'achieved_goal': self._state,
		# 		'desired_goal': self._goal,
		# }
		obs = self._state

		assert self.observation_space.contains(obs)
		return obs

	def step(self, action):
		"""Split action into num_substeps. Return the state before confronting obstacles."""
		assert self.action_space.contains(action)

		# get next state
		x, y = self._state % self._width, self._state // self._width
		next_x, next_y = x, y
		if action == 0: next_x -= 1   # left
		elif action == 1: next_x += 1   # right
		elif action == 2: next_y -= 1   # up
		elif action == 3: next_y += 1   # down

		# wall effect, obstacles or boundary
		if self._is_collide(next_x, next_y):
			next_x, next_y = x, y

		# check goal
		self._state = next_y * self._width + next_x
		done = (self._state == self._goal)
		reward = -1
		if done: 
			reward = 100 # Reach goal get reward 100

		# obs = {
		# 		'achieved_goal': self._state,
		# 		'desired_goal': self._goal,
		# }
		obs = self._state
		
		info = {'is_success': int(done)}

		return obs, reward, done, info


	def _is_collide(self, x, y):
		hit_boundary, hit_obstacle = False, False
		hit_boundary = x < 0 or x >= self._width or y < 0 or y >= self._height
		if not hit_boundary:
			hit_obstacle = self._walls[y, x] == 1
		return hit_obstacle or hit_boundary


	def _is_terminal(self, state):
		return state == self._goal


	def _state_to_xy(self, state):
		assert(isinstance(state, int) )
		x, y = state % self._width, state // self._width
		return x, y

	def _xy_to_state(self, x, y):
		return y * self._width + x

	def render(self, mode='human', close=False):
		u_size = 40 # cell size

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			m = 2 # gaps between two cells

			self.viewer = rendering.Viewer(self._width * u_size, self._height * u_size)

			for y in range(self._height):
				for x in range(self._width):
					# four vertices of a square block (ul,ur,lr,ll)
					v = [(x * u_size + m, y * u_size + m),
						((x + 1 ) * u_size - m, y * u_size + m),
						((x + 1) * u_size - m, (y + 1) * u_size - m),
							(x * u_size + m, (y + 1 ) * u_size - m)]

					# build geom obj based on its vertices 
					rect = rendering.FilledPolygon(v)
					self.viewer.add_geom(rect)

					# set obstacle, goal, rest to be different colors
					state = self._xy_to_state(x, y)
					if self._is_terminal(state):
						rect.set_color(0.0, 1, 1)
					elif self.walls[y,x] == 1: # obstacle cells are with gray color
						rect.set_color(0.2,0.2,0.2)
					else:
						rect.set_color(0.8,0.8,0.8)

			# draw agent
			self.agent = rendering.make_circle(u_size/4, 30, True)
			self.agent.set_color(1.0, 1.0, 0.0)
			self.viewer.add_geom(self.agent)
			self.agent_trans = rendering.Transform()
			self.agent.add_attr(self.agent_trans)

		# update position of an agent
		x, y = self._state_to_xy(self._state)

		self.agent_trans.set_translation((x + 0.5) * u_size, ( y + 0.5) * u_size)        
		return self.viewer.render(return_rgb_array = mode=='rgb_array')


	@property
	def walls(self):
		return self._walls
	
	@property
	def goal_copy(self):
		return self._goal.copy()


if __name__ == '__main__':
	from stable_baselines3.common.env_checker import check_env
	env = PointEnv("FourRooms")
	check_env(env, warn=True)