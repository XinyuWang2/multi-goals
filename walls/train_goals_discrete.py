import gym
from wall_env_goals import PointEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import time

# Usually the cause of failed task is both loss are so small while ep_len still big,
# thus training gets stuck and episode length and reward don't change

# when reward is -1 100
# FourRooms N
# Cross Y
# Spiral5x5 R1-Y R2-Y
# Spiral7x7 R1-N R2-N 

# when reward is 0 1
# Spiral5x5 R1-Y (150k: training gets almost stuck at the beginning, then it jumps out of suboptimal)
# Spiral7x7 R1-N 
# This implies setting reward to be 0 for not reaching terminal state is even harder!


MAX_EPISODE_STEPS = 1000
RESIZE = 1
# ENV_NAME = "Spiral7x7"
ENV_NAME = "Spiral5x5"
# ENV_NAME = "Small"
TIMESTEPS = int(1.5e5)

model_path = "trained_models/" + ENV_NAME + "01_*" + str(RESIZE) + '_' + str(TIMESTEPS)

env = PointEnv(ENV_NAME, resize_factor = 1)
# env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS) # use gym wrapper
obs = env.reset()

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=TIMESTEPS)
model.save(model_path)

# model = PPO.load(model_path)

obs = env.reset()
env.render()

for i in range(10):
	done = False
	obs = env.reset()
	# while not done:
	for i in range(10):
		action, _state = model.predict(obs, deterministic=True)
		print(action)
		time.sleep(0.5)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			print(info)
			time.sleep(0.3)
			break
	
	time.sleep(1)


# obs = env.reset()
# env.render()
# for i in range(5):
# 	done = False
# 	while not done:
# 		action = env.action_space.sample()
# 		obs, reward, done, info = env.step(action)
# 		# print("action", action)
# 		print(obs)
# 		env.render()
# 		if done:
# 			obs = env.reset()
# 			env.render()
# 			print(info)
		