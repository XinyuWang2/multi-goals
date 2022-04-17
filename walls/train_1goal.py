import gym
from wall_env_1goals import PointEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import time

MAX_EPISODE_STEPS = 1000

env = PointEnv("Spiral9x9", resize_factor = 1)
# env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS) # use gym wrapper
obs = env.reset()


# model = PPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=30000)

# obs = env.reset()
# env.render()
# for i in range(5):
# 	done = False
# 	while not done:
# 		action, _state = model.predict(obs, deterministic=True)
# 		obs, reward, done, info = env.step(action)
# 		# print("action", action)
# 		# print("obs", obs)
# 		env.render()
# 		if done:
# 			obs = env.reset()
# 			print(info)
# 		time.sleep(0.1)


obs = env.reset()
env.render()
for i in range(5):
	done = False
	while not done:
		action = env.action_space.sample()
		obs, reward, done, info = env.step(action)
		# print("action", action)
		# print("obs", obs)
		env.render()
		if done:
			obs = env.reset()
			print(info)
		time.sleep(0.5)