from wall.wall_env_goals import PointEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import time
import gym


MAX_EPISODE_STEPS = 100
RESIZE = 1
# ENV_NAME = "Spiral7x7"
# ENV_NAME = "Spiral5x5"
ENV_NAME = "Small"
TIMESTEPS = int(4e4)

model_path = "trained_models/" + ENV_NAME + str(RESIZE)

env = PointEnv(ENV_NAME, resize_factor=RESIZE)
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS) # use gym wrapper
obs = env.reset()

# model = PPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=TIMESTEPS)
# model.save(model_path)

# model = PPO.load(model_path)
model = PPO.load("trained_models/Small1_0.98")


obs = env.reset()
env.render()
for i in range(10):
	print("episode", i, end=" ")
	done = False
	obs = env.reset()
	# while not done:
	for i in range(10):
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			print(info)
			break
	