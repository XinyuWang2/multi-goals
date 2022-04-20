from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
import time
import gym
from numpy.linalg import norm
from wall_goalenv_continuous import PointEnv

# Usually the cause of failed task is both loss are so small while ep_len still big,
# thus training gets stuck and episode length and reward don't change


MAX_EPISODE_STEPS = 100
RESIZE = 1
# ENV_NAME = "Spiral5x5"
# ENV_NAME = "Small"
ENV_NAME = "Cross"
TIMESTEPS = int(1e5)

model_path = "trained_models/" + "sac_" + ENV_NAME + "_*" + str(RESIZE) + '_con_' + str(TIMESTEPS)

env = PointEnv(ENV_NAME, resize_factor = RESIZE)
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS) # use gym wrapper
obs = env.reset()


# load it into the loaded_model
model = SAC.load(model_path, env=env)

n_iter = 1000
succes_num = 0
for i in range(n_iter):
	done = False
	sample = False
	obs = env.reset()
	# while not done:
	for i in range(20):
		
		if sample:
			# action = env.action_space.sample()
			# Try sample using Q value

			sample = False
		else:
			action, _state = model.predict(obs, deterministic=True)
		next_obs, reward, done, info = env.step(action)

		if norm(obs["observation"] - next_obs["observation"]) < 1e-5:
			sample = True
			
		# env.render()
		# time.sleep(0.3)
		if done:
			succes_num += 1
			break
		obs = next_obs

print("succes_rate:", succes_num / n_iter)