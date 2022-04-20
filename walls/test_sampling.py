from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
import time
import gym
import numpy as np
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


# load it into the loaded_model
model = SAC.load(model_path, env=env)

def test(n_iter, sample_action_bound, do_sample):
	# Set randomness
	np.random.seed(0)

	succes_num = 0
	lengths = []
	for _ in range(n_iter):
		done = False
		sample_this_step = False
		length = 0
		obs = env.reset()
		# while not done:
		for _ in range(20):
			
			if sample_this_step:
				assert do_sample == True
				action = env.action_space.sample() * sample_action_bound
				# Try sample using Q value
				sample_this_step = False
			else:
				action, _state = model.predict(obs, deterministic=True)
			next_obs, reward, done, info = env.step(action)

			step_size = norm(obs["observation"] - next_obs["observation"])
			length += step_size

			if do_sample and step_size < 1e-5:
				sample_this_step = True
				
			# env.render()
			# time.sleep(0.3)
			if done:
				succes_num += 1
				lengths.append(length)
				break
			
			obs = next_obs
		

	succes_rate = succes_num / n_iter
	lengths = np.array(lengths)
	return succes_rate, lengths
	

# We can later use length of RL / length of A*
# But right now, lets just take a quick look of difference between sampling methods

# sr1, lengths1 = test(3000, 1, True)
# sr2, lengths2 = test(3000, 0.3, True)
# sr3, lengths3 = test(3000, 0.6, True)
# sr4, lengths4 = test(3000, 1, False)

# print(sr1, sr2, sr3, sr4)
# print(lengths1.mean(), lengths2.mean(), lengths3.mean(), lengths4.mean())

# 0.8833333333333333 0.8666666666666667 0.8166666666666667 0.7183333333333334
# 5.370242567985266  5.222501261407206 4.996503849704372 4.825832408733381
