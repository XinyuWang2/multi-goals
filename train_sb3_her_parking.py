
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

import gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise

import time

ENV_NAME = "parking-v0"
TIMESTEPS = int(1e5)
MAX_EPISODE_STEPS = 100

model_path = "trained_models/" + ENV_NAME +  "_her_sac_" + '_' + str(TIMESTEPS)

env = gym.make(ENV_NAME)

# Initialize the model
model_class = SAC  
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future', # future, final, episode
        online_sampling=True,
        max_episode_length=MAX_EPISODE_STEPS,
    ),
    verbose=1,
)
# model.buffer_size default to be 1e6

# Train the model
model.learn(TIMESTEPS)
# model.save(model_path)

# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load(model_path, env=env)

obs = env.reset()
env.render()
# Evaluate the agent
episode_reward = 0
for _ in range(1000):
	action, _ = model.predict(obs, deterministic=True)
	obs, reward, done, info = env.step(action)
	episode_reward += reward
    # if done or info[0].get('is_success', False):
    #     print("Reward:", episode_reward, "Success?", info[0].get('is_success', False))
	if done:
		episode_reward = 0.0
		obs = env.reset()
	
	# time.sleep(1)