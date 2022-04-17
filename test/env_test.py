import gym
# import numpy as np
# import mujoco_py

from stable_baselines3 import HerReplayBuffer, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise

# CartPole-v0
max_steps = 100    
exploration_action_noise = 0.1


max_steps = 100

# env = gym.make("BipedalWalker-v3")
env = gym.make("FetchReach-v1").unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

obs = env.observation_space.sample()

# model = DDPG(
#     "MultiInputPolicy",
#     env,
#     replay_buffer_class=HerReplayBuffer,
#     # Parameters for HER
#     replay_buffer_kwargs=dict(
#         max_episode_length=max_steps,
#         # n_sampled_goal=4, # default value
#         # goal_selection_strategy=goal_selection_strategy, # defult "future"
#         # online_sampling=True,
#     ),
#     verbose=1,
# )

# # Train for 1e5 steps
# model.learn(int(1e5))
# # Save the trained agent
# model.save('ddpg_fetch.pth')