from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.logger import configure

import time
import gym
from wall_goalenv_continuous import PointEnv

# Usually the cause of failed task is both loss are so small while ep_len still big,
# thus training gets stuck and episode length and reward don't change


MAX_EPISODE_STEPS = 100
RESIZE = 1
# ENV_NAME = "Spiral7x7"
# ENV_NAME = "Spiral5x5"
# ENV_NAME = "Small"
ENV_NAME = "Cross"
TIMESTEPS = int(1e5)

model_path = "trained_models/" + "sac_" + ENV_NAME + "_*" + str(RESIZE) + '_con_' + str(TIMESTEPS)

env = PointEnv(ENV_NAME, resize_factor = RESIZE)
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS) # use gym wrapper
obs = env.reset()

# Initialize the model
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        online_sampling=True,
        max_episode_length=MAX_EPISODE_STEPS,
    ),
    verbose=1,
)

# # # # # Set up Logger # # # # #
log_path = "sb3_log/"
new_logger = configure(log_path, ["stdout", "csv"])
model.set_logger(new_logger)

# # # # # Train and Save model # # # # #
model.learn(total_timesteps=TIMESTEPS)
model.save(model_path)
# model.save_replay_buffer("sac_replay_buffer") # now save the replay buffer too

# # # # # Load model # # # # #
# model = SAC.load(model_path, env=env)
# model.load_replay_buffer("sac_replay_buffer")
# model.learn(total_timesteps=int(2e4))


obs = env.reset()
env.render()

for i in range(10):
	done = False
	obs = env.reset()
	print("episode", i, end="")
	# while not done:
	for i in range(10):
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		time.sleep(0.3)
		if done:
			print(info)
			# time.sleep(0.3)
			break
	# time.sleep(1)
	print("")