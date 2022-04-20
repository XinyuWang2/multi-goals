from wall_goalenv_con_small_a import PointEnv
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import time
import gym


MAX_EPISODE_STEPS = 100
RESIZE = 1
# ENV_NAME = "Spiral7x7"
# ENV_NAME = "Spiral5x5"
ENV_NAME = "Simple"
TIMESTEPS = int(8e4)

model_path = "trained_models/" + "sac_" + ENV_NAME + "_*" + str(RESIZE) + '_con_small_a' + str(TIMESTEPS)

env = PointEnv(ENV_NAME, resize_factor=RESIZE)
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

# model.learn(total_timesteps=TIMESTEPS)
# model.save(model_path)

model_path = "trained_models/sac_Simple_*1_con_small_a80000_0.85"
model = SAC.load(model_path, env=env)


obs = env.reset()
env.render()
for i in range(10):
	print("episode", i, end=" ")
	done = False
	obs = env.reset()
	# while not done:
	for i in range(50):
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		time.sleep(0.1)
		if done:
			print(info)
			break
	print("")