from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from wall_goalenv import PointEnv
import time

# Spiral5x5

model_class = DQN  # works also with SAC, DDPG and TD3
RESIZE = 2
ENV_NAME = "Small"
MAX_EPISODE_STEPS = 100 # Time limit for the episodes
TIMESTEPS = int(3e5)

model_path = "trained_models/" + ENV_NAME + "_*" + str(RESIZE) + "_her" + '_' + str(TIMESTEPS)

env = PointEnv(ENV_NAME, resize_factor = RESIZE)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = False  #True


# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=MAX_EPISODE_STEPS,
    ),
    verbose=1,
)
# model.buffer_size default to be 1e6

# Train the model
model.learn(TIMESTEPS)
model.save(model_path)

# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load('./her_bit_env', env=env)

obs = env.reset()
env.render()

for i in range(10):
	done = False
	obs = env.reset()
	# while not done:
	for i in range(10):
		action, _state = model.predict(obs, deterministic=True)
		time.sleep(0.5)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			print(info)
			time.sleep(0.3)
			break
	
	time.sleep(1)