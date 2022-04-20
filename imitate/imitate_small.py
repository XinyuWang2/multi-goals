from wall_env_goals import PointEnv
from stable_baselines3 import PPO
import time
import gym
import torch

from BC.bc_agent import BCAgent

MAX_EPISODE_STEPS = 10
RESIZE = 1
# ENV_NAME = "Spiral7x7"
# ENV_NAME = "Spiral5x5"
ENV_NAME = "Small"

assert torch.cuda.is_available()
DEVICE = torch.device("cuda")


model_path = "trained_models/" + ENV_NAME + str(RESIZE)

env = PointEnv(ENV_NAME, resize_factor=RESIZE)
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS) # use gym wrapper


model = PPO.load("trained_models/Small1_0.98")

actino_dim = env.action_space.n # 4
obs_dim = env.observation_space.shape[0] # 2

batch_size = 500 # the batch to update policy
buffer_size = int(1e4) # maintained agent replay buffer
ratio = 0.8 # ratio of old transitions in the batch for training
new_size = int(batch_size * ratio) # size of new transition in training batch
replay_size = batch_size - new_size # size of new transition in training batch


# hidden_size = 64 # or 32
agent_params = {
	'ac_dim': actino_dim,
	'ob_dim': obs_dim, 
	'n_layers': 2,
	'hidden_size': 32, 
	'device': DEVICE,
	'learning_rate': 1e-2,
	'max_replay_buffer_size': buffer_size,
	'replay_size':replay_size
}

agent = BCAgent(env, agent_params)

# BC training
n_iter = 20
for i in range(n_iter):
	new_obs_batch = []
	new_action_batch = []

	# collect a small batch expert trajectories
	while (len(new_obs_batch) < batch_size):
		done = False
		obs = env.reset()
		
		while not done:
			action, _state = model.predict(obs, deterministic=True)
			
			# collect
			new_obs_batch.append(obs)
			new_action_batch.append(action)
			
			obs, reward, done, info = env.step(action)

			if done:
				break

	agent.add_to_replay_buffer(new_obs_batch, new_action_batch)


n = 20000
loss_values = []
for i in range(n):
	obs_batch, action_batch = agent.sample(batch_size)
	loss = agent.actor.update(obs_batch, action_batch)
	loss_values.append(loss)

agent.actor.save("actor.pt")

import matplotlib.pyplot as plt
rbuffer_size = len(agent.replay_buffer)
name = "rb" + str(rbuffer_size) + '_b' + str(batch_size)  +'_lr' + str(agent_params['learning_rate']) + '_n' + str(agent_params['n_layers']) + '_h' + str(agent_params['hidden_size'])
plt.plot(loss_values)
plt.savefig("fig/" + name + ".png")
plt.show()

