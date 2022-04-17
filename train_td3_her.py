from collections import deque
import gym
import random
import numpy as np
import torch

from algo.TD3 import TD3
from algo.rbuffer import HerReplayBuffer

# parameters:
max_steps = 100            # Maximum time steps for one episode
update_every = 200         # number of env interactions that should elapse between updates of Q-networks.
exploration_action_noise = 0.1

# env:
# currently HER can only run GoalEnv 
env_name = "FetchReach-v1"
# env_name = "MountainCarContinuous-v0"
# env_name = "BipedalWalker-v3"
env = gym.make(env_name)

# HER parameters:
buffer_size = int(1e5)      # number of transitions stored in the buffer


# TD3 parameters:
batch_size = 100            # num of transitions sampled from replay buffer
n_iter = 200                # update policy n_iter times in one DDPG update
obs_dim = env.observation_space["observation"].shape[0]
goal_dim = env.observation_space["achieved_goal"].shape[0]
state_size = obs_dim + goal_dim
action_size = env.action_space.shape[0]
action_upper_bound = env.action_space.high  # action space upper bound
action_lower_bound = env.action_space.low  # action space lower bound

# Other
timesteps_count = 0  # Counting the time steps
max_steps = 200  # Maximum time steps for one episode
ep_reward_list = deque(maxlen=50)
avg_reward = -9999


# reset env and set seed 
env.reset()
env.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


agent = TD3(state_size, action_size, action_upper_bound, action_lower_bound)
replay_buffer = HerReplayBuffer(env, buffer_size, max_episode_length=max_steps)


# we run 600 episodes
for ep in range(600):
    obs = env.reset()
    state = np.concatenate((obs["observation"], obs["desired_goal"]))
    episodic_reward = 0
    timestep_for_cur_episode = 0

    for st in range(max_steps):
        timestep_for_cur_episode += 1     
        timesteps_count += 1

        # Select action according to policy
        action = agent.policy(state)
        action = action + np.random.normal(0, exploration_action_noise)
        action = action.clip(action_lower_bound, action_upper_bound)

        # Recieve state and reward from environment.
        obs, reward, done, info = env.step(action)
        state = np.concatenate((obs["observation"], obs["desired_goal"]))

        env.render()

        episodic_reward += reward
        
        replay_buffer.add(obs, action, reward, done, info)
              
        if replay_buffer.n_episodes_stored < batch_size:
            continue

        if timestep_for_cur_episode % update_every == 0: # We update once every xx environment interations
            # Send the experience to the agent and train the agent
            agent.update(replay_buffer, n_iter, batch_size)
        
        
        # End this episode when `done` is True
        if done:
            break
        # state = next_state

    ep_reward_list.append(episodic_reward)
    print('Ep. {}, Ep.Timesteps {}, Episode Reward: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='')
    
    if len(ep_reward_list) == 50:
        # Mean of last 50 episodes
        avg_reward = sum(ep_reward_list) / 50
        print(', Moving Average Reward: {:.2f}'.format(avg_reward))
    else:
        print('')

print('Average reward over 50 episodes: ', avg_reward)
env.close()

# Save the actor
actor_path = "actor.pth"
torch.save(agent.actor.state_dict(), actor_path)


