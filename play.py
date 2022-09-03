# PG-Reinforce showcase

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import wandb
import os
from agent import ReinforceAgent
# add argparser

# Hyperparams
episodes = 5
weights_filename = 'CartPole-v1_ReinforceAgent_episode_3900_acc_r_500.h5'
# weights_filename = os.listdir('weights')[len(os.listdir('weights'))-1]

# Environment setup
env_name = 'CartPole-v1'
print(f'Showing a trained agent in the {env_name} environment.')
env = gym.make(env_name, new_step_api=True, render_mode='human')
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# Init agent
agent = ReinforceAgent(n_features=obs_shape, n_actions=action_shape, device='cpu', lr=0)
agent.load_params(weights_filename)

# Showcase loop
for episode in range(episodes):
    acc_reward = 0.0
    # get a trajectory from the trained policy
    obs = env.reset()
    while(1):
        action, action_log_likelihood = agent.get_action(obs[None, :])
        obs, reward, done, truncated, info = env.step(action)
        acc_reward += reward
        if done or truncated:
            break
            
    print(f'acc rewards of episode {episode}: {acc_reward}')        
    
env.close()