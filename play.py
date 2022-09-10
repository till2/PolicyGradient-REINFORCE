# PG showcase

# To play:
# python play.py --weights LunarLander-v2_ReinforceAgent_episode_46100_acc_r_181.h5

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import argparse
import wandb
import os
from agent import ReinforceAgent

# Init parser
parser = argparse.ArgumentParser(description='Required args: --cuda|--cpu and --weights=filename')
parser.add_argument('--weights', type=str,
                    help='weights [weights_filename]')
parser.add_argument('--cuda', action='store_true',
                    help='--cuda if the gpu should be used')
args = parser.parse_args()

# hyperparams
env_name = 'LunarLander-v2'
episodes = 5

# environment setup
print(f'Showing a trained agent in the {env_name} environment.')
env = gym.make(env_name, render_mode='human') #new_step_api=True,
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# load weights
if args.weights:
    weights_filename = args.weights
else:
    weights_filename = os.listdir('weights')[len(os.listdir('weights'))-1]
    
# init agent
agent = ReinforceAgent(n_features=obs_shape, n_actions=action_shape, device='cpu', lr=0)
agent.load_params(weights_filename)

# showcase loop
for episode in range(episodes):
    acc_reward = 0.0
    # get a trajectory from the trained policy
    obs, _ = env.reset()
    for step in range(250):
        action, action_log_likelihood = agent.get_action(obs[None, :])
        obs, reward, done, truncated, info = env.step(action)
        acc_reward += reward
        if done or truncated:
            break
            
    print(f'acc rewards of episode {episode}: {acc_reward}')        
    
env.close()