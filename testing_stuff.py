# Policy-Gradient RL

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import argparse
import wandb
import os
from reinforce_agent import ReinforceAgent
from actor_critic_agent import ActorCriticAgent

# hyperparams
env_name = 'LunarLander-v2'
episodes = 1_000
gamma = 0.99
lr = 0.001

# environment setup
print(f'Training in the {env_name} environment.')
env = gym.make(env_name) # new_step_api=True
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

device = torch.device('cpu')

# init agent
agent = ActorCriticAgent(n_features=obs_shape, n_actions=action_shape, device=device, lr=lr)
print(agent.parameters())

# get a trajectory from the current policy
obs, _ = env.reset()

print(obs)
print(agent.forward(obs))

env.close()