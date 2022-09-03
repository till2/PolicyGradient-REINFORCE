# PG-Reinforce training

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import wandb
import os
from agent import ReinforceAgent
# add argparser

# hyperparams
use_wandb = 1
episodes = 5000
gamma = 0.99
lr = 0.0005

# environment setup
env_name = 'CartPole-v1'
print(f'Training in the {env_name} environment.')
env = gym.make(env_name, new_step_api=True)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f'using device: {device}')

# wandb Setup
if use_wandb:
    wandb.init(project='REINFORCE-CartPoleV1')

# init Agent
agent = ReinforceAgent(n_features=obs_shape, n_actions=action_shape, device=device, lr=lr)
print(agent)

# training loop
for episode in range(episodes):
    
    rewards = []
    action_log_likelihoods = []
    
    # get a trajectory from the current policy
    obs = env.reset()
    for step in range(500):
        action, action_log_likelihood = agent.get_action(obs[None, :])
        obs, reward, done, truncated, info = env.step(action)
        action_log_likelihoods.append(action_log_likelihood)
        rewards.append(reward)
        if done or truncated:
            break
    
    # calculate loss and update policy net params
    loss = agent.get_loss(rewards, action_log_likelihoods, gamma)
    agent.update_policy_net(loss)
    
    # logging
    if use_wandb:
        wandb.log({
            'accumulated_reward': sum(rewards),
            'loss': loss,
            'avg log_likelihood': np.mean([log_l.detach().numpy() for log_l in action_log_likelihoods])
        })
    
    # save the trained weights
    if (episode%100 == 0) and sum(rewards) > 195:
        print(f'saving model: episode {episode} with acc_reward={sum(rewards)}')
        agent.save_params(env_name=env_name, episode=episode, acc_reward=sum(rewards))
    
env.close()