# Policy-Gradient RL

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import argparse
import wandb
import os
from reinforce import REINFORCE
from actor_critic import ActorCritic

# parse optional arguments
parser = argparse.ArgumentParser(description='Required args: --cuda|--cpu and --weights=filename')
parser.add_argument('--weights', type=str,
                    help='weights [weights_filename]')
parser.add_argument('--cuda', action='store_true',
                    help='--cuda if the gpu should be used')
parser.add_argument('--wandb', action='store_true',
                    help='--wandb to log the run')
args = parser.parse_args()

# hyperparams
env_name = 'LunarLander-v2'
episodes = 1_000
gamma = 0.99
lr = 0.0001

# environment setup
env = gym.make(env_name) # new_step_api=True
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')

# wandb setup
if args.wandb:
    wandb.init(project='PG-methods')

# init agent
agent = ActorCritic(n_features=obs_shape, n_actions=action_shape, device=device, lr=lr)
print(f'Training {agent.__class__.__name__} in the {env_name} environment...')
print(agent)

# load pretrained weights
#if args.weights:
#    weights_filename = args.weights
#    agent.load_params(weights_filename)
#    agent.policy_net.train()

# training loop
for episode in range(episodes):
    print(episode)
    
    rewards = []
    action_log_likelihoods = []
    
    # get a trajectory from the current policy
    obs, _ = env.reset()
    for step in range(500):
        action, action_log_likelihood = agent.select_action(obs[None, :])
        obs, reward, done, truncated, info = env.step(action)
        action_log_likelihoods.append(action_log_likelihood)
        rewards.append(reward)
        if done or truncated:
            break
    
    # calculate loss and update policy net params
    loss = agent.get_loss(rewards, action_log_likelihoods, gamma, device)
    agent.update_params(loss)
    
    # logging
    if args.wandb:
        wandb.log({
            'accumulated_reward': sum(rewards),
            'loss': loss,
            'avg log_likelihood': np.mean([log_l.detach().numpy() for log_l in action_log_likelihoods])
        })
    
    # save the trained weights
    #if (episode%100 == 0) and sum(rewards) > -100:
    #    print(f'saving model: episode {episode} with acc_reward={sum(rewards)}')
    #    agent.save_params(env_name=env_name, episode=episode, acc_reward=sum(rewards))
    
env.close()