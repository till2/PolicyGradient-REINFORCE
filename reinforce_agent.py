import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import os

# PG-Reinforce Agent
class ReinforceAgent(nn.Module):

    def __init__(self, n_features, n_actions, device, lr):
        self.device=device
        super(ReinforceAgent, self).__init__()
        layers = [
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        ]
        self.policy_net = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
    
    def forward(self, x):
        x = self.policy_net(torch.Tensor(x).to(self.device))
        return x
    
    def select_action(self, x):
        """
        Returns a tuple of the chosen action and the log-prob of that action.
        """
        out = self.forward(x)
        pd = torch.distributions.Categorical(logits=out)
        action = pd.sample()
        return (action[0].cpu().detach().numpy(), pd.log_prob(action))
    
    def get_loss(self, rewards, action_log_probs, gamma, device):
        """
        Calculates the loss efficiently.
        """
        T = len(rewards)
        returns = np.empty(T, dtype=np.float32)
        future_return = 0.0

        for t in reversed(range(T)):
            future_return = rewards[t] + gamma * future_return
            returns[t] = future_return
        returns = torch.Tensor(returns).to(device)

        action_log_probs = torch.stack(action_log_probs)
        loss = - action_log_probs * returns
        loss = torch.sum(loss)
        return loss
    
    def update_params(self, loss):
        """
        Updates the parameters of the policy network according to the given loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # clip gradient
        self.optimizer.step()
    
    def save_params(self, env_name=None, episode=None, acc_reward=None):
        """
        Saves the parameters of the agents policy network as a .h5 file.
        """
        weights_path = 'weights'
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        path = os.path.join('weights', f'{env_name}_{self.__class__.__name__}_episode_{episode}_acc_r_{acc_reward:3.0f}.h5')
        torch.save(self.policy_net.state_dict(), path)
    
    def load_params(self, weights_filename):
        """
        Loads parameters in a specified path to the agents policy network and sets eval mode.
        """
        self.policy_net.load_state_dict(torch.load(os.path.join('weights', weights_filename)))
        self.policy_net.eval()
        print('using weights:', weights_filename)