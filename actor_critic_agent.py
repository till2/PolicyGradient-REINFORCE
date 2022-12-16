import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import os

# Actor Critic Agent
class ActorCriticAgent(nn.Module):

    def __init__(self, n_features, n_actions, device, lr):

        self.device=device
        super(ActorCriticAgent, self).__init__()

        # shared body net
        body_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        ]
        self.body = nn.Sequential(*body_layers).to(self.device)

        # actor
        self.policy_head_layers = [
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1),
        ]
        self.policy_head = nn.Sequential(*self.policy_head_layers).to(self.device)

        # critic
        self.value_head_layers = [
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ]
        self.value_head = nn.Sequential(*self.value_head_layers).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    
    def forward(self, x):
        x = self.body(torch.Tensor(x).to(self.device))

        # actor: pick an action
        action_probs = self.policy_head(x)

        # critic: estimate the state-action value
        action_value = self.value_head(x)

        return action_probs, action_value
    

    def select_action(self, x):
        """
        Returns a tuple of the chosen action and the log-prob of that action.
        """
        action_probs, action_value = self.forward(x)
        pd = torch.distributions.Categorical(logits=action_probs)
        action = pd.sample()
        return (action[0].cpu().detach().numpy(), pd.log_prob(action), action_value[0])
    

    def get_loss(self, rewards, action_log_probs, action_values, gamma, device):
        """
        Calculates the loss efficiently.
        """
        T = len(rewards)
        returns = np.empty(T, dtype=np.float32)
        future_return = 0.0

        # Todo: normalize returns here:
        # returns = (returns - returns.mean()) / (returns.std() + eps) # from torch actor_critic.py

        for t in reversed(range(T)):
            future_return = rewards[t] + gamma * future_return
            returns[t] = future_return

        actor_losses = []
        critic_losses = []

        for action_log_prob, action_value, R in zip(action_log_probs, action_values, returns):
            advantage = R - action_value.item()

            # calculate actor (policy) loss
            actor_losses.append(-action_log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            critic_losses.append(F.smooth_l1_loss(action_value, torch.tensor([R])))


        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
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
        torch.save(self.state_dict(), path)
    
    
    def load_params(self, weights_filename):
        """
        Loads parameters in a specified path to the agents policy network and sets eval mode.
        """
        self.load_state_dict(torch.load(os.path.join('weights', weights_filename)))
        self.eval()
        print('using weights:', weights_filename)