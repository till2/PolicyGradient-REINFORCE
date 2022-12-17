import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import os

# https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic

# Advantage Actor Critic:
# A(s_t, a_t) = G_t - V(s_t)

class ActorCritic(nn.Module):

    def __init__(self, n_features:int, n_actions:int, device, lr:float) -> None:
        self.device=device
        super(ActorCritic, self).__init__()

        body_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
        ]

        critic_layers = [
            nn.Linear(32, 1), # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(32, n_actions), # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.body = nn.Sequential(*body_layers).to(self.device)
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
    
    def forward(self, x:tuple) -> tuple(torch.Tensor, torch.Tensor):
        x = self.body(torch.Tensor(x)).to(self.device)
        state_value = self.critic(x)
        action_logits = self.actor(x)
        return (state_value, action_logits)
    
    def select_action(self, x:tuple) -> tuple(int, float, torch.Tensor):
        """
        Returns a tuple of the chosen action and the log-prob of that action.
        """
        state_value, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(logits=action_logits)
        action = action_pd.sample()[0].cpu().detach().numpy()
        action_log_prob = action_pd.log_prob(action)
        return (action, action_log_prob, state_value)
    
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

    def update_critic(self, critic_loss):
        """
        Updates the parameters of the critic network according to the given loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def update_actor(self, actor_loss):
        """
        Updates the parameters of the actor network according to the given loss.
        """
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
    
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


# torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # clip gradient