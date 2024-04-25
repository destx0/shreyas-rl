import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Final.env.energy_management import EnergyManagementEnv
from torch.distributions import Normal
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, action_dim)
        self.sigma = nn.Linear(64, action_dim)

    def forward(self, state):
        x = self.network(state)
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))  # To ensure standard deviation is positive
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy_old = Actor(state_dim, action_dim)
        self.policy_old.load_state_dict(self.actor.state_dict())


def run_rl(env_settings):
    env = EnergyManagementEnv(**env_settings)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim, lr_actor=0.003, lr_critic=0.01, gamma=0.99, K_epochs=10, eps_clip=0.2)
    max_episodes = 500

    for episode in range(max_episodes):
        state = env.reset()
        if state is None:
            raise ValueError("Environment reset method returned None, which is invalid.")

        while True:
            state_tensor = torch.from_numpy(state).float()
            mu, sigma = ppo_agent.policy_old(state_tensor)
            dist = Normal(mu, sigma)
            action = dist.sample()
            action = action.clamp(env.action_space.low[0], env.action_space.high[0])  # Ensure action is within bounds
            next_state, reward, done, _ = env.step(action.numpy())

            if next_state is None:
                raise ValueError("Environment step method returned None as part of the next state, which is invalid.")

            state = next_state
            if done:
                break

    torch.save(ppo_agent.actor.state_dict(), 'ppo_energy_management_model.pth')
    return ppo_agent


if __name__ == "__main__":
    env_config = {
        'latitude': 80,
        'longitude': 72,
        'utility_prices': np.random.uniform(0.1, 0.2, 480),
        'prior_purchased': np.random.uniform(0.08, 0.12, 480)
    }
    model = run_rl(env_config)
