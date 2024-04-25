import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class MADDPG:
    def __init__(
        self,
        n_agents,
        state_size,
        action_size,
        hidden_size=256,
        gamma=0.99,
        tau=0.01,
        lr_actor=1e-4,
        lr_critic=1e-3,
    ):
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau

        self.actors = [
            Actor(state_size, action_size, hidden_size) for _ in range(n_agents)
        ]
        self.critics = [
            Critic(state_size * n_agents, action_size * n_agents, hidden_size)
            for _ in range(n_agents)
        ]
        self.actors_target = [
            Actor(state_size, action_size, hidden_size) for _ in range(n_agents)
        ]
        self.critics_target = [
            Critic(state_size * n_agents, action_size * n_agents, hidden_size)
            for _ in range(n_agents)
        ]

        for i in range(n_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.actor_optimizers = [
            optim.Adam(self.actors[i].parameters(), lr=lr_actor)
            for i in range(n_agents)
        ]
        self.critic_optimizers = [
            optim.Adam(self.critics[i].parameters(), lr=lr_critic)
            for i in range(n_agents)
        ]

        self.memory = []

    def store_transition(self, states, actions, rewards, next_states, dones):
        self.memory.append((states, actions, rewards, next_states, dones))

    def act(self, states, noise=0.1):
        actions = []
        for i in range(self.n_agents):
            if isinstance(states[i], np.ndarray):
                state = torch.FloatTensor(states[i]).unsqueeze(0)
            else:
                state = torch.FloatTensor([states[i]])
            action = self.actors[i](state).squeeze(0).detach().numpy()
            action += np.random.normal(0, noise, size=self.action_size)
            action = np.clip(action, -1, 1)
            actions.append(action)
        return np.array(actions)

    def learn(self, batch_size=128):
        if len(self.memory) < batch_size:
            return

        samples = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.memory[idx] for idx in samples]
        )

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1)

        for i in range(self.n_agents):
            next_actions_target = [
                self.actors_target[j](next_states[:, j]) for j in range(self.n_agents)
            ]
            next_actions_target = torch.cat(next_actions_target, dim=1)

            q_next = self.critics_target[i](
                next_states.view(batch_size, -1), next_actions_target
            ).detach()
            q_target = rewards[:, i] + self.gamma * q_next * (1 - dones[:, i])

            q = self.critics[i](
                states.view(batch_size, -1), actions.view(batch_size, -1)
            )
            critic_loss = nn.MSELoss()(q, q_target)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            actions_pred = [
                (
                    self.actors[j](states[:, j])
                    if j == i
                    else self.actors[j](states[:, j]).detach()
                )
                for j in range(self.n_agents)
            ]
            actions_pred = torch.cat(actions_pred, dim=1)

            actor_loss = -self.critics[i](
                states.view(batch_size, -1), actions_pred
            ).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        for i in range(self.n_agents):
            for target_param, param in zip(
                self.actors_target[i].parameters(), self.actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for target_param, param in zip(
                self.critics_target[i].parameters(), self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
