import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MADDPG:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to('cuda')
        self.critic = Critic(state_dim, action_dim, hidden_dim).to('cuda')
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).to('cuda')
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to('cuda')
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to('cuda')
        action = self.actor(state).cpu()
        return action[0]

    def update(self, batch_size, replay_buffer,flag, gamma=0.99, tau=0.005):
        states_1, actions_1, next_states_1, rewards_1, dones_1, states_2, actions_2, next_states_2, rewards_2, dones_2 = replay_buffer.sample(batch_size)
        if flag:
            states, actions, next_states, rewards, dones = states_1, actions_1, next_states_1, rewards_1, dones_1
        else:
            states, actions, next_states, rewards, dones = states_2, actions_2, next_states_2, rewards_2, dones_2
        states = torch.tensor(np.array(states), dtype=torch.float32).to('cuda')
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to('cuda')
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to('cuda')
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to('cuda')
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to('cuda')

        target_actions = self.target_actor(next_states)
        ns = next_states.squeeze()
        ta = target_actions.squeeze()
        target_q_values = self.target_critic(ns, ta)
        q_targets = rewards + gamma * (1 - dones) * target_q_values
        q_values = self.critic(states.squeeze(), actions.squeeze())
        critic_loss = F.mse_loss(q_values, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states.squeeze(), self.actor(states).squeeze()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, state_1, action_1, next_state_1, reward_1, done_1, state_2, action_2, next_state_2, reward_2, done_2):
        experience = (state_1, action_1, next_state_1, reward_1, done_1, state_2, action_2, next_state_2, reward_2, done_2)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states_1, actions_1, next_states_1, rewards_1, dones_1, states_2, actions_2, next_states_2, rewards_2, dones_2 = zip(*batch)
        return states_1, actions_1, next_states_1, rewards_1, dones_1, states_2, actions_2, next_states_2, rewards_2, dones_2

    def __len__(self):
        return len(self.buffer)