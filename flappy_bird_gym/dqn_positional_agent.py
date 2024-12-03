from collections import deque
import random
import torch
import torch.optim as optim
import torch.nn as nn
from flappy_bird_gym.flappy_dqn_agent import create_dqn
import numpy as np


class DQNAgent:
    """
    The class to create a Deep Q-Network (DQN) agent for the positional model.
    """

    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.999, epsilon_start=1.0, epsilon_min=0.01,
                 epsilon_decay=0.9997):
        """
        The constructor to initialize the DQN agent.
        Args:
            state_dim: The state dimension
            action_dim: The action dimension
            learning_rate: The learning rate with default value 1e-4
            gamma: The gamma value with default value 0.999
            epsilon_start: The epsilon start value with default value 1.0
            epsilon_min: The epsilon minimum value with default value 0.01
            epsilon_decay: The decay value for epsilon with default value 0.9997
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Replay buffer
        self.replay_buffer = deque(maxlen=50000)

        # Networks
        self.policy_net = create_dqn(state_dim, action_dim)
        self.target_net = create_dqn(state_dim, action_dim)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)

        # Synchronize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        """
        The function to select an action based on the state.
        Args:
            state: The current state

        Returns: The action based on the state

        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action (flap or no flap)
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state))
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        The function to store the transition in the replay buffer.
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
            done: The flag to indicate if the episode is done

        Returns: None

        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        """
        The function to train the agent using the replay buffer.
        Args:
            batch_size: The batch size with default value 64

        Returns: None

        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample mini-batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # Calculate target Q-values
        current_q_values = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss and optimization step
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        """
        The function to update the target
        Returns: None

        """
        self.target_net.load_state_dict(self.policy_net.state_dict())