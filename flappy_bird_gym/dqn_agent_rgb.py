# flappy_bird_gym/dqn_agent_rgb.py

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from prioritized_replay_buffer import PrioritizedReplayBuffer

class DQNRGB(nn.Module):
    def __init__(self, action_dim, architecture='original'):
        super(DQNRGB, self).__init__()
        if architecture == 'original':
            # Original architecture
            self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

            self.dummy_input = torch.zeros(1, 3, 64, 64)
            self.feature_map_size = self._get_conv_output(self.dummy_input)

            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_map_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_map_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim)
            )
        elif architecture == 'reduced':
            # Reduced architecture for computational efficiency
            self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

            self.dummy_input = torch.zeros(1, 3, 64, 64)
            self.feature_map_size = self._get_conv_output(self.dummy_input)

            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_map_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_map_size, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
        else:
            raise ValueError("Invalid architecture type")

    def _get_conv_output(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return int(torch.numel(x))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgentRGB:
    def __init__(self, action_dim, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, architecture='original'):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQNRGB(action_dim, architecture=architecture).to(self.device)
        self.target_net = DQNRGB(action_dim, architecture=architecture).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.scaler = GradScaler()

    def preprocess(self, state):
        """
        Preprocesses the input state:
        - Resizes to 64x64
        - Converts to tensor and normalizes pixel values
        - Permutes dimensions to match PyTorch's (C, H, W)
        """
        state = cv2.resize(state, (64, 64))
        state = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        return state

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        state = self.preprocess(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode ended.
        """
        state = self.preprocess(state)
        next_state = self.preprocess(next_state)
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self, batch_size=128, beta=0.4):
        """
        Trains the policy network using a batch of experiences.

        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): Importance-sampling weight annealing parameter.
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample from prioritized replay buffer
        samples = self.replay_buffer.sample(batch_size, beta=beta)
        states, actions, rewards, next_states, dones, weights, indices = samples

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        with autocast():
            current_q_values = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.view(-1, 1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute TD errors for prioritized replay buffer
            td_errors = target_q_values.detach() - current_q_values

            # Compute Huber loss with importance sampling weights
            loss = (weights * torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values.detach(), reduction='none')).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update priorities in the replay buffer
        priorities = td_errors.abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        # Adjust epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        """
        Updates the target network to match the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, folder_path="models", best=False, trial_number=1):
        """
        Saves the policy network's state dictionary.

        Args:
            folder_path (str): Directory to save the model.
            best (bool): If True, saves as the best model.
            trial_number (int): Identifier for the trial.
        """
        os.makedirs(folder_path, exist_ok=True)
        suffix = f"trial_{trial_number}_best" if best else f"trial_{trial_number}_policy_net"
        torch.save(self.policy_net.state_dict(), os.path.join(folder_path, f"{suffix}.pth"))
        print(f"Model saved: {folder_path}/{suffix}.pth")

    def load_model(self, folder_path="models", best=False, trial_number=1):
        """
        Loads the policy network's state dictionary.

        Args:
            folder_path (str): Directory where the model is saved.
            best (bool): If True, loads the best model.
            trial_number (int): Identifier for the trial.
        """
        model_file = f"trial_{trial_number}_best.pth" if best else f"trial_{trial_number}_policy_net.pth"
        path = os.path.join(folder_path, model_file)
        if os.path.exists(path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net.load_state_dict(torch.load(path, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded: {path}")
        else:
            print("No saved model found.")
