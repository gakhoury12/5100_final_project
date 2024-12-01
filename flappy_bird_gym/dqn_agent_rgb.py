import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.cuda.amp import GradScaler
from prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNRGB(nn.Module):
    def __init__(self, action_dim):
        super(DQNRGB, self).__init__()
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
    def __init__(self, action_dim, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNRGB(action_dim).to(self.device)

    def preprocess(self, state):
        state = cv2.resize(state, (64, 64))
        state = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        return state

    def select_action(self, state):
        state = self.preprocess(state).unsqueeze(0)
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()
