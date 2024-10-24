import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import flappy_bird_gym
from collections import deque
from torchvision import transforms

# DQN Neural Network for RGB Frames (Convolutional)
class DQNRGB(nn.Module):
    def __init__(self, action_dim):
        super(DQNRGB, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgentRGB:
    def __init__(self, action_dim, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Networks
        self.policy_net = DQNRGB(action_dim)
        self.target_net = DQNRGB(action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Synchronize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Transform for RGB frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action (flap or no flap)
        else:
            with torch.no_grad():
                state = self.transform(state).unsqueeze(0)  # Add batch dimension
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample mini-batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert state images to tensors
        states = torch.stack([self.transform(np.array(state)) for state in states])
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack([self.transform(np.array(next_state)) for next_state in next_states])
        dones = torch.FloatTensor(dones)

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
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training the DQN Agent for RGB Frames
def train_dqn_agent_rgb():
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    action_dim = env.action_space.n

    agent = DQNAgentRGB(action_dim)
    num_episodes = 1000
    target_update_freq = 10
    batch_size = 64

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Render the environment
            env.render()

            # Select action using epsilon-greedy policy
            action = agent.select_action(state)
            
            # Step in the environment
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # Train the agent
            agent.train(batch_size)

            state = next_state
            total_reward += reward

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    env.close()

if __name__ == "__main__":
    train_dqn_agent_rgb()