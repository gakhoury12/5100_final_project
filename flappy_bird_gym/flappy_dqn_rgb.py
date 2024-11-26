import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import flappy_bird_gym
import matplotlib.pyplot as plt
from collections import deque
from torch.cuda.amp import GradScaler, autocast


class DQNRGB(nn.Module):
    def __init__(self, action_dim):
        super(DQNRGB, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(2 * 2 * 512, 1024)  # Updated the input size of the first fully connected layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))  # New layer
        x = torch.relu(self.conv5(x))  # New layer
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgentRGB:
    def __init__(self, action_dim, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9999):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = deque(maxlen=100000)

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQNRGB(action_dim).to(self.device)
        self.target_net = DQNRGB(action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.scaler = GradScaler()

    def preprocess(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        return torch.nn.functional.interpolate(state.unsqueeze(0), size=(84, 84)).squeeze(0)

    def select_action(self, state):
        state = self.preprocess(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        state = self.preprocess(state)
        next_state = self.preprocess(next_state)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        with autocast():
            # Double DQN: Use policy network to select next actions and target network to calculate next Q-values
            current_q_values = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.view(-1, 1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, folder_path="models"):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(folder_path, "policy_net.pth"))
        torch.save(self.target_net.state_dict(), os.path.join(folder_path, "target_net.pth"))
        print(f"Models saved in folder: {folder_path}")


def train_dqn_agent_rgb(isPlot=False):
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    action_dim = env.action_space.n
    agent = DQNAgentRGB(action_dim)
    num_episodes = 1000
    target_update_freq = 10
    batch_size = 512

    rewards = []
    average_rewards = []

    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    line, = ax.plot([], [], label="Rewards per Episode", color="blue")
    ax.legend()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train(batch_size)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if episode % target_update_freq == 0:
            agent.update_target_network()

        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards[-100:]) / 100
            average_rewards.append(avg_reward)
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        # Update the plot after each episode
        line.set_xdata(range(len(rewards)))
        line.set_ydata(rewards)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    agent.save_model()

    plt.ioff()  # Turn off interactive mode after training
    plt.show()

    print(f"Training completed. Final Average Reward: {sum(rewards[-100:]) / 100:.2f}")
    env.close()


if __name__ == "__main__":
    train_dqn_agent_rgb(isPlot=True)
