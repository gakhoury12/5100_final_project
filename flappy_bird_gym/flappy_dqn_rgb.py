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
import contextlib
import cv2


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
    def __init__(self, action_dim, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = deque(maxlen=100000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQNRGB(action_dim).to(self.device)
        self.target_net = DQNRGB(action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.scaler = GradScaler()

    def preprocess(self, state):
        # Convert the state to a numpy array, resize using OpenCV, and normalize
        state = cv2.resize(state, (64, 64))  # Resize using OpenCV to 64x64
        state = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        return state

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

    def save_model(self, folder_path="models", best=False):
        os.makedirs(folder_path, exist_ok=True)
        suffix = "best" if best else "policy_net"
        torch.save(self.policy_net.state_dict(), os.path.join(folder_path, f"{suffix}.pth"))
        print(f"Model saved: {folder_path}/{suffix}.pth")

    def load_model(self, folder_path="models", best=False):
        model_file = "best.pth" if best else "policy_net.pth"
        path = os.path.join(folder_path, model_file)
        if os.path.exists(path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net.load_state_dict(torch.load(path , map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded: {path}")
        else:
            print("No saved model found.")


def train_dqn_agent_rgb(isPlot=False, isTraining=True):
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    action_dim = env.action_space.n
    agent = DQNAgentRGB(action_dim)
    agent.load_model(best=True)

    num_episodes = 10000
    target_update_freq = 10
    batch_size = 512

    rewards = []
    best_reward = -float('inf')

    for episode in range(num_episodes):
        print(f'Episode : {episode + 1}')
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if not isTraining:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            clipped_reward = max(min(reward, 1.0), -1.0)

            if isTraining:
                agent.store_transition(state, action, clipped_reward, next_state, done)
                agent.train(batch_size)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if isTraining:
            if episode % target_update_freq == 0:
                agent.update_target_network()

            avg_reward = sum(rewards[-100:]) / len(rewards[-100:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(best=True)

            if (episode + 1) % 100 == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(rewards, label="Reward per Episode")
                plt.xlabel("Episodes")
                plt.ylabel("Reward")
                plt.title("Training Rewards")
                plt.legend()
                plt.savefig(f"reward_plot_{episode + 1}.png")
                plt.close()
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    if isTraining:
        agent.save_model()
    env.close()


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()  # Flush the stream immediately

    def flush(self):
        self.stream.flush()

if __name__ == '__main__':
    # log_file = "training_log.txt"
    # with open(log_file, "w") as f:
    #     unbuffered_f = Unbuffered(f)  # Wrap the file in an unbuffered wrapper
    #     with contextlib.redirect_stdout(unbuffered_f):
    train_dqn_agent_rgb(True, True)
        # Convert the state to a numpy array, resize using OpenCV, and normalize
        state = cv2.resize(state, (64, 64))  # Resize using OpenCV to 64x64
            if not isTraining:
                env.render()

