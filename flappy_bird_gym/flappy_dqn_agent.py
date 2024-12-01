import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import flappy_bird_gym
import matplotlib.pyplot as plt
import pickle
from collections import deque

# DQN Neural Network class
def create_dqn(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    )


# Function to save the model
def save_model(agent, filename="dqn_positional_agent.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.policy_net.state_dict(), f)
    print(f"Model saved to {filename}")

# Function to load the model
def load_model(agent, filename="dqn_positional_agent.pkl"):
    with open(filename, "rb") as f:
        model_state = pickle.load(f)
    agent.policy_net.load_state_dict(model_state)
    print(f"Model loaded from {filename}")


def plot_rewards(rewards,window_size=100):
    # episodes = range(1, len(rewards) + 1)
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(episodes, rewards, label='Rewards', linewidth=2)
    # plt.xlabel('Episodes')
    # plt.ylabel('Total Reward')
    # plt.title('Reward vs. Episodes for Flappy DQN Agent')
    # plt.grid(alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    rewards = np.array(rewards)
    rolling_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    # Plot the rolling average
    plt.figure(figsize=(10, 6))
    plt.plot(range(window_size - 1, len(rewards)), rolling_avg,
             label=f"Rolling Avg (window={window_size})", color='blue', linewidth=2)
    plt.title("Rolling Average Reward vs Number of Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Rolling Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.995, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=200000)
        
        # Networks
        self.policy_net = create_dqn(state_dim, action_dim)
        self.target_net = create_dqn(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        # Synchronize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action (flap or no flap)
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state))
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample mini-batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Calculate target Q-values
        current_q_values = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()
        next_actions = self.policy_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions.view(-1, 1)).squeeze()
        # next_q_values = self.target_net(next_states).max(1)[0]
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

# Training the DQN Agent
def train_dqn_agent():
    env = flappy_bird_gym.make("FlappyBird-v0")
    state_dim = env.observation_space.shape[0]  # Using position information
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 20000
    target_update_freq = 10
    batch_size = 256
    rewards=[]

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
            reward = np.clip(reward, -1, 1)
            agent.store_transition(state, action, reward, next_state, done)

            # Train the agent
            agent.train(batch_size)

            state = next_state
            total_reward += reward

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
        rewards.append(total_reward)

    plot_rewards(rewards)
    save_model(agent, "dqn_positional_agent.pkl")
    env.close()


if __name__ == "__main__":
    train_dqn_agent()