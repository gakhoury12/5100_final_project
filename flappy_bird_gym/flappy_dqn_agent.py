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
import os

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")

    
# DQN Neural Network class
def create_dqn(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512),  # First fully connected layer
        nn.LeakyReLU(),        
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
        nn.Linear(128, output_dim)  # Output layer (Q-values for each action)
    )

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.999, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9997):
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
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action (flap or no flap)
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state))
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
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
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training the DQN Agent
# Training the DQN Agent
def train_dqn_agent(resume_training=False, pickle_file='trained_agent_dqn.pkl'):
    env = flappy_bird_gym.make("FlappyBird-v0")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    state_dim = env.observation_space.shape[0]  # Using position information
    action_dim = env.action_space.n

    print(resume_training)
    print(os.path.exists(pickle_file))
    if resume_training and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            agent = pickle.load(f)
        print("Loaded agent from pickle file.")
    else:
        agent = DQNAgent(state_dim, action_dim)
        print("Initialized new agent.")
    num_episodes = 10000
    target_update_freq = 10
    batch_size = 64

    # Variables to track average rewards
    avg_rewards_100 = []
    avg_rewards_1000 = []
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Render the environment
            # env.render()

            # Select action using epsilon-greedy policy
            action = agent.select_action(state)
            
            # Step in the environment
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # Train the agent
            agent.train(batch_size)

            state = next_state
            total_reward += reward
        # Store the total reward for the current episode
        episode_rewards.append(total_reward)

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Calculate and store the average reward every 100 and 1000 episodes
        if (episode + 1) % 100 == 0:
            avg_100 = np.mean(episode_rewards[-100:])
            avg_rewards_100.append(avg_100)

        if (episode + 1) % 1000 == 0:
            avg_1000 = np.mean(episode_rewards[-1000:])
            avg_rewards_1000.append(avg_1000)

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}, average:{avg_100}")

    env.close()

    with open('trained_agent_dqn_2.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print("Trained agent saved to 'trained_agent_dqn_2.pkl'.")

    # Save Plot 1: Average reward every 100 episodes
    plt.figure(figsize=(12, 6))
    plt.plot(range(100, num_episodes + 1, 100), avg_rewards_100, label="Average Reward (100 episodes)", color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Every 100 Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_avg_rewards_100.png")  # Save as PNG file
    print("Plot saved as dqn_avg_rewards_100.png")

    # Save Plot 2: Average reward every 1000 episodes
    plt.figure(figsize=(12, 6))
    plt.plot(range(1000, num_episodes + 1, 1000), avg_rewards_1000, label="Average Reward (1000 episodes)", color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Every 1000 Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_avg_rewards_1000.png")  # Save as PNG file
    print("Plot saved as dqn_avg_rewards_1000.png")


    # Play the game using the trained model
def play_game_with_trained_agent():
    # Load the trained agent from the pickle file
    with open('trained_agent_dqn_1.pkl', 'rb') as f:
        agent = pickle.load(f)
    
    env = flappy_bird_gym.make("FlappyBird-v0")
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select the best action based on the trained model
        action = agent.select_action(state)
        
        # Step in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Render the environment (optional, but useful for visualizing the agent playing)
        env.render()

        state = next_state
        total_reward += reward

    print(f"Total reward from the trained agent: {total_reward}")
    env.close()


if __name__ == "__main__":
    train_dqn_agent(resume_training=True)