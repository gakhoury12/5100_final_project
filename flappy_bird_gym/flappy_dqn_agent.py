import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import flappy_bird_gym
from collections import deque


# DQN Neural Network class
def create_dqn(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim)
    )

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9997):
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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
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
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
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

# Training the DQN Agent
# Training the DQN Agent
def train_dqn_agent():
    env = flappy_bird_gym.make("FlappyBird-v0")
    state_dim = env.observation_space.shape[0]  # Using position information
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 20000
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
    train_dqn_agent()