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
from dqn_positional_agent import DQNAgent


if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")


def create_dqn(input_dim, output_dim):
    """
    The function to creates a Deep Q-Network (DQN) neural network model for the positional model.
    Args:
        input_dim: The input dimension from the flappy bird environment
        output_dim: The action for the flappy bird environment

    Returns: The DQN model

    """
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


def train_dqn_agent(resume_training=False, pickle_file='/models/trained_agent_dqn.pkl'):
    """
    The function to train the DQN agent for the positional model.
    Args:
        resume_training: The flag to resume training
        pickle_file:  The pickle file to save the trained agent

    Returns: None

    """
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

    with open('models/trained_agent_dqn_2.pkl', 'wb') as f:
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



def play_game_with_trained_agent():
    """
    The function to play the game with the trained agent.
    Returns: None

    """
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
    """
    The main function to train the DQN agent for the positional model.
    """
    train_dqn_agent(resume_training=True)