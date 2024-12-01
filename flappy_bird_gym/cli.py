# flappy_bird_gym/cli.py

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import flappy_bird_gym
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to avoid Qt warnings
import matplotlib.pyplot as plt
from collections import deque
from torch.cuda.amp import GradScaler, autocast
import cv2
import numpy as np
import json
import datetime  # Import datetime for timestamps
from dqn_agent_rgb import DQNAgentRGB
from utils import set_seed

def train_dqn_agent_rgb(trial_number=1, num_episodes=10000, learning_rate=1e-4, batch_size=128,
                        epsilon_decay=0.995, architecture='original', final_run=False):
    """
    Trains a DQN agent on the Flappy Bird Gym environment.

    Args:
        trial_number (int): Identifier for the current trial.
        num_episodes (int): Number of episodes to train.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epsilon_decay (float): Decay rate for epsilon in epsilon-greedy policy.
        architecture (str): Type of neural network architecture ('original' or 'reduced').
        final_run (bool): If True, adjusts logging frequency for the final run.
    """
    # Set random seed for reproducibility
    set_seed(42 + trial_number)

    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    action_dim = env.action_space.n
    agent = DQNAgentRGB(action_dim, learning_rate=learning_rate, epsilon_decay=epsilon_decay,
                        architecture=architecture)
    # Uncomment the line below to load a pre-trained model
    # agent.load_model(best=True, trial_number=trial_number)

    target_update_freq = 10
    beta_start = 0.4
    beta_frames = 10000

    rewards = []
    gates_cleared = []
    durations = []
    flaps_list = []
    best_reward = -float('inf')

    # Logging frequencies
    log_interval = 20  # Log every 20 episodes
    summary_interval = 100  # Summarize every 100 episodes
    save_model_interval = 500  # Save model every 500 episodes

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_gates = 0  # Reset at the start of each episode
        steps = 0
        flaps = 0

        while not done:
            # Uncomment to render the environment
            # env.render()

            action = agent.select_action(state)
            next_state, original_reward, done, info = env.step(action)

            # Count flaps (assuming action '1' corresponds to a flap)
            if action == 1:
                flaps += 1

            # Increment steps
            steps += 1

            # Survival reward
            survival_reward = 1  # Small positive reward per time step

            # Check if a gate was cleared
            current_score = info.get('score', 0)
            if current_score > total_gates:
                gate_reward = 101  # Reward for gate cleared
                total_gates = current_score
            else:
                gate_reward = 0

            # Death penalty
            death_penalty = -100 if done else 0

            # Total reward
            reward = survival_reward + gate_reward + death_penalty

            # Store transition and train agent
            agent.store_transition(state, action, reward, next_state, done)
            beta = min(1.0, beta_start + episode * (1.0 - beta_start) / beta_frames)
            agent.train(batch_size=batch_size, beta=beta)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        gates_cleared.append(total_gates)
        durations.append(steps)
        flaps_list.append(flaps)

        # Detailed logging per episode
        if (episode + 1) % log_interval == 0:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] Episode {episode + 1}: Reward: {total_reward:.2f}, Gates Cleared: {total_gates}, "
                  f"Duration: {steps} steps, Flaps: {flaps}, Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        # Compute moving averages
        if (episode + 1) % summary_interval == 0:
            avg_reward = sum(rewards[-100:]) / len(rewards[-100:])
            avg_gates = sum(gates_cleared[-100:]) / len(gates_cleared[-100:])
            avg_duration = sum(durations[-100:]) / len(durations[-100:])

            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(best=True, trial_number=trial_number)

            # Add timestamp to summary
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] Episode {episode + 1} Summary:")
            print(f"  Average Reward (last 100 episodes): {avg_reward:.2f}")
            print(f"  Average Gates Cleared (last 100 episodes): {avg_gates:.2f}")
            print(f"  Average Duration (last 100 episodes): {avg_duration:.2f} steps")
            print(f"  Epsilon: {agent.epsilon:.3f}")

            # Plotting rewards
            plt.figure(figsize=(10, 5))
            plt.plot(rewards, label="Reward per Episode")
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            plt.title(f"Training Rewards - Trial {trial_number}")
            plt.legend()
            plt.savefig(f"reward_plot_trial_{trial_number}_episode_{episode + 1}.png")
            plt.close()

            # Plotting gates cleared
            plt.figure(figsize=(10, 5))
            plt.plot(gates_cleared, label="Gates Cleared per Episode", color='orange')
            plt.xlabel("Episodes")
            plt.ylabel("Gates Cleared")
            plt.title(f"Gates Cleared Over Episodes - Trial {trial_number}")
            plt.legend()
            plt.savefig(f"gates_plot_trial_{trial_number}_episode_{episode + 1}.png")
            plt.close()

        # Save model at intervals
        if (episode + 1) % save_model_interval == 0:
            agent.save_model(trial_number=trial_number)
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] Model saved at Episode {episode + 1}")

    # Save final model
    agent.save_model(trial_number=trial_number)
    env.close()

    # Save training statistics
    stats = {
        'rewards': rewards,
        'gates_cleared': gates_cleared,
        'durations': durations,
        'flaps': flaps_list
    }
    with open(f'training_stats_trial_{trial_number}.json', 'w') as f:
        json.dump(stats, f)

    # Generate final summary plots
    # Plotting rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"Final Training Rewards - Trial {trial_number}")
    plt.legend()
    plt.savefig(f"final_rewards_trial_{trial_number}.png")
    plt.close()

    # Plotting gates cleared
    plt.figure(figsize=(10, 5))
    plt.plot(gates_cleared, label="Gates Cleared per Episode", color='orange')
    plt.xlabel("Episodes")
    plt.ylabel("Gates Cleared")
    plt.title(f"Final Gates Cleared Over Episodes - Trial {trial_number}")
    plt.legend()
    plt.savefig(f"final_gates_trial_{trial_number}.png")
    plt.close()


if __name__ == '__main__':
    # Run Trial 1 with batch size 128 for 10,000 episodes
    trial_number = 1
    print(f"Starting Trial {trial_number} with Batch Size 128 for 10,000 episodes")
    train_dqn_agent_rgb(
        trial_number=trial_number,
        num_episodes=10000,  # Run for 10,000 episodes
        learning_rate=1e-4,
        batch_size=128,  # Batch size 128
        epsilon_decay=0.995,
        architecture='original',
        final_run=False  # Set to False to maintain logging frequencies
    )
