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
import cv2
import numpy as np
import json
from dqn_agent_rgb import DQNAgentRGB
from utils import set_seed

def train_dqn_agent_rgb(trial_number=1, num_episodes=2000, learning_rate=1e-4, batch_size=256,
                        epsilon_decay=0.995, architecture='original', final_run=False):
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

    # Adjust logging frequency based on whether it's the final run
    if final_run:
        log_interval = 500
        summary_interval = 1000
        save_model_interval = 2000
    else:
        log_interval = 100
        summary_interval = 100
        save_model_interval = num_episodes  # Save only at the end

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_gates = 0
        steps = 0
        flaps = 0

        while not done:
            # Uncomment to render the environment
            # env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Count flaps
            if action == 1:
                flaps += 1

            # Increment steps
            steps += 1

            # Check for passing a pipe (gate cleared)
            if 'score' in info and info['score'] > total_gates:
                total_gates = info['score']

            # Adaptive reward scaling
            reward = 0.1  # Small positive reward for each time step
            if done:
                reward = -1.0  # Negative reward upon death

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
        if (episode + 1) % log_interval == 0 or final_run:
            print(f"Episode {episode + 1}: Reward: {total_reward:.2f}, Gates Cleared: {total_gates}, "
                  f"Duration: {steps} steps, Flaps: {flaps}, Epsilon: {agent.epsilon:.3f}")

        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Compute moving averages
        if (episode + 1) % summary_interval == 0 or final_run:
            avg_reward = sum(rewards[-100:]) / len(rewards[-100:])
            avg_gates = sum(gates_cleared[-100:]) / len(gates_cleared[-100:])
            avg_duration = sum(durations[-100:]) / len(durations[-100:])

            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(best=True, trial_number=trial_number)

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

            # Summary logging
            print(f"Episode {episode + 1} Summary:")
            print(f"  Average Reward (last 100 episodes): {avg_reward:.2f}")
            print(f"  Average Gates Cleared (last 100 episodes): {avg_gates:.2f}")
            print(f"  Average Duration (last 100 episodes): {avg_duration:.2f} steps")
            print(f"  Epsilon: {agent.epsilon:.3f}")

        # Save model at intervals during the final run
        if final_run and (episode + 1) % save_model_interval == 0:
            agent.save_model(trial_number=trial_number)

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

    # Generate final summary plots if it's the final run
    if final_run:
        # Plotting rewards
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label="Reward per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Final 10k Episode Run Rewards - Trial {trial_number}")
        plt.legend()
        plt.savefig(f"final_10k_rewards_trial_{trial_number}.png")
        plt.close()

        # Plotting gates cleared
        plt.figure(figsize=(10, 5))
        plt.plot(gates_cleared, label="Gates Cleared per Episode", color='orange')
        plt.xlabel("Episodes")
        plt.ylabel("Gates Cleared")
        plt.title(f"Final 10k Episode Run Gates Cleared - Trial {trial_number}")
        plt.legend()
        plt.savefig(f"final_10k_gates_trial_{trial_number}.png")
        plt.close()


if __name__ == '__main__':
    # Define trial configurations
    trials = [
        {'trial_number': 1, 'learning_rate': 1e-4, 'batch_size': 256, 'epsilon_decay': 0.995, 'architecture': 'original'},
        {'trial_number': 2, 'learning_rate': 5e-5, 'batch_size': 256, 'epsilon_decay': 0.995, 'architecture': 'reduced'},
        {'trial_number': 3, 'learning_rate': 1e-4, 'batch_size': 512, 'epsilon_decay': 0.990, 'architecture': 'original'},
        #{'trial_number': 4, 'learning_rate': 5e-5, 'batch_size': 128, 'epsilon_decay': 0.990, 'architecture': 'reduced'},
        #{'trial_number': 5, 'learning_rate': 1e-5, 'batch_size': 256, 'epsilon_decay': 0.985, 'architecture': 'original'},
    ]

    # Run shorter trials to find the best hyperparameters
    for trial in trials:
        print(f"Starting Trial {trial['trial_number']}")
        train_dqn_agent_rgb(
            trial_number=trial['trial_number'],
            num_episodes=2000,  # Shorter trials
            learning_rate=trial['learning_rate'],
            batch_size=trial['batch_size'],
            epsilon_decay=trial['epsilon_decay'],
            architecture=trial['architecture'],
            final_run=False
        )

    # After analyzing results, set the best trial number
    # For demonstration, let's assume trial 3 was the best
    best_trial_number = 3
    best_trial = next((trial for trial in trials if trial['trial_number'] == best_trial_number), None)

    if best_trial:
        print(f"Starting Final 10k Episode Run with Trial {best_trial_number}")
        train_dqn_agent_rgb(
            trial_number=best_trial['trial_number'],
            num_episodes=10000,  # Final long run
            learning_rate=best_trial['learning_rate'],
            batch_size=best_trial['batch_size'],
            epsilon_decay=best_trial['epsilon_decay'],
            architecture=best_trial['architecture'],
            final_run=True
        )
    else:
        print("Best trial configuration not found.")