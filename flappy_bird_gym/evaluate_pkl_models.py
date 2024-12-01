import os
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
import flappy_bird_gym

import dill

def load_pkl_model(file_path):
    try:
        # Monkey-patch numpy._core if necessary
        sys.modules['numpy._core'] = numpy.core

        with open(file_path, 'rb') as f:
            model = dill.load(f)
        return model
    except Exception as e:
        print(f"Failed to load model {file_path}: {e}")
        return None


# Function to evaluate the .pkl model
def evaluate_pkl_model(model, env, num_episodes=500):
    """
    Evaluate a .pkl model over multiple episodes.
    Args:
        model: The loaded model.
        env: The environment instance.
        num_episodes: Number of episodes to evaluate.
    Returns:
        rewards: List of total rewards for each episode.
        gates_cleared: List of gates cleared for each episode.
    """
    rewards = []
    gates_cleared = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_gates = 0

        while not done:
            try:
                action = model(state)  # Assuming .pkl model is callable
                state, reward, done, info = env.step(action)
                total_reward += reward
                total_gates = info.get("score", total_gates)
            except Exception as e:
                print(f"Error during evaluation in episode {episode}: {e}")
                break

        rewards.append(total_reward)
        gates_cleared.append(total_gates)

    return rewards, gates_cleared

# Function to generate plots
def plot_results(rewards, gates_cleared, model_name, output_folder, num_episodes):
    """
    Generate and save plots for rewards and gates cleared.
    Args:
        rewards: List of rewards.
        gates_cleared: List of gates cleared.
        model_name: Name of the model.
        output_folder: Directory to save plots.
        num_episodes: Number of episodes evaluated.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Rewards line plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Performance: {model_name} (Episodes: {num_episodes})")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"{model_name}_rewards.png"))
    plt.close()

    # Gates cleared line plot
    plt.figure(figsize=(10, 5))
    plt.plot(gates_cleared, label="Gates Cleared", color="orange")
    plt.xlabel("Episodes")
    plt.ylabel("Gates Cleared")
    plt.title(f"Gates Cleared: {model_name} (Episodes: {num_episodes})")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"{model_name}_gates.png"))
    plt.close()

    # Rewards histogram
    plt.figure(figsize=(10, 5))
    plt.hist(rewards, bins=50, alpha=0.75, edgecolor='black')
    plt.xlabel("Reward")
    plt.ylabel("Number of Trials")
    plt.title(f"Histogram of Rewards: {model_name} (Episodes: {num_episodes})")
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_folder, f"{model_name}_rewards_histogram.png"))
    plt.close()

    # Gates cleared histogram
    plt.figure(figsize=(10, 5))
    plt.hist(gates_cleared, bins=range(0, max(gates_cleared) + 2), alpha=0.75, edgecolor='black')
    plt.xlabel("Gates Cleared")
    plt.ylabel("Number of Trials")
    plt.title(f"Histogram of Gates Cleared: {model_name} (Episodes: {num_episodes})")
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_folder, f"{model_name}_gates_histogram.png"))
    plt.close()

# Main function for evaluating all .pkl models
def evaluate_all_pkl_models(folder_path="final_models", output_folder="output_results", num_episodes=500):
    """
    Evaluate all .pkl models in a folder and generate performance graphs.
    """
    os.makedirs(output_folder, exist_ok=True)
    env = flappy_bird_gym.make("FlappyBird-v0")

    for model_file in os.listdir(folder_path):
        if not model_file.endswith('.pkl'):
            continue

        model_path = os.path.join(folder_path, model_file)
        print(f"Evaluating model: {model_file}")

        model = load_pkl_model(model_path)
        if model is None:
            print(f"Failed to load model {model_file}")
            continue

        try:
            rewards, gates_cleared = evaluate_pkl_model(model, env, num_episodes)
            plot_results(rewards, gates_cleared, os.path.splitext(model_file)[0], output_folder, num_episodes)
            print(f"Results for {model_file} saved to {output_folder}.")
        except Exception as e:
            print(f"Error during evaluation of {model_file}: {e}")

    env.close()

if __name__ == "__main__":
    evaluate_all_pkl_models(folder_path="final_models", output_folder="output_results", num_episodes=500)
