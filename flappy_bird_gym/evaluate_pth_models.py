import os
import torch
import pickle
import gym
import flappy_bird_gym
import matplotlib.pyplot as plt
from dqn_agent_rgb import DQNAgentRGB

# Function to load models dynamically
def load_model(file_path, action_dim, device):
    """
    Load a model based on file extension (.pth for RGB models, .pkl for positional models).
    """
    if file_path.endswith('.pth'):
        agent = DQNAgentRGB(action_dim=action_dim, epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=1.0)
        agent.policy_net.load_state_dict(torch.load(file_path, map_location=device))
        agent.policy_net.eval()
        return agent
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model  # Assuming the .pkl model is usable directly in evaluation
    else:
        raise ValueError(f"Unsupported model file extension for: {file_path}")

# Evaluation function
def evaluate_model_stats(model, env, num_episodes=500, is_rgb_model=True):
    """
    Evaluate the model over multiple episodes and return stats.
    """
    rewards = []
    gates_cleared = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_gates = 0

        while not done:
            if is_rgb_model:
                action = model.select_action(state)  # RGB model expects a select_action function
            else:
                action = model(state)  # Positional model assumes a callable model

            state, reward, done, info = env.step(action)
            total_reward += reward
            total_gates = info.get("score", total_gates)

        rewards.append(total_reward)
        gates_cleared.append(total_gates)

    return rewards, gates_cleared

# Main evaluation script
def evaluate_all_models(folder_path="final_models", output_folder="positional_results", num_episodes=500):
    """
    Evaluate all models in the specified folder and generate performance graphs.
    """
    os.makedirs(output_folder, exist_ok=True)
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_file in os.listdir(folder_path):
        model_path = os.path.join(folder_path, model_file)
        print(f"Evaluating model: {model_file}")

        is_rgb_model = model_file.endswith('.pth')

        try:
            model = load_model(model_path, action_dim, device)
            rewards, gates_cleared = evaluate_model_stats(model, env, num_episodes, is_rgb_model)

            # Plot rewards
            plt.figure(figsize=(10, 5))
            plt.plot(rewards, label=f"Rewards")
            plt.xlabel("Episodes")
            plt.ylabel("Total Reward")
            plt.title(f"Performance: {os.path.splitext(model_file)[0]} (Episodes: {num_episodes})")
            plt.legend()
            plt.savefig(os.path.join(output_folder, f"{os.path.splitext(model_file)[0]}_rewards.png"))
            plt.close()

            # Plot gates cleared
            plt.figure(figsize=(10, 5))
            plt.plot(gates_cleared, label=f"Gates Cleared", color="orange")
            plt.xlabel("Episodes")
            plt.ylabel("Gates Cleared")
            plt.title(f"Gates Cleared: {os.path.splitext(model_file)[0]} (Episodes: {num_episodes})")
            plt.legend()
            plt.savefig(os.path.join(output_folder, f"{os.path.splitext(model_file)[0]}_gates.png"))
            plt.close()

            # Histogram for rewards
            plt.figure(figsize=(10, 5))
            plt.hist(rewards, bins=50, alpha=0.75, edgecolor='black')
            plt.xlabel("Reward")
            plt.ylabel("Number of Trials")
            plt.title(f"Histogram of Rewards: {os.path.splitext(model_file)[0]} (Episodes: {num_episodes})")
            plt.grid(axis='y')
            plt.savefig(os.path.join(output_folder, f"{os.path.splitext(model_file)[0]}_rewards_histogram.png"))
            plt.close()

            # Histogram for gates cleared
            plt.figure(figsize=(10, 5))
            plt.hist(gates_cleared, bins=range(0, max(gates_cleared) + 2), alpha=0.75, edgecolor='black')
            plt.xlabel("Gates Cleared")
            plt.ylabel("Number of Trials")
            plt.title(f"Histogram of Gates Cleared: {os.path.splitext(model_file)[0]} (Episodes: {num_episodes})")
            plt.grid(axis='y')
            plt.savefig(os.path.join(output_folder, f"{os.path.splitext(model_file)[0]}_gates_histogram.png"))
            plt.close()

            print(f"Model {model_file} evaluated successfully. Results saved to {output_folder}.")

        except Exception as e:
            print(f"Failed to evaluate model {model_file}: {e}")

    env.close()

if __name__ == "__main__":
    evaluate_all_models(folder_path="final_models", output_folder="positional_results", num_episodes=500)
