import os
import torch
import gym
import flappy_bird_gym
from dqn_agent_rgb import DQNAgentRGB
from PIL import Image

def evaluate_model_with_render(trial_number=1, num_episodes=100, model_path="./models/best.pth", mirror=False):
    """
    Evaluates a trained DQN model on the Flappy Bird environment with visualization and recording.

    Args:
        trial_number (int): Identifier for the trial.
        num_episodes (int): Number of episodes to evaluate.
        model_path (str): Path to the trained model.
        mirror (bool): If True, mirror the frames horizontally.
    """
    # Create recordings directory if it doesn't exist
    os.makedirs("recordings", exist_ok=True)

    # Initialize environment
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    action_dim = env.action_space.n

    # Initialize agent and load the trained model
    agent = DQNAgentRGB(
        action_dim, epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=1.0
    )
    agent.load_model(
        folder_path=os.path.dirname(model_path), best=True, trial_number=trial_number
    )

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_gates = 0
        steps = 0
        frames = []  # To store frames for GIF creation

        print(f"Starting Episode {episode + 1}...")

        while not done:
            # Render the environment and capture the frame
            frame = env.render(mode="rgb_array")
            # Convert the frame to a PIL Image
            frame_image = Image.fromarray(frame)
            # Rotate the frame 270 degrees counterclockwise (equivalent to 90 degrees clockwise)
            rotated_frame = frame_image.rotate(270, expand=True)
            # Apply horizontal mirroring if mirror=True
            if mirror:
                rotated_frame = rotated_frame.transpose(Image.FLIP_LEFT_RIGHT)
            frames.append(rotated_frame)

            # Select the best action based on the trained policy
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)

            steps += 1
            total_reward += reward
            total_gates = info.get("score", total_gates)

        # Save the GIF with gates cleared in the filename
        gif_path = f"./recordings/episode_{episode + 1}_gates_{total_gates}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=40,  # Adjust duration for speed (lower = faster)
            loop=0,
        )
        print(
            f"Episode {episode + 1} saved as {gif_path}: Reward: {total_reward}, "
            f"Gates Cleared: {total_gates}, Steps: {steps}"
        )

    env.close()

if __name__ == "__main__":
    # Example usage:
    # To run without mirroring:
    # evaluate_model_with_render(trial_number=1, num_episodes=100, model_path="./models/best.pth", mirror=False)
    
    # To run with mirroring:
    # evaluate_model_with_render(trial_number=1, num_episodes=100, model_path="./models/best.pth", mirror=True)
    
    # Adjust the 'mirror' parameter as needed
    evaluate_model_with_render(
        trial_number=1,
        num_episodes=100,
        model_path="./models/best.pth",
        mirror=True  # Set to True or False based on the output
    )
