
import argparse
import time
import original_game
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

sys.path.insert(0, os.path.abspath('../flappy_bird_gym'))
print(os.path.abspath('./flappy_bird_gym'))

import flappy_bird_gym
import gym
from flappy_dqn_agent import train_dqn_agent
from flappy_dqn_rgb import train_dqn_agent_rgb


def _get_args():
    """ Parses the command line arguments and returns them. """
    parser = argparse.ArgumentParser(description=__doc__)
    # Argument for the mode of execution (human, random, dqn, or dqn_rgb):
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="random",
        choices=["human", "random", "dqn", "dqn_rgb"],
        help="The execution mode for the game.",
    )

    return parser.parse_args()


def random_agent_env():
    env = gym.make("FlappyBird-v0")
    env.reset()
    score = 0
    while True:
        # env.render()

        # Getting random action:
        action = env.action_space.sample()

        # Processing:
        obs, reward, done, _ = env.step(action)

        score += reward
        print(f"Obs: {obs}\n"
              f"Action: {action}\n"
              f"Score: {score}\n")

        time.sleep(1 / 30)

        if done:
            # env.render()
            time.sleep(0.5)
            break


def main():
    args = _get_args()
    if args.mode == "human":
        original_game.main()
    elif args.mode == "random":
        random_agent_env()
    elif args.mode == "dqn":
        train_dqn_agent(resume_training=True)
    elif args.mode == "dqn_rgb":
        train_dqn_agent_rgb()
    else:
        print("Invalid mode!")


if __name__ == '__main__':
    main()