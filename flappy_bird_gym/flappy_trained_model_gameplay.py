import flappy_bird_gym
from flappy_bird_gym.flappy_dqn_agent import DQNAgent, load_model


def play_with_trained_model():
    env = flappy_bird_gym.make("FlappyBird-v0")
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=action_dim)

    # Load the trained model
    load_model(agent, "dqn_positional_agent.pkl")

    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)  # Use the policy network to select actions
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    print(f"Total Reward: {total_reward}")
    env.close()

play_with_trained_model()
