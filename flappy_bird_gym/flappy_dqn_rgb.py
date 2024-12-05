import gym
import flappy_bird_gym
import matplotlib.pyplot as plt
from dqn_agent_rgb import DQNAgentRGB


def train_dqn_agent_rgb(isPlot=False, isTraining=True):
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    action_dim = env.action_space.n
    agent = DQNAgentRGB(action_dim)
    agent.load_model(best=True)

    num_episodes = 10000
    target_update_freq = 10
    batch_size = 512

    rewards = []
    best_reward = -float('inf')

    for episode in range(num_episodes):
        print(f'Episode : {episode + 1}')
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if not isTraining:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            clipped_reward = max(min(reward, 1.0), -1.0)

            if isTraining:
                agent.store_transition(state, action, clipped_reward, next_state, done)
                agent.train(batch_size)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if isTraining:
            if episode % target_update_freq == 0:
                agent.update_target_network()

            avg_reward = sum(rewards[-100:]) / len(rewards[-100:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(best=True)

            if (episode + 1) % 100 == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(rewards, label="Reward per Episode")
                plt.xlabel("Episodes")
                plt.ylabel("Reward")
                plt.title("Training Rewards")
                plt.legend()
                plt.savefig(f"reward_plot_{episode + 1}.png")
                plt.close()
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    if isTraining:
        agent.save_model()
    env.close()



if __name__ == '__main__':
    train_dqn_agent_rgb(True, True)