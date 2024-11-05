import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
import flappy_bird_gymnasium


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, *args, **kwargs):
        super(DQN, self).__init__()

        # Define the model
        # It has 2 layers, the first layer has the same number of neurons as the state_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # The second layer has the same number of neurons as the action_dim
        self.output = nn.Linear(hidden_dim, action_dim)


    def forward(self , x):
        # Define the forward pass
        x = F.relu(self.fc1(x))
        return self.output(x)


if __name__ == "__main__":
    # state_dim = 12
    # action_dim = 2
    #
    # dqn = DQN(state_dim, action_dim)
    # # Generate a random state of dimension n x state_dim
    # state = torch.randn(1, state_dim)
    # # Generates Q values for each action
    # output = dqn(state)
    # print(output)

    env = gym.make("FlappyBird-v0")
    obs = env.reset()
    numActions = env.action_space.n
    numState = env.observation_space.shape[0]
    print(numState, numActions)