import time

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from models.policy import FFPolicy
from models.value import FFValue
from util.replay_buffer import ReplayBuffer
from util.rl_board import RLBoard

mse = nn.MSELoss()

def exec_policy(env, policy):
    observation = env.reset()

    for i in range(100000):
        env.render()

        state = torch.tensor(observation).float()

        act = policy(state.reshape((1,-1)))

        action = torch.argmax(act)
        observation, _, d, _ = env.step(action.item())

        if d:
            observation = env.reset()


def main():
    env = gym.make('CartPole-v0')

    actions = env.action_space.n
    inp = env.observation_space.shape[0]

    policy = FFPolicy(input_size=inp, actions=actions)
    policy.load_state_dict(torch.load('res/models/cart_pole/policy.pth'))
    policy = policy.eval()

    exec_policy(env, policy)


main()
