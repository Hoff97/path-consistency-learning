import random

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from models.policy import FFPolicy
from models.value import FFValue
from util.replay_buffer import ReplayBuffer
from util.rl_board import RLBoard

mse = nn.MSELoss()

def exec_policy(env, policy, seq_len, epsilon):
    states = torch.zeros(seq_len+1, env.observation_space.shape[0])
    actions = torch.zeros(seq_len, dtype=torch.long)
    rewards = torch.zeros(seq_len)
    done = torch.zeros(seq_len)

    total_len = 0

    observation = env.reset()

    for i in range(seq_len):
        states[i] = torch.tensor(observation)

        action = None
        if random.random() > epsilon:
            act = policy(states[i].reshape((1,-1)))
            action = torch.argmax(act).item()
        else:
            action = env.action_space.sample()
        actions[i] = action
        observation, reward, d, info = env.step(action)

        rewards[i] = reward
        done[i] = d
        if d:
            total_len = i
            break

    states[total_len+1] = torch.tensor(observation)

    return states, actions, rewards, done, total_len


def exec_policy_batch(env, policy, seq_len, batches, epsilon):
    states = torch.zeros(batches, seq_len+1, env.observation_space.shape[0])
    actions = torch.zeros(batches, seq_len, dtype=torch.long)
    rewards = torch.zeros(batches, seq_len)
    done = torch.zeros(batches, seq_len)
    lens = torch.zeros(batches, dtype=torch.long)

    for i in range(batches):
        s, a, r, d, l = exec_policy(env, policy, seq_len, epsilon)
        states[i] = s
        actions[i] = a
        rewards[i] = r
        done[i] = d
        lens[i] = l

    return states, actions, rewards, done, lens


def main():
    env = gym.make('CartPole-v0')

    actions = env.action_space.n
    inp = env.observation_space.shape[0]
    batch_size = 10
    seq_len = 200
    lamb = 1
    tau = 1
    epsilon = 1

    device = 'cpu'

    its = 10000

    policy = FFPolicy(input_size=inp, actions=actions)
    policy.to(device)
    value = FFValue(input_size=inp)
    value.to(device)

    optim_p = optim.Adam(policy.parameters())
    optim_v = optim.Adam(value.parameters())

    buffer = ReplayBuffer(capacity=100)

    board = RLBoard()

    for i in range(its):
        states, actions, rewards, done, lens = exec_policy_batch(env, policy, seq_len, batch_size, epsilon)

        sequence = (states, actions, rewards, done, lens)

        buffer.insert(sequence)

        optim_p.zero_grad()
        optim_v.zero_grad()

        c = consistency(sequence, policy, value, lamb, tau)
        l = consistency_loss(c)
        l.backward()

        optim_p.step()
        optim_v.step()

        sequence = buffer.get_random()

        optim_p.zero_grad()
        optim_v.zero_grad()

        c = consistency(sequence, policy, value, lamb, tau)
        l = consistency_loss(c)
        l.backward()

        optim_p.step()
        optim_v.step()

        board.log(i, rewards, l, lens, epsilon)

        if i%100 == 0:
            print('Saving models')
            save_models(policy, value)

        epsilon = epsilon*0.99


def save_models(policy, value):
    torch.save(policy.state_dict(), 'policy.pth')
    torch.save(value.state_dict(), 'value.pth')

def consistency_loss(consistency):
    return mse(consistency, torch.zeros_like(consistency))

def consistency(sequence, policy, value, lamb, tau):
    states, actions, rewards, done, lens = sequence

    batch_size = actions.shape[0]
    seq_len = actions.shape[1]

    res = torch.zeros(batch_size)
    res = res.to(states.device)

    value_end = torch.zeros(batch_size)
    for i in range(batch_size):
        value_end[i] = value(states[i,lens[i]+1].reshape((1,-1)))
    value_start = value(states[:,0])

    res += ((lamb**lens.float())*value_end).reshape(batch_size)
    res -= value_start.reshape(batch_size)

    for i in range(seq_len):
        act = policy(states[:,i])
        chosen_actions = act.gather(1, actions[:,i].view((-1,1)))
        rew = rewards[:,i]

        res += done[:,i]*(lamb**i)*(rew - tau*torch.log(chosen_actions.reshape(batch_size)))

    return res


main()
