import random

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces.discrete import Discrete
from gym.spaces.tuple import Tuple
from models.policy import FFPolicy
from models.value import FFValue
from util.replay_buffer import ReplayBuffer
from util.rl_board import RLBoard

mse = nn.MSELoss()


def exec_policy(env, policy, seq_len, epsilon):
    states = torch.zeros(seq_len+1, observation_dim(env))
    actions = torch.zeros(seq_len, dtype=torch.long)
    rewards = torch.zeros(seq_len)
    done = torch.zeros(seq_len)

    total_len = None

    observation = env.reset()

    for i in range(seq_len):
        states[i] = state_enc(env, observation)

        action = None
        if random.random() > epsilon:
            act = policy(states[i].reshape((1,-1)))
            action = torch.argmax(act).item()
        else:
            sample = env.action_space.sample()
            action = encode_action(env, sample)
        actions[i] = action
        decoded = decode_action(env, action)
        observation, reward, d, info = env.step(decoded)

        rewards[i] = reward
        done[i] = d
        if d:
            total_len = i
            break

    if total_len is None:
        total_len = seq_len-1
    states[total_len+1] = state_enc(env, observation)

    return states, actions, rewards, done, total_len


def observation_dim(env):
    if isinstance(env.observation_space, Discrete):
        return env.observation_space.n
    else:
        return env.observation_space.shape[0]


def one_hot(dim, n):
    res = torch.zeros(dim)
    res[n] = 1
    return res


def encode_action(env, action):
    if isinstance(env.action_space, Tuple):
        act_code = 0
        k = 1
        for i, a in enumerate(env.action_space):
            act_code += k*action[i]
            k *= a.n
        return act_code
    else:
        return action


def product(ls):
    k = 1
    for l in ls:
        k *= l
    return k


def decode_action(env, act_code):
    if isinstance(env.action_space, Tuple):
        action = []
        num_prods = len(env.action_space)
        k = product([a.n for a in env.action_space[0:-1]])
        for i in range(num_prods):
            act = act_code // k
            action.insert(0, act)
            act_code = act_code % k
            if i < num_prods - 1:
                k = k // env.action_space[num_prods - i - 2].n
        return tuple(action)
    else:
        return act_code


def state_enc(env, observation):
    if isinstance(env.observation_space, Discrete):
        return one_hot(env.observation_space.n, observation)
    else:
        return torch.tensor(observation)

def exec_policy_batch(env, policy, seq_len, batches, epsilon):
    states = torch.zeros(batches, seq_len+1, observation_dim(env))
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
    env = gym.make('Acrobot-v1')

    actions = 1
    if isinstance(env.action_space, Tuple):
        for a in env.action_space:
            actions *= a.n
    else:
        actions = env.action_space.n
    inp = observation_dim(env)
    batch_size = 1
    seq_len = 1000
    lamb = 1
    tau = 1
    epsilon = 1
    e_decay = 0.99

    device = 'cpu'

    its = 10000

    policy = FFPolicy(input_size=inp, actions=actions)
    policy.to(device)
    value = FFValue(input_size=inp)
    value.to(device)

    optim_p = optim.Adam(policy.parameters())
    optim_v = optim.Adam(value.parameters())

    buffer = ReplayBuffer(capacity=500)

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

        epsilon = epsilon*e_decay


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
