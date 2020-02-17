import gym

env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())

env.close()
