import gym
import numpy as np
import matplotlib.pyplot as plt


def test_environment(environment='FrozenLake-v0',
                     verbose=False,
                     num_testing_episodes=1):

    env = gym.make(environment)
    print(env.observation_space.shape)
    print(env.action_space)
    results = []
    experience = []
    for i_episode in range(num_testing_episodes):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = np.random.choice(
                np.arange(env.action_space.n))  #making random choices
            new_observation, reward, done, info = env.step(action)
            experience.append([observation, action, reward, new_observation])
            observation = new_observation
            if verbose:
                print("observation: \t", observation, "\taction: \t", action,
                      "\treward: \t", reward)
            if done:
                if verbose:
                    print("Episode finished after {} timesteps".format(t + 1))
                results.append([i_episode, observation, reward, t + 1])
                break
    env.close()
    print(experience)
    return results


test_environment(environment='MountainCar-v0', verbose=True)
