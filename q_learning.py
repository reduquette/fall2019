import gym
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def q_learning(environment='FrozenLake-v0',
               alpha=1,
               discount_factor=.8,
               epsilon=.1,
               num_training_episodes=10,
               verbose=False):
    env = gym.make(environment)
    num_states = env.observation_space.n  #only works for discrete state spaces
    num_actions = env.action_space.n

    # initialize Q(s,a) randomly-- s x a matrix
    # could make this a numpy dist rather than a matrix -- harder to update
    Q = np.zeros(
        (num_states,
         num_actions)) / num_actions  #assigns equal prob. to all outcomes
    rewards = []
    num_steps = []
    # in an episode:
    #initialize starting state
    for i in range(num_training_episodes):
        current_state = env.reset()
        done = False
        t = 0
        total_rewards = 0
        while not done:
            t += 1
            # env.render()

            if np.random.uniform(0, 1) > 1 - epsilon:
                #randomly according to probability
                action = env.action_space.sample()

                #if choose randomly based on current probabilities rather than uniformly:
                # if all(Q[current_state] == 0):
                #     action = env.action_space.sample()
                # else:
                #     action = np.random.choice(
                #     np.arange(num_actions),
                #     p=normalize(Q[current_state].reshape(1, -1),
                #                 norm='l1').flatten())
            else:  #greedy algorithm
                action = np.argmax(Q[current_state])

            # Take action a, observe r, s'
            next_state, reward, done, info = env.step(action)
            total_rewards += reward
            if verbose:
                print("next state: \t", next_state, "\taction: \t", action,
                      "\treward: \t", reward)

            # update Q and move to next state
            new_q_value = Q[current_state][action] + alpha * (
                reward + discount_factor * max(Q[next_state]) -
                Q[current_state][action])

            Q[current_state][action] = new_q_value
            current_state = next_state
        rewards.append(total_rewards)
        num_steps.append(t)
        # print(Q)

        # print("Episode finished after {} timesteps".format(t + 1))
    env.close()
    return Q, rewards, num_steps


def test_q_function(Q,
                    environment='FrozenLake-v0',
                    verbose=False,
                    num_testing_episodes=1):

    env = gym.make(environment)
    results = []
    for i_episode in range(num_testing_episodes):
        observation = env.reset()
        total_rewards = 0
        for t in range(100):
            env.render()
            action = np.argmax(Q[observation])
            observation, reward, done, info = env.step(action)
            total_rewards += reward
            if verbose:
                print("observation: \t", observation, "\taction: \t", action,
                      "\treward: \t", reward)
            if done:
                if verbose:
                    print("Episode finished after {} timesteps".format(t + 1))
                results.append(
                    [i_episode, total_rewards, observation, reward, t + 1])
                break
    env.close()
    return results


def test_environment(environment='FrozenLake-v0',
                     verbose=False,
                     num_testing_episodes=1):

    env = gym.make(environment)
    results = []
    for i_episode in range(num_testing_episodes):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = np.random.choice(np.arange(env.action_space.n))
            observation, reward, done, info = env.step(action)
            if verbose:
                print("observation: \t", observation, "\taction: \t", action,
                      "\treward: \t", reward)
            if done:
                if verbose:
                    print("Episode finished after {} timesteps".format(t + 1))
                results.append([i_episode, observation, reward, t + 1])
                break
    env.close()
    return results


def vary_alpha():
    vary_alpha_q = []
    vary_alpha = []
    for a in [.1, .5, .75, 1]:
        Q = q_learning(
            alpha=a, discount_factor=.8, epsilon=.1, num_training_episodes=250)
        vary_alpha_q.append(Q)
        results = test_q_function(Q, num_testing_episodes=10)
        vary_alpha.append(results)

    vary_alpha = np.array(vary_alpha)
    vary_alpha_q = np.array(vary_alpha_q)
    print(vary_alpha)
    print(np.mean(vary_alpha[:, :, 2], axis=1))


def main():
    Q, training_rewards, episode_length = q_learning(
        environment='CliffWalking-v0',
        num_training_episodes=500,
        verbose=False)
    print(Q)
    print(training_rewards)
    # results = test_q_function(
    #     Q, environment='CliffWalking-v0', num_testing_episodes=1)
    # episode_reward = results[0][1]
    fig = plt.figure()
    # plt.plot(np.arange(1, 501), training_rewards, c='b', label='Training')
    plt.plot(np.arange(1, 501), episode_length, c='b', label='Timesteps')
    plt.xlabel("Episode")
    # plt.ylabel("Reward")
    plt.ylabel("Number of Timesteps")
    plt.title("Q-Learning Training")
    # plt.plot(np.arange(1, 101), [episode_reward] * 100, c='r', label='Testing')
    # plt.legend()
    plt.show()
    test_q_function(Q, environment='CliffWalking-v0')
    # test_environment(environment='CliffWalking-v0', verbose=True)


main()
