import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error as mse
from keras.optimizers import SGD


class nn:
    def __init__(self, env):

        input_dim, = env.observation_space.shape  #assumes second value is null
        output_dim = env.action_space.n

        self.model = Sequential()
        self.model.add(Dense(units=50, activation='relu', input_dim=input_dim))
        self.model.add(Dense(units=output_dim, activation='relu'))

        self.model.compile(optimizer=SGD(), loss=mse, metrics=['accuracy'])

    def train_on(self, x, y):
        self.model.fit(x, y, epochs=10)

    def predict(self, x):
        Q_predicted = self.model.predict_on_batch(x)
        return Q_predicted


class dql:
    def __init__(self,
                 environment='MountainCar-v0',
                 learning_rate=.01,
                 discount_factor=.8):
        self.env = gym.make(environment)
        self.q_model = nn(self.env)
        self.experience = []
        self.batch_size = 1000
        self.lr = learning_rate
        self.df = discount_factor

    def policy(self):
        return 1

    def update(self):
        batch_index = np.random.choice(
            len(self.experience), size=self.batch_size)

        state_observations = np.array(
            [self.experience[index][0] for index in batch_index])
        next_states = np.array(
            [self.experience[index][3] for index in batch_index])
        rewards = np.array(
            [self.experience[index][2] for index in batch_index])
        actions = np.array(
            [self.experience[index][1] for index in batch_index])

        q_val_current = self.q_model.predict(state_observations)
        q_val_next = self.q_model.predict(next_states)
        targets = rewards + self.df * np.max(q_val_next, axis=1)

        for i in range(len(q_val_current)):
            q_val_current[i][actions[i]] = targets[i]

        self.q_model.train_on(state_observations, q_val_current)

    def episode(self, num_timesteps=200, adjust_reward=True):
        observation = self.env.reset()
        for t in range(num_timesteps):
            # self.env.render()
            q_values = self.q_model.predict(np.array([observation]))
            if np.all(q_values == 0):
                action = np.random.randint(0, self.env.action_space.n)
            else:
                action = np.argmax(
                    self.q_model.predict(np.array([observation])))
            new_observation, reward, done, info = self.env.step(action)

            if adjust_reward:
                reward = new_observation[0] + 0.5
                if new_observation[0] >= 0.5:
                    reward += 1

            self.experience.append((observation, action, reward,
                                    new_observation))
            observation = new_observation

            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break

    def train(self, num_episodes=200):
        final_positions = []
        final_rewards = []
        for i in range(num_episodes):
            self.episode()
            final_positions.append(self.experience[-1][0][0])
            final_rewards.append(self.experience[-1][2])
            self.update()
        plt.figure()
        plt.plot(np.arange(num_episodes), final_positions)
        plt.title("Learning")
        plt.xlabel("Training Episodes")
        plt.ylabel("Final Position")
        plt.show()

    def plot_policy(self):
        velocities = np.array([
            self.experience[index][0][1]
            for index in range(len(self.experience))
        ])
        positions = np.array([
            self.experience[index][0][0]
            for index in range(len(self.experience))
        ])
        actions = np.array([
            self.experience[index][1] for index in range(len(self.experience))
        ])
        print(actions)
        fig = plt.figure()
        plt.scatter(positions, velocities, c=actions, cmap='viridis')
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        plt.title("Policy")
        plt.savefig("policymap2000.png")
        plt.show()

    def plot_experience(self):
        state_observations = np.array([
            self.experience[index][0] for index in range(len(self.experience))
        ])
        num_observations = np.arange(len(state_observations))
        positions = state_observations[:, 0]
        velocities = state_observations[:, 1]
        fig = plt.figure()
        plt.plot(num_observations, positions)
        plt.ylabel("Position")
        plt.xlabel("Observations")
        plt.title("Position over time")
        plt.savefig("position_v_time_episode.png")
        plt.show()

    def close(self):
        self.env.close()


qt = dql()
qt.episode()
qt.plot_experience()
qt.train(num_episodes=500)
qt.plot_policy()
qt.episode()
qt.close()
