import gym
import random
import tensorflow as tf
from tensorflow.keras import optimizers  # layers,Sequential,

from keras import layers, Sequential
from keras.layers import Dense
from collections import deque

from keras.optimizers import adam
import matplotlib.pyplot as plt

import numpy as np

env = gym.make('MountainCar-v0')
env = env.unwrapped

Oracle_ENABLE = True

# env.seed(0)
# np.random.seed(0)


class DQN:
    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0 #0.5
        self.gamma = .95
        self.reg = 0.1
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .986
        self.learning_rate = 0.001
        self.target_replace_iter = 50
        self.target_iter = 0
        self.memory = deque(maxlen=10000)
        self.model = self.build_model()
        self.target_model = self.build_model_without_backward()
        self.model.save("my_h5_model.h5")
        self.target_model.load_weights("my_h5_model.h5")

    def build_model_without_backward(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        return model

    def build_model(self):

        model = self.build_model_without_backward()
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def Oracle_Loss(self, Qsa1, Qsa2):

        return - (Qsa2 - Qsa1) + self.reg * (Qsa1 - Qsa2) ** 2

    def Oracle(self, x):
        # how to get a better Q?
        # suppose the input model is model_input
        # For a in Actions:
        #   model_actor[a] = model_input.copy()
        #   model_actor[a].perform(gradident accent for 100 steps)
        # best_a = argmax([model_actor[a].Qvalue_of(a) for a in Actions])
        self.model.save_weights("Temp_model_save.h5")

        model_temp = self.build_model_without_backward()
        # weights = [1 if i == a else 0 for i in range(self.action_space)]
        model_temp.compile(loss=self.Oracle_Loss, optimizer=adam(lr=self.learning_rate))
        model_temp.load_weights("Temp_model_save.h5")
        model_temp.fit(x, self.model.predict(x), epochs=5, verbose=False)
        #uncertainties = (model_temp.predict(x) + np.abs(model_temp.predict(x) - self.model.predict(x)))
        uncertainties = (self.model.predict(x) + np.abs(model_temp.predict(x) - self.model.predict(x)))

        action = np.argmax(np.array(uncertainties))
        # print('Oracle Output action: ', action)
        return action

    def act(self, state, mode='train'):
        global Oracle_ENABLE
        if mode == 'test':
            act_values = self.model.predict(state)
            # print(act_values)
            return np.argmax(act_values)

        if np.random.rand() <= self.epsilon:
            if Oracle_ENABLE:
                return self.Oracle(state)
            else:
                return int(np.random.randint(0, self.action_space))#random.randrange(self.action_space)
        act_values = self.model.predict(state)
        # print(act_values)
        return np.argmax(act_values)

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (
                    1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # self.target_iter += 1
        # if self.target_iter % self.target_replace_iter == 0:
        #     self.model.save("my_h5_model.h5")
        #     #reconstructed_model = keras.models.load_model("my_h5_model.h5")
        #     self.target_model.save(str(self.Y_iter) + '.h5')
        #     self.target_model.load_weights("my_h5_model.h5")
        #     self.target_iter = 0
        #     self.Y_iter += 1
        #     if self.Y_iter % self.Y_capacity == 0:
        #         self.Y_iter = 0

        self.target_iter += 1
        if self.target_iter % self.target_replace_iter == 0:
            self.model.save("my_h5_model.h5")
            # reconstructed_model = keras.models.load_model("my_h5_model.h5")
            self.target_model.load_weights("my_h5_model.h5")
            self.target_iter = 0


def test_dqn(agent):
    loss = []

    for e in range(50):
        state = env.reset()
        state_shape = state.shape[0]
        state = np.reshape(state, (1, state_shape))
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            # if e >= int(0.95*episode):
            # env.render()
            action = agent.act(state, mode='test')
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_shape))
            # agent.remember(state, action, reward, next_state, done)
            state = next_state
            # agent.replay()
            if done:
                # print("episode: {}/{}, score: {}".format(e, episode, score))
                break

        loss.append(score)
    #print('test loss ', ' :', loss)
    return loss#np.mean(np.array(loss))


def train_dqn(episode):
    loss = []
    test_loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state_shape = state.shape[0]
        state = np.reshape(state, (1, state_shape))
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            # if e >= int(0.95*episode):
            #     env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_shape))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
        if e % 39 == 4 and e != 4:
            test_loss.append(test_dqn(agent))
            # print('test loss on episode ', episode, ' :', test_loss)

    return loss, test_loss


#
# def random_policy(episode, step):
#
#     for i_episode in range(episode):
#         env.reset()
#         for t in range(step):
#             env.render()
#             action = env.action_space.sample()
#             state, reward, done, info = env.step(action)
#             if done:
#                 print("Episode finished after {} timesteps".format(t+1))
#                 break
#             print("Starting next episode")
#

if __name__ == '__main__':
    ep = 200



    for i in range(20):
        env.seed(i)
        np.random.seed(400-i)
        loss, test_loss = train_dqn(ep)
        test_loss = np.array(test_loss)
        np.save('third_oracle_loss' + str(i)+'.npy', loss)
        np.save('third_oracle_test_loss' + str(i)+'.npy', test_loss)
        plt.plot([i + 1 for i in range(ep)], loss)
        plt.show()

    Oracle_ENABLE = False
    for i in range(10):
        env.seed(i)
        np.random.seed(400-i)
        loss, test_loss = train_dqn(ep)
        test_loss = np.array(test_loss)
        np.save('third_random_loss' + str(i)+'.npy', loss)
        np.save('third_random_test_loss' + str(i)+'.npy', test_loss)
        plt.plot([i + 1 for i in range(ep)], loss)
        plt.show()

    # plt.plot([i + 1 for i in range(int(ep/50))], test_loss)
    # plt.show()
