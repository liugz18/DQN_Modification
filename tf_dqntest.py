
import gym
import random
import tensorflow as tf
from tensorflow.keras import optimizers#layers,Sequential,

from keras import layers, Sequential
from keras.layers import Dense
from collections import deque

from keras.optimizers import adam
import matplotlib.pyplot as plt

import numpy as np
env = gym.make('MountainCar-v0')
#env.seed(0)
#np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
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

        '''Y'''
        self.Y_iter = 0
        self.Y_capacity = 4
        self.Y_replacy_iter = 10


        for i in range(self.Y_capacity):
            self.target_model.save(str(i) + '.h5')


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


    def Oracle(self, x):
        QSAs = np.zeros((self.Y_capacity, self.action_space))
        for i in range(self.Y_capacity):
            temp_model = self.build_model_without_backward()
            temp_model.load_weights(str(i) + '.h5')
            QSAs[i] = temp_model.predict(x)
            #temp_model.__del__()
        argmaxAction = np.ptp(QSAs,axis=0)
        #print(QSAs)
        #print(argmaxAction)
        action = np.argmax(argmaxAction)
        #print(action)
        return action
    
        #how to get a better Q?
        #suppose the input model is model_input
        #For a in Actions: 
        #   model_actor[a] = model_input.copy()
        #   model_actor[a].perform(gradident accent for 100 steps)
        #best_a = argmax([model_actor[a].Qvalue_of(a) for a in Actions])
        

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return self.Oracle(state)
            #return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

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

        targets = rewards + self.gamma*(np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
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





        if self.target_iter % self.Y_replacy_iter==0:
            self.model.save(str(self.Y_iter) + '.h5')
            self.Y_iter += 1
            if self.Y_iter % self.Y_capacity == 0:
                self.Y_iter = 0




def train_dqn(episode):

    loss = []
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
    return loss

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

    ep = 400

    for i in range(3):
        loss = train_dqn(ep)
        np.save('second_oracle' + str(i+2)+'.npy', loss)
        plt.plot([i + 1 for i in range(0, ep, 2)], loss[::2])
        plt.show()

