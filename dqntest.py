'''
    这是DQN系算法的调试程序

'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.0001  # learning rate
EPSILON = 0.7  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency

#  增大MEMORY_CAPACITY 可以减少突然很低分的出现
MEMORY_CAPACITY = 5000
env = gym.make('MountainCarContinuous-v0')
env = env.unwrapped
print(env.observation_space)
# print(env.action_space.high)
# print(env.action_space.low)

N_ACTIONS = 3#env.action_space.n
N_STATES = env.observation_space.shape[0]


'''策略二维可视化'''
def Print_Policy(dqn, num):
    X = np.random.uniform(-1.2, 0.6, 10000)
    Y = np.random.uniform(-0.07, 0.07, 10000)
    Z = []
    EPSILON = 1
    for i in range(len(X)):
        temp = dqn.choose_action(np.array([X[i],Y[i]]).astype(np.float64))
        z = temp
        Z.append(z)
    Z = pd.Series(Z)
    colors = {0:'blue',1:'lime',2:'red'}
    colors = Z.apply(lambda x:colors[x])
    labels = ['Left','Right','Nothing']


    from matplotlib.colors import ListedColormap
    fig = plt.figure(3, figsize=[7,7])
    ax = fig.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X,Y, c=Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    recs = []
    colors = ['blue','lime','red']
    for i in range(0,3):
         print(i)
         #sorted(colors)[i]
         recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs,labels,loc=4,ncol=3)
    fig.savefig('PolicyAt' + num + '.png')
    plt.show()

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 45)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # self.fc2 = nn.Linear(10,80)
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(45, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


'''添加dueling dqn'''
class DuelingNet(nn.Module):
    def __init__(self, ):
        super(DuelingNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 45)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # self.fc2 = nn.Linear(10,80)
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(45, N_ACTIONS + 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        actions_value = self.out(x)
        Vs = actions_value[:,0]


        actions_advantage = actions_value[:,1:N_ACTIONS+1]
        actions_advantage_average = torch.mean(actions_advantage,dim=1,keepdim=True)

        actions_advantage -= actions_advantage_average

        actions_advantage += Vs.view(-1,1)


        return actions_advantage


class DQN(object):
    def __init__(self):

        #Dueling DQN
        self.eval_net = Net()#DuelingNet()##
        self.target_net = Net()##DuelingNet()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))  # initailize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = int(torch.max(actions_value, 1)[1].data.numpy())  # [0,0]
            # print(action)
        else:
            action = int(np.random.randint(0, N_ACTIONS))
        return action

    def store_transition(self, s, a, r, done, s_):
        transition = np.hstack((s, [a, done, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_done = Variable(torch.LongTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 2:N_STATES + 3]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        # print('!!!!!')
        # #print(torch.t(b_a).size())
        # #b_a = torch.t(b_a)
        # print(b_a)
        # print('???')
        # print(b_s)
        q_eval = self.eval_net(b_s)  # .gather(1,b_a)

        # print(q_eval)
        q_eval = torch.gather(q_eval, 1, b_a)

        '''DDQN'''
        actions_value = self.eval_net(b_s_).detach()
        action_ = actions_value.max(1)[1]

        q_next = self.target_net(b_s_).detach()
        #q_target = GAMMA * q_next.max(1)[0]


        '''DDQN'''
        q_target = GAMMA * q_next.gather(1, action_.view(-1, 1))



        q_target = b_done * q_target.view(-1, 1)

        q_target += b_r
        #print(q_target)
        # print(b_done.detach().shape)
        # q_target = b_r + GAMMA * q_next.gather(1,action_)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
# torch.save(dqn.state_dict(),'01.pkl')
step = 0
score_list = []

print('\nCollecting experience...')
for i_episode in range(200):
    s = env.reset()
    score = 0
    # for i in range(2000):
    while True:
        env.render()

        a = dqn.choose_action(s)

        a_number = 0.0
        '''插值'''
        if a==0:
            a_number = -1.0
        elif a==1:
            a_number = 0.0
        else:
            a_number = 1.0
        # take action

        s_, r, done, info = env.step([a_number])
        # print(r)
        # modify the reward
        # x,x_dot,theta,theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2
        pos, velo = s_

        '''Modified Reward'''
        score += r
        #pos += 0.5
        r = abs(velo)

        if velo > 0:
            r *= 2

        #r += 2 * pos
        if pos > 0.4:
            r += 10 * pos


        if done:
            r+=3

        # r += abs(10*pos) ** 3
        #
        # if(pos>0):
        #     r += 5 * int(pos/0.1)
        # else:
        #     r *= 0.1
        #
        # print(r)

        # print(s)
        dqn.store_transition(s, a, r, ~done, s_)
        # if dqn.memory_counter > MEMORY_CAPACITY:
        # dqn.learn()

        if done:
            score_list.append(score)
            break

            # env.reset()
        s = s_
        step += 1

        if dqn.memory_counter == MEMORY_CAPACITY:
            print('Start Training!!!')
        if (dqn.memory_counter > MEMORY_CAPACITY):  # and (step%5 == 0):
            dqn.learn()

    if i_episode % 10 == 0:
        Print_Policy(dqn, str(i_episode))

    if (EPSILON < 0.9):
        EPSILON += 0.05
        # step = 0
    print('episode:', i_episode, 'score:', score)
# if np.mean(score_list[-10:]) > -400:
# dqn.save_model()
# break

# torch.save(dqn.state_dict(),'01.pkl')
Print_Policy(dqn, 'Final')

fig = plt.figure(3, figsize=[7,7])
ax = fig.gca()
ax.set_xlabel('Episode')
ax.set_ylabel('Score')
ax.set_title('Score-Episode')
plt.plot(score_list, color='green')
plt.axhline(y=-200,ls="--",c="red",label='Score=-200')#添加水平直线
plt.savefig('02.png')
plt.show()


#打印policy

# X = np.random.uniform(-1.2, 0.6, 10000)
# Y = np.random.uniform(-0.07, 0.07, 10000)
# Z = []
# EPSILON = 1
# for i in range(len(X)):
#     temp = dqn.choose_action(np.array([X[i],Y[i]]).astype(np.float64))
#     z = temp
#     Z.append(z)
# Z = pd.Series(Z)
# colors = {0:'blue',1:'lime',2:'red'}
# colors = Z.apply(lambda x:colors[x])
# labels = ['Left','Right','Nothing']
#
#
# from matplotlib.colors import ListedColormap
# fig = plt.figure(3, figsize=[7,7])
# ax = fig.gca()
# plt.set_cmap('brg')
# surf = ax.scatter(X,Y, c=Z)
# ax.set_xlabel('Position')
# ax.set_ylabel('Velocity')
# ax.set_title('Policy')
# recs = []
# colors = ['blue','lime','red']
# for i in range(0,3):
#      print(i)
#      #sorted(colors)[i]
#      recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
# plt.legend(recs,labels,loc=4,ncol=3)
# fig.savefig('Policy.png')
# plt.show()


#env.destroy()
# dqn.learn()