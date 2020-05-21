'''
    这是DQN系算法的调试程序

'''
import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import random
import time
#from Prioritized_exp import Memory

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.001  # learning rate
EPSILON = 1  # greedy policy
EPSILON_min = 0.01
EPSILON_decay = 0.995
GAMMA = 0.95  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency

#  增大MEMORY_CAPACITY 可以减少突然很低分的出现
MEMORY_CAPACITY = 10000
LR_decay_steps = 100
LR_decay_gamma = 0.5
#env = gym.make('MountainCarContinuous-v0')
env = gym.make('MountainCar-v0')
#env = env.unwrapped
print(env.observation_space)
# print(env.action_space.high)
# print(env.action_space.low)

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
print(N_STATES)


# import os
# from google.colab import drive
# drive.mount('/content/drive')
#
# path = "/content/drive/My Drive"
#
# os.chdir(path)
# os.listdir(path)


def ReInitializeHPs():
    global LR,EPSILON,EPSILON_min

    LR = 0.001  # learning rate
    EPSILON = 1  # greedy policy








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
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 24)
        self.fc1.weight.data.normal_(0, 1)  # initialization
        self.fc2 = nn.Linear(24,24)
        self.fc2.weight.data.normal_(0, 1)  # initialization
        self.out = nn.Linear(24, N_ACTIONS)
        self.out.weight.data.normal_(0, 1)  # initialization

    def forward(self, x):
        x = x.cuda()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value




class DQN():
    def __init__(self):

        self.eval_net = Net()
        self.target_net = Net()
        self.target_net.requires_grad = False
        if torch.cuda.is_available():
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.learn_step_counter = 0  # for target updating
        '''Prioritized Exp Replay'''
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))  # initailize memory
        # s s' a r done
        #self.memory = Memory(MEMORY_CAPACITY)
        self.reg = 0.1
        self.Oracle_GD_Times = 5

        # ailize DQN memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.LR_reducer = StepLR(self.optimizer, LR_decay_steps, LR_decay_gamma)
        '''Prioritized Exp Replay'''
        #self.loss_func = WeightedMSE#lambda y_true, y_pred, weight: (y_true - y_pred)**2.mm()
        self.loss_func = nn.MSELoss()

    def Oracle_Loss(self, Qsa1, Qsa2):
        return - (Qsa2 - Qsa1) + self.reg * (Qsa1 - Qsa2) ** 2

    def ucb(self, x, a, a_init_value):
        Temp_Net = Net().cuda()
        Temp_Net.load_state_dict(self.eval_net.state_dict())
        Temp_optimizer = torch.optim.SGD(Temp_Net.parameters(), lr=0.004)# LR change?

        for i in range(self.Oracle_GD_Times):
            Temp_optimizer.zero_grad()
            a_pred = Temp_Net(x).squeeze()
            #print(a_pred)
            a_pred = a_pred[a]
            #print(a_pred)
            loss = self.Oracle_Loss(a_pred, a_init_value)
            #print(loss)
            loss.backward()
            Temp_optimizer.step()
        with torch.no_grad():
            ucb = Temp_Net(x).squeeze()[a].detach()

        return ucb


    def Oracle(self, x, actions_value):
        ucbs = np.zeros(N_ACTIONS)



        for action in range(N_ACTIONS):
            #ucbi = self.ucb(x, action, actions_value.squeeze()[action]).cpu().numpy()

            ucbs[action] = self.ucb(x, action, actions_value.squeeze()[action]).cpu().numpy()

        return np.argmax(ucbs)
        #2020-5-22: wrong ucb
        # QSAs = np.zeros((Y_CAPACITY, N_ACTIONS))
        # for i, model in enumerate(self.Ys):
        #     temp_model = Net()
        #     temp_model.load_state_dict(model)
        #     QSAs[i] = temp_model.forward(x).detach().numpy()
        #     #temp_model.__del__()
        # argmaxAction = np.ptp(QSAs,axis=0)
        # #print(QSAs)
        # #print(argmaxAction)
        # action = np.argmax(argmaxAction)
        # #print(action)
        # return action



    def choose_action(self, x):
        #self.eval_net.eval()
        x = torch.unsqueeze(torch.FloatTensor(x), 0)#, requires_grad = False

        with torch.no_grad():
            actions_value = self.eval_net.forward(x).detach()

        if np.random.uniform() > EPSILON:  # greedy

            #action = np.max(actions_value.cpu().numpy())
            #print(actions_value)
            action = torch.max(actions_value, 1)[1].cpu().numpy()# [0,0]
            #print(action)
        else:
            #action = np.random.randint(0, N_ACTIONS)#random.randrange(N_ACTIONS)#
            action = self.Oracle(x, actions_value)
            print('Oracle output action: ', action)

        action = int(action)

        return action

    def store_transition(self, s, a, r, done, s_):
        transition = np.hstack((s, [a, done, r], s_))
        # replace the old memory with new memory
        '''Prioritized Exp Replay'''
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        #self.memory.store(transition)



    def learn(self):
        # target net update

        #self.eval_net.train()
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # '''ADD  Y'''
            # self.Ys[self.Y_counter % Y_CAPACITY] = copy.deepcopy(self.eval_net.state_dict())
            # self.Y_counter += 1
            self.target_net.load_state_dict(self.eval_net.state_dict())

        '''Prioritized Exp Replay'''
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index, :]
        #tree_idx, b_memory, ISWeights = self.memory.sample(BATCH_SIZE)

        b_s = torch.cuda.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.cuda.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_done = torch.cuda.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_r = torch.cuda.FloatTensor(b_memory[:, N_STATES + 2:N_STATES + 3])
        b_s_ = torch.cuda.FloatTensor(b_memory[:, -N_STATES:])



        self.optimizer.zero_grad()
        q_eval = self.eval_net.forward(b_s)  # .gather(1,b_a)

        q_eval = torch.gather(q_eval, 1, b_a)

        '''DDQN'''
        # actions_value = self.eval_net(b_s_).detach()
        # action_ = actions_value.max(1)[1]
        #
        with torch.no_grad():
            q_target = self.target_net(b_s_).detach()


        #)#, q_target.size())
        q_target = b_r.squeeze() + GAMMA * torch.mul(torch.max(q_target, 1)[0], 1-b_done.squeeze())#.reshape((-1,1))
        q_target = q_target.view((BATCH_SIZE, 1))
        #print(q_target, q_target.size())
        #print(b_done, b_done.size())


        #q_eval_temp = self.eval_net(b_s).detach()
        # b_a = b_a.reshape(BATCH_SIZE).numpy()
        # q_eval_temp[:, b_a] = q_target
        # print('q_eval',q_eval,q_eval.shape)
        #
        # print('q_target',q_target,q_target.shape)
        # print('br',b_r,b_r.shape)

        #
        '''DDQN'''
        # q_target = GAMMA * q_next.gather(1, action_.view(-1, 1))
        # #
        # #
        # #
        # q_target = b_done * q_target.view(-1, 1)
        #print(q_target)
        #print(b_r)

        #print(q_target)
        # print(b_done.detach().shape)
        #q_target = b_r + GAMMA * q_next.gather(1,action_)
        #loss = self.loss_func(q_eval, q_target, ISWeights)
        loss = self.loss_func(q_eval, q_target)



        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        #q_eval_new = self.eval_net(b_s).detach()  # .gather(1,b_a)
       # q_eval_new = torch.gather(q_eval_new, 1, b_a)

        #abs_errors = abs(q_target - q_eval_new)

       # self.memory.batch_update(tree_idx, abs_errors)


def train_dqn(episodes):
    dqn = DQN()
    # torch.save(dqn.state_dict(),'01.pkl')
    step = 0
    score_list = []

    global EPSILON
    print('\nCollecting experience...')
    for i_episode in range(episodes):
        s = env.reset()
        score = 0

        # for i in range(2000):
        while True:
            #if i_episode >= 150:
            env.render()


            action = dqn.choose_action(s)
            # a_number = 0.0
            # if action == 0:
            #     a_number = -1.0
            # elif action == 1:
            #     a_number = 0.0
            # elif action == 2:
            #     a_number = 1.0
            # elif action == 3:
            #     a_number = -0.5
            # elif action == 4:
            #     a_number = 0.5
            # elif action == 5:
            #     a_number = 0.75
            # elif action == 6:
            #     a_number = -0.75
            # elif action == 7:
            #     a_number = 0.25
            # elif action == 8:
            #     a_number = -0.25

            # a_number = 0.0
            # '''插值'''
            # if a==0:
            #     a_number = -1.0
            # elif a==1:
            #     a_number = 0.0
            # else:
            #     a_number = 1.0
            # take action

            s_, r, done, info = env.step(action)
            # print(r)
            # modify the reward
            # x,x_dot,theta,theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            #pos, velo = s_

            # '''Modified Reward'''
            score += r
            # pos += 0.5
            # r = abs(velo)
            #
            # if velo > 0:
            #     r *= 2
            #
            # #r += 2 * pos
            # if pos > 0.4:
            #     r += 10 * pos


            # if done:
            #     r+=3


            dqn.store_transition(s, action, r, done, s_)
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


        # if i_episode % 30 == 0:
        #     Print_Policy(dqn, str(i_episode))

        dqn.LR_reducer.step()
        if (EPSILON > EPSILON_min):
            EPSILON *= EPSILON_decay
            # step = 0
        print('episode:', i_episode, 'Epsilon:', EPSILON, 'score:', score)
    return score_list
# if np.mean(score_list[-10:]) > -400:
# dqn.save_model()
# break

# torch.save(dqn.state_dict(),'01.pkl')
#Print_Policy(dqn, 'Final')


for i in range(20):

    ReInitializeHPs()

    seed = i
    env.seed(seed)
    np.random.seed(400-seed)
    torch.manual_seed(600 - seed)  # 为CPU设置随机种子
    #torch.cuda.manual_seed(600 - seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(600 - seed)  # 为所有GPU设置随机种子

    episodes = 1000
    loss = np.array(train_dqn(episodes))
    #test_loss = np.array(test_loss)
    np.save('Torch_e_random_v1' + str(i)+'.npy', loss)
    #np.save('third_oracle_test_loss_Contin' + str(i)+'.npy', test_loss)
    plt.plot([i + 1 for i in range(episodes)], loss)
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
