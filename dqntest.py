
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
#import pandas as pd
import gym
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import copy
import random
import time
from collections import deque
# from tensorboardX import SummaryWriter


# Hyper Parameters
BATCH_SIZE = 64
LR = 0.001  # learning rate
EPSILON = 1  # greedy policy
EPSILON_min = 0.01
EPSILON2 = 1
EPSILON_decay = 0.943
GAMMA = 0.95  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency


MEMORY_CAPACITY = 50000
LR_decay_steps = 100
LR_decay_gamma = 0.5
#env = gym.make('MountainCarContinuous-v0')
env = gym.make('MountainCar-v0')
#env = gym.make('CartPole-v0')
#env = gym.make('Pendulum-v0')
#env = env.unwrapped
print(env.observation_space)
# print(env.action_space.high)
# print(env.action_space.low)

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
print(N_STATES, N_ACTIONS)


import os
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive"

os.chdir(path)
os.listdir(path)
#UCB_writer = SummaryWriter('./assets_new/log/Baselines/',comment='Test9')


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
        if torch.cuda.is_available():
          x = x.cuda()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value



UCB_BATCH_SIZE = 512
class DQN():
    def __init__(self):

        self.eval_net = Net()
        self.target_net = Net()
        self.target_net.requires_grad = False
        if torch.cuda.is_available():
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.learn_step_counter = 0

        self.memory = deque(maxlen=MEMORY_CAPACITY)

        self.Oracle_GD_Times = 5

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        self.loss_func = nn.MSELoss()


#---------------------------------------UCB Calculation---------------------------------------
    def Oracle_Loss1(self, Q_eval, Q_target):
        reg = len(self.memory)/UCB_BATCH_SIZE
        reg*=0.2

        return reg * torch.sum((Q_eval[1:] - Q_target[1:] ) ** 2) + (Q_target[0] - Q_eval[0])
    def Oracle_Loss2(self, Q_eval, Q_target):
        reg = len(self.memory)/UCB_BATCH_SIZE
        reg*=0.2

        return reg * torch.sum((Q_eval[1:] - Q_target[1:] ) ** 2) - (Q_target[0] - Q_eval[0])

    def ucb(self, x, a, a_init):

        a_init_value = torch.unsqueeze(torch.unsqueeze(a_init, 0), 1)

        if torch.cuda.is_available():
          Temp_Net1 = Net().cuda()
          Temp_Net2 = Net().cuda()
        else:
          Temp_Net1 = Net()
          Temp_Net2 = Net()
        Temp_Net1.load_state_dict(self.eval_net.state_dict())
        Temp_Net2.load_state_dict(self.eval_net.state_dict())
        Temp1_optimizer = torch.optim.Adam(Temp_Net1.parameters(), lr=0.001)# LR change?
        Temp2_optimizer = torch.optim.Adam(Temp_Net2.parameters(), lr=0.001)# LR change?



        b_memory = random.sample(self.memory, UCB_BATCH_SIZE)
        b_s = np.array([i[0] for i in b_memory])
        b_a = np.array([i[1] for i in b_memory])
        b_r = np.array([i[2] for i in b_memory])
        b_done = np.array([i[3] for i in b_memory])
        b_s_ = np.array([i[4] for i in b_memory])

        b_s = np.squeeze(b_s)
        b_s_ = np.squeeze(b_s_)

        if torch.cuda.is_available():

            b_s = np.vstack((x.reshape((1, -1)), b_s))
            b_s = torch.cuda.FloatTensor(b_s)
            b_a = np.insert(b_a, 0, a)
            b_a = torch.unsqueeze(torch.cuda.LongTensor(b_a), 1)

            b_a = torch.cuda.LongTensor(b_a)
            b_done = torch.cuda.FloatTensor(b_done)
            b_r = torch.cuda.FloatTensor(b_r)
            b_s_ = torch.cuda.FloatTensor(b_s_)
        else:

            b_s = np.vstack((x.reshape((1, -1)), b_s))
            b_s = torch.FloatTensor(b_s)
            b_a = np.insert(b_a, 0, a)
            b_a = torch.unsqueeze(torch.LongTensor(b_a), 1)

            b_a = torch.LongTensor(b_a)
            b_done = torch.FloatTensor(b_done)
            b_r = torch.FloatTensor(b_r)
            b_s_ = torch.FloatTensor(b_s_)



        for i in range(self.Oracle_GD_Times):


            with torch.no_grad():
                q_target = self.eval_net(b_s_).detach()

            q_target = b_r.squeeze() + GAMMA * torch.mul(torch.max(q_target, 1)[0],
                                                         1 - b_done.squeeze())

            q_target = torch.cat((a_init_value, q_target.view((UCB_BATCH_SIZE, 1))), 0)

            Temp1_optimizer.zero_grad()
            q_eval1 = Temp_Net1.forward(b_s)
            #b_a = torch.unsqueeze(b_a, 1)
           # print(q_eval.shape, b_a.shape)
            q_eval1 = torch.gather(q_eval1, 1, b_a)
            loss = self.Oracle_Loss1(q_eval1, q_target)
            loss.backward()
            Temp1_optimizer.step()

            Temp2_optimizer.zero_grad()
            q_eval2 = Temp_Net2.forward(b_s)
            # b_a = torch.unsqueeze(b_a, 1)
            # print(q_eval.shape, b_a.shape)
            q_eval2 = torch.gather(q_eval2, 1, b_a)
            loss = self.Oracle_Loss2(q_eval2, q_target)
            loss.backward()
            Temp2_optimizer.step()


        with torch.no_grad():
            ucb = (Temp_Net1(x).squeeze()[a].detach() - Temp_Net2(x).squeeze()[a].detach())

        return ucb


    def Oracle(self, x, actions_value):
        ucbs = np.zeros(N_ACTIONS)

        if torch.cuda.is_available():
            ucbs = torch.cuda.FloatTensor(ucbs)
        else:
            ucbs = torch.FloatTensor(ucbs)

        for action in range(N_ACTIONS):
            ucbs[action] = self.ucb(x, action, actions_value.squeeze()[action])

        #chose_action = np.argmax(ucbs)
        #print(ucbs, chose_action)
        return torch.unsqueeze(ucbs,0)

    def choose_action_ORACLE(self, x):
        if len(self.memory) < UCB_BATCH_SIZE:
            return int(np.random.randint(0, N_ACTIONS))

        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        with torch.no_grad():
            actions_value_setup = self.eval_net.forward(x).detach()
            actions_value = copy.deepcopy(actions_value_setup)

        ucbs = self.Oracle(x, actions_value)
        actions_value_setup += ucbs

        action = torch.max(actions_value_setup, 1)[1].cpu().numpy()

        action = int(action)
        #print(ucbs, action)
        return action
        
    def choose_action_MIX(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        with torch.no_grad():
            actions_value = self.eval_net.forward(x).detach()

        if np.random.uniform() > EPSILON:  # greedy

            action = torch.max(actions_value, 1)[1].cpu().numpy()

        else:
            if np.random.uniform() <= EPSILON2:
                action = self.choose_action_ORACLE(x)
            else:
                action = np.random.randint(0, N_ACTIONS)

        action = int(action)

        return action

    def choose_action_BASELINE(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        with torch.no_grad():
            actions_value = self.eval_net.forward(x).detach()

        if np.random.uniform() > EPSILON:  # greedy

            action = torch.max(actions_value, 1)[1].cpu().numpy()

        else:
            action = np.random.randint(0, N_ACTIONS)

        action = int(action)

        return action

    def store_transition(self, s, a, r, done, s_):
        self.memory.append((s, a, r, done, s_))



    def learn(self):


        if len(self.memory) < BATCH_SIZE:
            return

        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        for i in range(3):
            b_memory = random.sample(self.memory, BATCH_SIZE)
            b_s = np.array([i[0] for i in b_memory])
            b_a = np.array([i[1] for i in b_memory])
            b_r = np.array([i[2] for i in b_memory])
            b_done = np.array([i[3] for i in b_memory])
            b_s_ = np.array([i[4] for i in b_memory])

            b_s = np.squeeze(b_s)
            b_s_ = np.squeeze(b_s_)


            if torch.cuda.is_available():
                b_s = torch.cuda.FloatTensor(b_s)
                b_a = torch.cuda.LongTensor(b_a)
                b_done = torch.cuda.FloatTensor(b_done)
                b_r = torch.cuda.FloatTensor(b_r)
                b_s_ = torch.cuda.FloatTensor(b_s_)
            else:
                b_s = torch.FloatTensor(b_s)
                b_a = torch.LongTensor(b_a)
                b_done = torch.FloatTensor(b_done)
                b_r = torch.FloatTensor(b_r)
                b_s_ = torch.FloatTensor(b_s_)


            self.optimizer.zero_grad()
            q_eval = self.eval_net.forward(b_s)

            b_a = torch.unsqueeze(b_a,1)

            q_eval = torch.gather(q_eval, 1, b_a)


            with torch.no_grad():
                q_target = self.target_net(b_s_).detach()


            q_target = b_r.squeeze() + GAMMA * torch.mul(torch.max(q_target, 1)[0], 1-b_done.squeeze())
            q_target = q_target.view((BATCH_SIZE, 1))
            loss = self.loss_func(q_eval, q_target)
            loss.backward()
            self.optimizer.step()

        self.learn_step_counter += 1




def train_dqn(episodes):
    dqn = DQN()
    step = 0
    score_list = []

    global EPSILON
    print('\nCollecting experience...')
    for i_episode in range(episodes):
        s = env.reset()
        score = 0

        while True:

            #env.render()
            #s[1]*=10
            action = dqn.choose_action_ORACLE(s)

            #a_number = 0.0

            #if a==0:
             #   a_number = -1.0
            #elif a==1:
            #    a_number = 0.0
            #elif a==2:
            #    a_number = 1.0
            #elif a==3:
            #    a_number = 0.5 
            #elif a==4:
            #    a_number = -0.5 
            #action = [a_number]
            for i in range(4):
              s_, r, done, info = env.step(action)
              score += r
              if done:
                score_list.append(score)
                break
            #s_[1]*=10
            dqn.store_transition(s, action, r, done, s_)
            if done:
                
              break

            
            s = s_
            step += 1


            dqn.learn()

        EPSILON *= EPSILON_decay

        print('episode:', i_episode, 'Epsilon:', EPSILON, 'score:', score)
        #UCB_writer.add_scalar('Baseline_Reward'+str(i),score,i_episode)
    return score_list, dqn
    

def test_dqn(episodes, dqn):
    #dqn = DQN()
    #dqn.eval()
    step = 0
    score_list = []

    EPSILON = -1
    print('\nTesting...')
    for i_episode in range(episodes):
        s = env.reset()
        score = 0

        while True:

            #env.render()
            
            action = dqn.choose_action_BASELINE(s)

            s_, r, done, info = env.step(action)
            

            score += r

            #dqn.store_transition(s, action, r, done, s_)


            if done:
                score_list.append(score) 
                break
            s = s_
            step += 1


            

        #EPSILON *= EPSILON_decay

        print('Test_episode:', i_episode, 'Epsilon:', EPSILON, 'score:', score)
        #UCB_writer.add_scalar('Baseline_Reward'+str(i),score,i_episode)
    return score_list

def ReInitializeHPs():
    global LR,EPSILON,EPSILON_min

    LR = 0.001  # learning rate
    EPSILON = 1  # greedy policy




for i in range(20):

    i = 19-i

    ReInitializeHPs()

    seed = i
    env.seed(seed)
    np.random.seed(400-seed)
    random.seed(400 - seed)
    torch.manual_seed(600 - seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(600 - seed)

    episodes = 300
    loss, dqn = np.array(train_dqn(episodes))
    #test_episodes = 70
    #loss = test_dqn(test_episodes, dqn)
    np.save('Mountaincar_UCB_v6m_' + str(i)+'.npy', loss)
   # plt.plot([i + 1 for i in range(episodes)], loss)
   # plt.show()
# train_writer.close()
