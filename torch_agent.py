import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import random

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


'''
------------------------------------
NN MODEL
------------------------------------
'''
class NN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()

        self.linear = nn.Linear(state_size, 50)
        self.linear1 = nn.Linear(50, 30)
        self.linear2 = nn.Linear(30, 50)
        #self.lstm = nn.LSTM(15, 15)
        self.linear3 = nn.Linear(50, action_size)

    def forward(self, x):
        x = F.relu( self.linear(x) )
        x = F.relu( self.linear1(x) )
        x = F.relu( self.linear2(x) )
        #x = F.relu( self.lstm(x) )
        return F.sigmoid( self.linear3(x) )

'''
------------------------------------
DEEP Q NETWORK
------------------------------------
'''
class DQN():
    def __init__(self, state_size, action_size):

        # Settings
        self.state_size = state_size
        self.action_size = action_size
        self.batchsize = 32
        self.discount_factor = 0.99
        self.learning_rate = 0.005
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.train_start = 1000

        # Replay memory
        self.capacity = 2000
        self.memory = []
        self.position = 0

        # Neural nets
        self.model = NN(state_size, action_size)
        self.target_model = NN(state_size, action_size)

        self.optimizer = optim.Adam(self.model.parameters()
                , lr=self.learning_rate)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            q_value = np.random.rand(self.action_size,)
            q_value[3] = 1.
            return q_value
        else:
            state = state.view(1,len(state))
            x = Variable(state, volatile=True).type(FloatTensor)
            q_value = self.model(x)
            q_value[0][3] = 1.

            return q_value.data[0].numpy() # Torch tensor to np array

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        torch.save(self.model.state_dict(), './save_model/dqn_agent.pt')

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return Variable(Tensor([0.]))

        batch_size = min(self.batchsize, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # (batch_size, # features)
        #states = Variable(torch.zeros(batch_size,
        #    self.state_size).type(Tensor))
        predictions = Variable(torch.zeros(batch_size,
            self.action_size).type(Tensor))
        # (batch_size, # actions)
        targets     = Variable(torch.zeros(batch_size,
            self.action_size).type(Tensor), volatile=True)

        # Construct training batch
        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]

            s = Variable(state).view(1,len(state))
            #si = Variable(state).view(len(state))
            #states[i] = si
            ns = Variable(next_state).view(1,len(next_state))

            prediction = self.model(s)[0]
            target = prediction.clone()
            max_idx = np.argmax(action[:2])

            # Q Learning, get maximum Q value at s'
            if done:
                target[max_idx] = reward
                target[3] = reward
            else:
                t = self.target_model(ns)[0]
                target[max_idx] = reward + self.discount_factor * \
                                    t[max_idx].data

                # Always update % to buy/sell
                target[3] = reward + self.discount_factor * t[3].data

            predictions[i] = prediction
            targets[i] = target

        # Allow gradients to be computing for model fitting
        targets.volatile = False

        # Compute Huber loss
        '''
        out = self.model(states)
        l = torch.nn.MSELoss()
        loss = l(out, targets)
        '''
        loss = F.smooth_l1_loss(predictions, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        #for param in self.model.parameters():
        #    param.grad.data.clamp_(-1,1)
        self.optimizer.step()

        self.save_model()

        return loss
