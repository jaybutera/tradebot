import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

'''
------------------------------------
NN MODEL
------------------------------------
'''
class NN(nn.Module):
    def __init__(self, windowsize, state_size, action_size):
        super(NN, self).__init__()

        #self.linear = nn.Linear(windowsize, 50)
        '''
        self.conv1 = nn.Conv1d(1, 4, 2, stride=1)
        self.conv2 = nn.Conv1d(4, 6, 2, stride=1)
        #self.linear1 = nn.Linear(8, 30)
        self.linear2 = nn.Linear(48, 20)
        #self.lstm = nn.LSTM(15, 15)
        '''
        self.linear2 = nn.Linear(windowsize, 10)
        self.linear3 = nn.Linear(10, action_size)

    def forward(self, x):
        '''
        x = F.relu( self.conv1(x) )
        x = F.relu( self.conv2(x) )
        x = x.view(-1, 48)
        x = F.relu( self.linear2(x) )
        '''
        x = F.relu( self.linear2(x) )
        return F.relu( self.linear3(x) )

'''
------------------------------------
CONVOLUTIONAL MODEL
CARTPOLE
------------------------------------
'''
class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

