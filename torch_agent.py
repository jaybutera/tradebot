import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


'''
------------------------------------
REPLAY MEMORY
------------------------------------
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
'''

'''
------------------------------------
DEEP Q NETWORK
------------------------------------
'''
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        # Settings
        self.state_size = state_size
        self.action_size = action_size
        self.batchsize = 32
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.train_start = 1000

        # Replay memory
        self.capacity = 2000
        self.memory = []
        self.position = 0

        self.linear = nn.Linear(state_size, action_size)

    def forward(self, x):
        return F.relu( self.linear(x) )

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            q_value = np.random.rand(self.action_size,)
            q_value[3] = 1.
            return q_value
        else:
            #state = np.array(state)
            #state = state.reshape((state.size))
            q_value = self( Variable(state, volatile=True).type(FloatTensor)).data
            q_value[3] = 1.

            return q_value.numpy() # Torch tensor to np array

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

def train_replay(agent):
    if len(agent.memory) < agent.train_start:
        return

    batch_size = min(agent.batchsize, len(agent.memory))
    mini_batch = random.sample(agent.memory, batch_size)

    # (batch_size, # features)
    predictions = Variable(torch.zeros(batch_size,
        agent.state_size).type(Tensor), volatile=True)
    # (batch_size, # actions)
    targets     = Variable(torch.zeros(batch_size,
        agent.action_size).type(Tensor), volatile=True)

    # Construct training batch
    for i in range(batch_size):
        state, action, reward, next_state, done = mini_batch[i]
        state = np.array(state)
        #state = state.reshape((1,1,state.size))
        next_state = np.array(next_state)
        #next_state = next_state.reshape((1,1,next_state.size))
        prediction = agent.get_action(state)
        target = np.array(prediction, copy=True)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
            target[max_idx] = reward
            target[3] = reward
        else:
            max_idx = np.argmax(action[:2]) # Choose buy/sell/hold
            target[max_idx] = reward + agent.discount_factor * \
                agent.target_model.predict(next_state)[0][max_idx]

            # Always update % to buy/sell
            target[3] = reward + agent.discount_factor * \
                agent.target_model.predict(next_state)[0][3]

        predictions[i] = prediction
        targets[i] = target

    # Allow gradients to be computing for model fitting
    predictions.volatile = False
    tagets.volatile = False

    # Compute Huber loss
    loss = F.smooth_11_loss(predictions, targets)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in model.parameters():
    #    param.grad.data.camp_(-1,1)
    optimizer.step()
