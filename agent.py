import sys
import random
import numpy as np
from collections import deque
from keras.layers import Dense, LSTM, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.models import Sequential


# this is DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        '''
        model.add(TimeDistributed(Dense(24, activation='sigmoid', \
            kernel_initializer='he_uniform'), \
            input_shape=(None,1,self.state_size)))
        '''
        model.add(LSTM(24, activation='sigmoid', init='uniform', stateful=True,
            batch_input_shape=(1,1,self.state_size)))
        #model.add(Dense(24, activation='sigmoid', init='he_uniform'))
        model.add(Dense(self.action_size, activation='sigmoid', init='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        #if np.random.rand() <= self.epsilon:
        if True:
            q_value = np.random.rand(self.action_size,)
            q_value[3] = 1.
            return q_value
        else:
            state = np.array(state)
            state = state.reshape((1,1,state.size))
            q_value = self.model.predict(state)[0]
            q_value[3] = 1.
            return q_value

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        # First reset lstm states
        lstm_layer = self.model.layers[0]
        # Store lstm states
        state_record = lstm_layer.states
        # Reset states
        self.model.layers[0].reset_states()
        self.target_model.layers[0].reset_states()

        batch_size = min(self.batch_size, len(self.memory))
        #mini_batch = random.sample(self.memory, batch_size)
        # Mini batch must hold temporal information. No rand sampling
        rand_idx = random.randint(0, len(self.memory)-1)
        mini_batch = [self.memory[rand_idx-i] for i in range(batch_size)]

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            state = np.array(state)
            state = state.reshape((1,1,state.size))
            next_state = np.array(next_state)
            next_state = next_state.reshape((1,1,next_state.size))
            target = self.model.predict(state)[0]

            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done:
                target[max_idx] = reward
                target[3] = reward
            else:
                max_idx = np.argmax(action[:2]) # Choose buy/sell/hold
                target[max_idx] = reward + self.discount_factor * \
                    self.target_model.predict(next_state)[0][max_idx]

                # Always update % to buy/sell
                target[3] = reward + self.discount_factor * \
                    self.target_model.predict(next_state)[0][3]

            update_input[i] = state
            update_target[i] = target

        update_input=update_input.reshape(update_input.shape[0],update_input.shape[1])
        update_target=update_target.reshape(update_target.shape[0],update_target.shape[1])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        '''
        for ui, ut in zip(update_input, update_target):
            ui = ui.reshape(1,1,ui.size)
            ut = ut.reshape(1,ut.size)
            self.model.fit(ui, ut, batch_size=1, \
                    shuffle=False, nb_epoch=1, verbose=0)
        '''
        ui = update_input[-1].reshape(1,1,update_input[-1].size)
        ut = update_target[-1].reshape(1,update_target[-1].size)
        self.model.fit(ui, ut, batch_size=1, \
                shuffle=False, nb_epoch=1, verbose=0)

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

