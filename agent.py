import sys
from matplotlib import pyplot as plt
import random
import data as dl
import numpy as np
from sim import Simulator
from collections import deque
from keras.layers import Dense, LSTM, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 300


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


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step

    #data = dl.get_norm_data('https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1435699200&end=9999999999&period=14400')
    #orig_data = dl.get_data('https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1435699200&end=9999999999&period=14400')
    '''
    data = dl.get_norm_data('btc_eth_lowtrend.npy')[1000:2000]
    orig_data = dl.get_data('btc_eth_lowtrend.npy')[1000:2000]
    '''
    orig_data, data = dl.test_data_sin(250)
    state_size = len( data[0] ) + 2 # last 2 are current assets (usd, crypt)
    action_size = 4 # [Buy, Sell, Hold, % to buy/sell]

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    # Settings
    log = False
    verbose = 1

    sim = Simulator(orig_data, data)

    for e in range(EPISODES):
        score = 0

        #for i in range(len(data)):
        while not sim.sim_done():
            state = sim.state # Get state
            action = agent.get_action(state)
            if log and verbose > 1:
                print(action)

            #actions[i] = action

            # Simulate trading
            #-----------
            max_idx = np.argmax(action[:3]) # Choose buy/sell/hold
            reward, done = sim.step(max_idx, action[3])
            next_state = sim.state # Get new state
            #-----------

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)

            # every time step do the training
            lstm_layer = agent.model.layers[0]
            #print(lstm_layer.states)
            # Store lstm states
            state_record = lstm_layer.states
            # Reset states
            agent.model.layers[0].reset_states()
            agent.target_model.layers[0].reset_states()

            agent.train_replay()
            # Restore states
            agent.model.layers[0].states = state_record

            score += reward
            state = next_state

            if done:
                break

        # every episode update the target model to be same with model
        agent.update_target_model()

        # every episode, plot the play time
        scores.append(score)
        episodes.append(e)
        #plt.plot(episodes, scores, 'b')

        '''
        Plot normalized data
        '''
        if verbose > 0:
            try:
                t = np.arange(len(data))
                # Usd
                u = usd_db[:len(data)]
                plt.plot(t, np.divide(u, np.max(u)), 'b', label='usd')
                # Crypt
                l = crypt_db[:len(data)]
                l = [x if x > 0. else 0. for x in l]
                plt.plot(t, np.divide(l, np.max(l)), 'r', label='crypto')
                # Assets
                a = assets_db[:len(data)]
                plt.plot(t, np.divide(a, np.max(a)), 'g', label='assets')
                # Normalized weighted avg data
                w_avg = [x[5] for x in data]
                plt.plot(t, w_avg, 'k', label='norm data')
                # Display legend
                plt.legend(loc='lower right')

                plt.savefig("./save_graph/activity_e" + str(e) + ".png")
                if log:
                    plt.show()

                # Clear plot
                plt.clf()
            except:
                print(e)

        #plt.savefig("./save_graph/Cartpole_DQN.png")
        print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
              "  epsilon:", agent.epsilon)

        # save the model
        if e % 5 == 0:
            agent.save_model("./save_model/agent.h5")
