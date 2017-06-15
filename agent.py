import sys
import gym
import pylab
import random
import data as dl
import numpy as np
from collections import deque
from keras.layers import Dense
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
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size,)
        else:
            state = np.array(state)
            state = state.reshape((1, state.size))
            q_value = self.model.predict(state)[0]
            return q_value

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state = mini_batch[i]
            state = np.array(state)
            state = state.reshape((1, state.size))
            next_state = np.array(next_state)
            next_state = next_state.reshape((1, next_state.size))
            target = self.model.predict(state)[0]

            # like Q Learning, get maximum Q value at s'
            # But from target model
            max_idx = np.argmax(action[:2]) # Choose buy/sell/hold
            target[max_idx] = reward + self.discount_factor * \
                self.target_model.predict(next_state)[0][max_idx]

            # Always update % to buy/sell
            target[3] = reward + self.discount_factor * \
                self.target_model.predict(next_state)[0][3]

            update_input[i] = state
            update_target[i] = target

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step

    data = dl.get_data('https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1435699200&end=9999999999&period=14400')
    state_size = len( data[0] ) + 2 # last 2 are current assets (usd, crypt)
    action_size = 4 # [Buy, Sell, Hold, % to buy/sell]

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        score = 0

        # Starting amounts
        usd = 1000.
        crypt = 0.

        # Initial state
        state = data[0] + [usd, crypt]
        # Total worth is usd + weightedAvg of crypt amount
        assets = usd + state[5] * crypt

        # Storage
        actions = np.empty( len(data) , dtype=list)
        usd_db = np.empty( len(data) )
        crypt_db = np.empty( len(data) )

        for i,tick in enumerate(data[1:50]):
            action = agent.get_action(state)

            actions[i] = action

            # Simulate trading
            #-----------
            max_idx = np.argmax(action[:2]) # Choose buy/sell/hold

            if max_idx == 0: # Buy crypt
                # (Weightedavg price) * (usd amount) * (% to buy)
                crypt += state[5] * usd * action[3]
                usd -= usd * action[3]
            elif max_idx == 1: # Sell crypt
                # (Weightedavg price) * (crypt amount) * (% to sell)
                usd += state[5] * crypt * action[3]
                crypt -= crypt * action[3]
            #-----------

            # Store info
            usd_db[i] = usd
            crypt_db[i] = crypt


            next_state = tick + [usd, crypt]
            new_assets = usd + state[5] * crypt
            # Reward is % change of assets
            reward = new_assets / assets
            # Update assets
            assets = new_assets


            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state)
            # every time step do the training
            agent.train_replay()
            score += reward
            state = next_state

        # every episode update the target model to be same with model
        agent.update_target_model()

        # every episode, plot the play time
        scores.append(score)
        episodes.append(e)
        #pylab.plot(episodes, scores, 'b')
        t = np.arange(len(data))
        pylab.plot(t, usd_db, 'b')
        pylab.plot(t, crypt_db, 'r')
        pylab.show()
        #pylab.savefig("./save_graph/Cartpole_DQN.png")
        print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
              "  epsilon:", agent.epsilon)

        # save the model
        if e % 5 == 0:
            agent.save_model("./save_model/agent.h5")
