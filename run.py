from agent import *
import data as dl
import numpy as np
#from viz import Visualizer
from sim import Simulator
from matplotlib import pyplot as plt


EPISODES = 300

if __name__ == "__main__":
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

    sim = Simulator(orig_data, data)
    #viz = Visualizer(

    for e in range(EPISODES):
        score = 0

        while not sim.sim_done():
            state = sim.state # Get state
            action = agent.get_action(state)

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
                sim.reset()
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
        if False:
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

        # Reset the simulation
        sim.reset()

        # save the model
        if e % 5 == 0:
            agent.save_model("./save_model/agent.h5")
