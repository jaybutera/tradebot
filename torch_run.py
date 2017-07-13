from torch_agent import *
import data as dl
import numpy as np
from sim import Simulator
from matplotlib import pyplot as plt


EPISODES = 200


if __name__ == "__main__":
    #data = dl.get_norm_data('https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1435699200&end=9999999999&period=14400')
    #orig_data = dl.get_data('https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1435699200&end=9999999999&period=14400')
    '''
    data = dl.get_norm_data('btc_eth_lowtrend.npy')[1000:2000]
    orig_data = dl.get_data('btc_eth_lowtrend.npy')[1000:2000]
    '''
    orig_data, data = dl.test_data_sin(1500)
    state_size = len( data[0] ) #+ 2 # last 2 are current assets (usd, crypt)
    action_size = 4 # [Buy, Sell, Hold, % to buy/sell]
    windowsize = 10

    #load_agent = False
    agent = DQN(windowsize, state_size, action_size)
    #agent.load_state()

    losses, scores, episodes = [], [], []

    sim = Simulator(orig_data, data, windowsize=windowsize)

    for e in range(EPISODES):
        # Write actions to log file
        score = 0
        state = Tensor( sim.reset() )

        while not sim.sim_done():
            #state = Tensor(sim.state) # Get state
            action = agent.get_action(state)

            # Simulate trading
            #-----------
            max_idx = np.argmax(action[:3]) # Choose buy/sell/hold
            next_state, reward, done = sim.step(max_idx, action[3])
            next_state = Tensor(next_state)
            #-----------

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)
            state = next_state.clone()

            #loss = agent.train_replay()
            #losses.append(loss.data.numpy()[0])

            score += reward

            if done:
                print('done!')
                sim.reset()
                break

        loss = agent.train_replay()
        # every episode update the target model to be same with model
        agent.update_target_model()
        if e % 5 == 0:
            agent.save_state()

        # every episode, plot the play time
        scores.append(score)
        episodes.append(e)


        '''
        Plot normalized data
        '''
        if True:
            t = np.arange(len(data))
            # Usd
            u = sim.usd_db[:len(data)]
            plt.plot(t, np.divide(u, np.max(u)), 'b', label='usd')
            # Crypt
            l = sim.crypt_db[:len(data)]
            l = [x if x > 0. else 0. for x in l]
            plt.plot(t, np.divide(l, np.max(l)), 'r', label='crypto')
            # Assets
            a = sim.assets_db[:len(data)]
            plt.plot(t, np.divide(a, np.max(a)), 'g', label='assets')
            # Normalized weighted avg data
            w_avg = [x[5] for x in data]
            plt.plot(t, w_avg, 'k', label='norm data')
            # Display legend
            plt.legend(loc='lower right')

            plt.savefig("./save_graph/activity_e" + str(e) + ".png")

            # Clear plot
            plt.clf()

        print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
          "  epsilon:", agent.epsilon)

        # Reset the simulation
        sim.reset()

    plt.plot(scores)
    plt.savefig('./loss.png')
    plt.show()
