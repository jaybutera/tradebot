from torch_agent import *
import gym
import data as dl
import numpy as np
from sim import Simulator
from matplotlib import pyplot as plt

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

EPISODES = 400

'''
screen_width = 600

class AtariEnv(object):
    def __init__(self, env, input_shape=(84, 84)):
        self.env = env
        self.input_shape = input_shape

    def step(self, action):
        o, r, done, env_info = self.env.step(action)
        o = self._preprocess_obs(o)
        max_o = np.maximum(o, self.last_o)
        self.last_o = o
        r = self._preprocess_r(r)
        return max_o, r, done, env_info
    
    def reset(self):
        o = self.env.reset()
        o = self._preprocess_obs(o)
        self.last_o = o
        return o

    def render(self):
        self.env.render()

    def seed(self, s):
        self.env.seed(s)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def _preprocess_obs(self, obs):
        assert obs.ndim == 3  # (height, width, channel)
        img = Image.fromarray(obs)
        img = img.resize(self.input_shape).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == self.input_shape
        return self._to_float(processed_observation).reshape(1, *self.input_shape)

    def _to_float(self, data):
        """
        int to float
        """
        processed_data = data.astype('float32') / 255.
        return processed_data

    def _preprocess_r(self, reward):
        return np.clip(reward, -1., 1.)
'''

if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    action_size = 2
    state_size = 4
    agent = DQN(state_size, action_size, windowsize=state_size)

    losses, scores, episodes = [], [], []

    for e in range(EPISODES):
        # Write actions to log file
        score = 0
        state = env.reset()
        state = Tensor(state)

        while True:
            #state = Tensor(sim.state) # Get state
            action = agent.get_action(state)

            # Simulate trading
            #-----------
            max_idx = np.argmax(action)
            next_state, reward, done, _ = env.step(max_idx)
            next_state = Tensor(next_state)
            #-----------

            # save the sample <s, a, r, s'> to the replay memory
            reward = reward if not done else -10
            agent.replay_memory(state, action, reward, next_state, done)
            state = next_state.clone()

            #loss = agent.train_replay()
            #losses.append(loss.data.numpy()[0])

            score += reward

            if done:
                break

        loss = agent.train_replay()
        # every episode update the target model to be same with model
        agent.update_target_model()
        '''
        if e % 5 == 0:
            agent.save_state()
        '''

        # every episode, plot the play time
        scores.append(score)
        episodes.append(e)


        '''
        Plot normalized data
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
        '''

        print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
          "  epsilon:", agent.epsilon)

        # Reset the simulation
        env.reset()

    plt.plot(scores)
    plt.savefig('./loss.png')
    plt.show()
