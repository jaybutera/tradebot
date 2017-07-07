import numpy as np

class Simulator(object):
    def __init__ (self, orig_data, data, usd=1000., crypt=0., windowsize=1):
        self.data = data
        self.orig_data = orig_data
        self.episode = -1
        self.windowsize = windowsize

        # Start first log file

    def reset (self, usd=1000., crypt=0.):
        self.episode += 1
        self.usd = usd
        self.crypt = crypt

        # Initial state
        state = [i[0] for i in self.data[0:self.windowsize]] #+ [self.usd/10000, self.crypt/10000]
        # Total worth is usd + weightedAvg of crypt amount
        self.assets = self.usd + self.orig_data[0][5] * self.crypt
        # Time tracker
        self.i = 1
        # Real time
        self.t = lambda : self.i + self.windowsize

        # Storage
        self.usd_db = np.empty( len(self.data) )
        self.crypt_db = np.empty( len(self.data) )
        self.assets_db = np.empty( len(self.data) )
        self.reward_db = np.empty( len(self.data) )

        # Settings
        self.log = True
        self.verbose = 1

        # Log actions
        self.action_log = open('./action_logs/record_book_' + str(self.episode) + '.txt', 'w+')

        return state

    def step (self, move, perc):
        '''
        Action = [move, percent]
        move = 0 | Buy
        move = 1 | Sell
        move = 2 | Hold
        '''

        if self.log and self.verbose > 2:
            print('using', perc, '%')

        # Simulate trading
        # ----------------
        if move == 0: # Buy crypt
            u = self.usd * perc # Amount to use
            c = u / self.orig_data[self.t()][5]
            self.crypt += c
            self.usd -= u
            if self.log and self.verbose > 1:
                print('buying ' , c , ' crypto with ' , u , \
                        'usd [own:', self.usd, 'usd | ', self.crypt, ' crypt')
            self.action_log.write('buying ' + str(c) + ' crypto with ' + str(u) + \
                    'usd [own:'+ str(self.usd) + 'usd | '+ str(self.crypt) + 'crypt]' \
                    + ' T: ' + str(self.i))
        elif move == 1: # Sell crypt
            c = self.crypt * perc
            u = self.orig_data[self.t()][5] * c
            self.usd += u
            self.crypt -= c
            if self.log and self.verbose > 1:
                print('selling ' , c , ' crypto for ' , u , \
                      'usd [own:', self.usd, 'usd | ', self.crypt, ' crypt')
            self.action_log.write('selling ' + str(c) + ' crypto for ' + str(u) + \
                  'usd [own:' + str(self.usd) + 'usd | ' + str(self.crypt) + 'crypt]' \
                  + ' T: ' + str(self.i))
        else: # Hold
            if self.log and self.verbose > 1:
                print('holding')
            #self.action_log.write('holding ' + ' T: ' + str(self.i))
        # ----------------


        # Store info
        self.usd_db[self.i] = self.usd
        self.crypt_db[self.i] = self.crypt

        # Edge cases
        done = True if self.usd < 0.5 and self.crypt < 0.5 else False
        self.usd = np.max([self.usd, 0.])
        self.crypt = np.max([self.crypt, 0.])

        new_assets = self.usd + self.orig_data[self.t()][5] * self.crypt

        # Reward is % change of assets
        reward = (new_assets / self.assets - 1) * 10
        reward = reward if not done else -10 # Punish if all assets are lost
        self.reward_db[self.i] = reward

        # Log reward
        if move < 2:
            self.action_log.write(' R: ' + str(reward) + '\n')

        # Update assets
        self.assets = new_assets
        self.assets_db[self.i] = self.assets

        # Update state
        state = [i[0] for i in self.data[self.i:self.i+self.windowsize]] #+ [self.usd/10000, self.crypt/10000]

        # Update time
        self.i += 1

        return state, reward, done

    def sim_done(self):
        if self.t() == len(self.data):
            self.action_log.close()
            return True
        else:
            return False

