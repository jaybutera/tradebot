import numpy as np

class Simulator(object):
    def __init__ (self, orig_data, data, usd=1000., crypt=0.):
        self.data = data
        self.orig_data = orig_data
        self.episode = -1

        self.reset()

        # Start first log file

    def reset (self, usd=1000., crypt=0.):
        self.episode += 1
        self.usd = usd
        self.crypt = crypt

        # Initial state
        self.state = self.data[0] + [self.usd/10000, self.crypt/10000]
        # Total worth is usd + weightedAvg of crypt amount
        self.assets = self.usd + self.orig_data[0][5] * self.crypt
        # Time tracker
        self.t = 0

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
            c = u / self.orig_data[self.t][5]
            self.crypt += c
            self.usd -= u
            if self.log and self.verbose > 1:
                print('buying ' , c , ' crypto with ' , u , \
                        'usd [own:', self.usd, 'usd | ', self.crypt, ' crypt')
            self.action_log.write('buying ' + str(c) + ' crypto with ' + str(u) + \
                    'usd [own:'+ str(self.usd) + 'usd | '+ str(self.crypt) + 'crypt]' \
                    + ' T: ' + str(self.t))
        elif move == 1: # Sell crypt
            c = self.crypt * perc
            u = self.orig_data[self.t][5] * c
            self.usd += u
            self.crypt -= c
            if self.log and self.verbose > 1:
                print('selling ' , c , ' crypto for ' , u , \
                      'usd [own:', self.usd, 'usd | ', self.crypt, ' crypt')
            self.action_log.write('selling ' + str(c) + ' crypto for ' + str(u) + \
                  'usd [own:' + str(self.usd) + 'usd | ' + str(self.crypt) + 'crypt]' \
                  + ' T: ' + str(self.t))
        else: # Hold
            if self.log and self.verbose > 1:
                print('holding')
            #self.action_log.write('holding ' + ' T: ' + str(self.t))
        # ----------------


        # Store info
        self.usd_db[self.t] = self.usd
        self.crypt_db[self.t] = self.crypt

        # Edge cases
        done = True if self.usd < 0.5 and self.crypt < 0.5 else False
        self.usd = np.max([self.usd, 0.])
        self.crypt = np.max([self.crypt, 0.])

        new_assets = self.usd + self.orig_data[self.t][5] * self.crypt

        # Reward is % change of assets
        reward = new_assets / self.assets - 1
        reward = reward if not done else -10 # Punish if all assets are lost
        self.reward_db[self.t] = reward

        # Log reward
        if move < 2:
            self.action_log.write(' R: ' + str(reward) + '\n')

        # Update assets
        self.assets = new_assets
        self.assets_db[self.t] = self.assets

        # Update state
        self.state = self.data[self.t] + [self.usd/10000, self.crypt/10000]

        # Update time
        self.t += 1

        return reward, done

    def sim_done(self):
        if self.t == len(self.data):
            self.action_log.close()
            return True
        else:
            return False

