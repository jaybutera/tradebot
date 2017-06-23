import numpy as np

class Simulator(object):
    def __init__ (self, orig_data, data, usd=1000., crypt=0.):
        self.data = data
        self.orig_data = orig_data

        self.reset()

    def reset (self, usd=1000., crypt=0.):
        self.usd = usd
        self.crypt = crypt

        # Initial state
        self.state = self.data[0] + [self.usd, self.crypt]
        # Total worth is usd + weightedAvg of crypt amount
        self.assets = self.usd + self.orig_data[0][5] * self.crypt
        # Time tracker
        self.t = 0

        # Storage
        self.usd_db = np.empty( len(self.data) )
        self.crypt_db = np.empty( len(self.data) )
        self.assets_db = np.empty( len(self.data) )

        # Settings
        self.log = True
        self.verbose = 1

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
        elif move == 1: # Sell crypt
            c = self.crypt * perc
            u = self.orig_data[self.t][5] * c
            self.usd += u
            self.crypt -= c
            if self.log and self.verbose > 1:
                print('selling ' , c , ' crypto for ' , u , \
                      'usd [own:', self.usd, 'usd | ', self.crypt, ' crypt')
        else: # Hold
            if self.log and self.verbose > 1:
                print('holding')
        # ----------------


        # Store info
        self.usd_db[self.t] = self.usd
        self.crypt_db[self.t] = self.crypt
        self.assets_db[self.t] = self.assets

        # Edge cases
        done = True if self.usd < 0.5 and self.crypt < 0.5 else False
        self.usd = np.max([self.usd, 0.])
        self.crypt = np.max([self.crypt, 0.])

        new_assets = self.usd + self.orig_data[self.t][5] * self.crypt

        # Reward is % change of assets
        reward = new_assets / self.assets - 1
        reward = reward if not done else -10 # Punish if all assets are lost

        # Update assets
        self.assets = new_assets

        # Update state
        self.state = self.data[self.t] + [self.usd, self.crypt]

        # Update time
        self.t += 1

        return reward, done

    def sim_done(self):
        return self.t == len(self.data)