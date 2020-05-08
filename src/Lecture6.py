# Written by Fernando Lasso to replicate the examples in lectures at ESADE


from math import exp
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def brownian_motion(T, N, vol, seed = 123, origin = 0):

    np.random.seed(seed)

    # Process parameters

    dt = T/N

    # Initial condition.
    x = origin
    wiener = [x]
    # Iterate to compute the steps of the Brownian motion.
    for k in range(N):
        x = x + vol * np.sqrt(dt) *norm.rvs()
        wiener.append(x)

    #brownian motion with trend
    rate = 0.15

    np.random.seed(seed)

    # Initial condition.
    x = origin

    gwiener = [x]
    trend = [origin]
    # Iterate to compute the steps of the Brownian motion.
    for k in range(N):
        x = x + dt * rate + np.sqrt(dt) * vol *norm.rvs()
        gwiener.append(x)
        trend.append(origin+k*dt*rate)

        
    pd.DataFrame({'Wiener': wiener, 'General Wiener': gwiener, 'Trend': trend}).plot()


# Montecarlo
class montecarlo(object):

    def __init__(self, mu, vol, T, r):
        self.mu = mu
        self.vol = vol
        self.T = T
        self.r = r

    def plot_step_simulation(self, origin, steps, n_sim = 50):

        dt = self.T / steps
        
        montecarlo = []

        for i in range(n_sim):
            S = origin
            values = [S]

            for i in range(steps):
                expected = (self.mu-self.vol**2 / 2) * dt 
                unexpected = self.vol * norm.rvs() * np.sqrt(dt)
                S = S* exp( expected + unexpected)
                values.append(S)

            montecarlo.append(values)
            plt.plot(values)

        plt.show() 

    def plot_one_step(self, origin, n_sim = 50000):
        # Now, thousand times more with one time step.
        expected = (self.mu-self.vol**2 / 2) * self.T 
        unexpected = self.vol * norm.rvs(size=n_sim) * np.sqrt(self.T)
        x = origin * np.exp(expected + unexpected)
        plt.hist(x, bins='auto')
        plt.show

    def present_value(self, origin, K, call = True, n_sim = 50000):

        c_to_p = call*2-1

        expected = (self.mu-self.vol**2 / 2) * self.T 
        unexpected = self.vol * norm.rvs(size=n_sim) * np.sqrt(self.T)
        x = origin * np.exp(expected + unexpected)

        return np.mean(np.maximum( c_to_p * (x - K), 0)) * exp(-self.r*self.T)
