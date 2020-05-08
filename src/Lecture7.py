# Written by Fernando Lasso to replicate the examples in lectures at ESADE


from math import exp
import numpy as np
from scipy.stats import norm
from scipy import optimize

# As per slide 15/43

def black_scholes(S, K, vol, r, T, call = True):
    d1 = (np.log(S/K) + (r + 0.5* vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if call:
        return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    
    return K * exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1) 

def historic_volatility(prices,days):
    n = len(prices)
    ret = np.log(prices/prices.shift(1))[1:n]
    std = np.sqrt((1/(n-1))*sum((ret-np.mean(ret))**2))
    return std / np.sqrt(days/252)

def minimize_bs(vol, S, K, r, T, price, call = True):
    return abs(black_scholes(S, K, vol, r, T, call) - price)

def implied_volatility(price, S, K, r, T, call = True):
    return optimize.minimize(minimize_bs, 0.2, (S,K,r,T,price, call)).x[0]

    