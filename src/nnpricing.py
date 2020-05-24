import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
import pylab
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
from keras.utils.vis_utils import plot_model
import seaborn as sns
import matplotlib as mpl
from arch import arch_model
from visualizations import plot_prediction_error, plot_error_dist
import sys 
from sklearn.model_selection import train_test_split

def split_data(df, volatility, y, normalize = True, test_split = 0.8):
    if normalize:
         ## Normalize the data exploiting the fact that the BS Model is linear homogenous in S,K
        df["strike_price"] = df["strike_price"] /1000
        df["Price"] = df["Price"]/df["strike_price"]
        df["midpoint"] = df["midpoint"]/df["strike_price"]
        df["T"] = df["T"] / 365
    df['cp'] = 1
    df['cp'][df.cp_flag == "C"] =0

    tickers = pd.get_dummies(df['ticker'], prefix='Ticker')
    df = pd.concat([df, tickers], axis = 1)
    
    train, test = train_test_split(df, test_size=0.2)
    train_attributes = train[['cp_flag','ticker','strike_price']]
    test_attributes = test[['cp_flag','ticker','strike_price']]
    
    cols = ['Price', 'T', 'q', volatility, 'rf','cp',
            'Ticker_AAPL','Ticker_DIS','Ticker_JPM','Ticker_WMT']
    
    X_train = train[cols].values
    y_train = train[y].values
    
    X_test = test[cols].values
    y_test = test[y].values
    

    return X_train, y_train, X_test, y_test, train_attributes, test_attributes

def custom_activation(x):
    return backend.exp(x)

def create_neural_network(X_train, nodes = 120,save = True):

    model = Sequential()

    model.add(Dense(nodes, input_dim=X_train.shape[1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))

    model.add(Dense(nodes, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))

    model.add(Dense(1))
    model.add(Activation(custom_activation))
              
    model.compile(loss='mse',optimizer='rmsprop')
    if save: plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model 

def checkAccuracy(y,y_hat, attributes, plot = True):
    stats = dict()
    
    # Undo normalization
    y = y * attributes['strike_price']
    y_hat = y_hat * attributes['strike_price']

    stats['diff'] = (y - y_hat)
    stats['diff_pct'] = 100 * (y - y_hat) / y
    stats['diff_pct'].name = "Percentage Error"
    
    # Remove outliers (3 std away)
    remove_idx = np.abs(stats['diff_pct'])> (3*np.std(stats['diff_pct']))
    print("Removed {0} percent outliers".format(round(100*sum(remove_idx)/len(stats['diff_pct']),2)))
    
    stats['diff'] = stats['diff'][~remove_idx]
    stats['diff_pct'] = stats['diff_pct'][~remove_idx]
    y = y[~remove_idx]
    y_hat = y_hat[~remove_idx]

    stats['mse'] = np.mean(stats['diff']**2)
    print("Mean Squared Error:      ", stats['mse'])
    
    stats['rmse'] = np.sqrt(stats['mse'])
    print("Root Mean Squared Error: ", stats['rmse'])
    
    stats['mae'] = np.mean(abs(stats['diff']))
    print("Mean Absolute Error:     ", stats['mae'])
    
    stats['mpe'] = 100*np.sqrt(stats['mse'])/np.mean(y)
    print("Mean Percent Error:      ", stats['mpe'])
    
    if plot:
        #plots
        data = pd.DataFrame({"Actual":y, "Predicted":y_hat,
            "Ticker":attributes['ticker'], "CP Flag":attributes['cp_flag']})
            
        plot_prediction_error(data)
        plot_error_dist(stats['diff_pct'],attributes['ticker'])
    
    return stats

def rolling_garch_volatility(options, stocks):
    predictions = []
    for t in ["WMT","AAPL","JPM","DIS"]:
        print("\n"+t+":")
        data = stocks[t]
        returns = data.pct_change().dropna()*100
        # Forecast one ahead: 
        am = arch_model(returns, vol='GARCH', power=2.0, p=1, o=1, q=1)
        index = returns.index
        start_loc = 0
        end_loc = np.where(index >= '2016-01-01')[0].min()

        forecasts = {}

        for i in range(len(returns) - end_loc+1):
            
            if i%100 == 0 :
                sys.stdout.write(str(i))
                sys.stdout.flush()
            res = am.fit(first_obs=i, last_obs=i + end_loc, disp='off')
            temp = res.forecast(horizon=1).variance
            fcast = temp.iloc[i + end_loc - 1]
            forecasts[fcast.name] = fcast
        # Save the volatilities in 
        vol_hat = pd.DataFrame(forecasts).T.shift(1).dropna() / np.sqrt(252)
        vol_hat['ticker'] = t
        predictions.append(vol_hat) 

    garch_vol = pd.concat(predictions).reset_index()
    garch_vol.columns = ['date','GARCH_F1','ticker']
    return options.merge(garch_vol, how = 'left')

