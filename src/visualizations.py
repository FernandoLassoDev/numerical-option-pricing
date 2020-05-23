import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
import random
import seaborn as sns

def plot_strike_price(data, ticker, flag):
	s = data.loc[(data.ticker == ticker) & (data.cp_flag == flag),["date", "strike_price","Price","midpoint"]]
	s = s.iloc[np.sort(random.sample(range(1, len(s)), int(len(s)/20))),:]
	s = s.sort_values(by='date', ascending=True)
	s.index = s.date

	fig, ax = plt.subplots()

	scatter = ax.scatter(s.index,s['strike_price']/1000,zorder = 1, c = s.midpoint, cmap='PuBuGn')
	legend1 = ax.legend(*scatter.legend_elements(), title="Option Price")
	ax.add_artist(legend1)
	ax.plot(s['Price'],zorder=2, color = 'r', linewidth=2.0) 
	plt.title(ticker + " (" + flag +")")
	plt.xticks(rotation='vertical')
	plt.show()

# Read data
def read_daily_stock(path,dt = 'Time'):
	s = pd.read_csv(path)
	s.index = pd.to_datetime(s[dt])
	return s.resample('D').mean().dropna().Last

def load_data(plot = True):
	options = pd.read_csv("Data/options.csv")
	options.date = pd.to_datetime(options.date)
	options = options[options.date > pd.to_datetime("2016-01-01")]
	stocks = pd.DataFrame({'WMT' : read_daily_stock("Data/WMT.csv"),
						   'AAPL': read_daily_stock("Data/AAPL.csv"),
	                       'JPM' : read_daily_stock("Data/JPM.csv"),
	                       'DIS' : read_daily_stock("Data/DIS.csv")
	                       })

	dividends = pd.read_csv("Data/dividend.csv")
	dividends.index = pd.to_datetime(dividends.date)
	dividends = dividends[['TICKER','DIVAMT']]
	risk_free = pd.read_csv("Data/risk-free.csv")
	risk_free.index = pd.to_datetime(risk_free.date)
	risk_free = risk_free.rf


	# Create one dataframe with all necessary parameters
	options = options.merge(risk_free, how = 'left', left_on = 'date', right_index = True)
	options['T'] = (pd.to_datetime(options.exdate) - options.date).dt.days 
	options = options.merge(
	    pd.melt(stocks.reset_index(), id_vars='Time', var_name='Stock', value_name='Price'),
	    how = 'left', left_on = ['date','ticker'], right_on = ['Time','Stock'])
	del options['Stock']
	options['midpoint'] = (options.best_bid + options.best_offer)/2

	#convert dividend to continous rate in option pricing
	yearly_div = dividends.resample("Y").sum()
	yearly_div['year'] = yearly_div.index.year
	options['year'] = options.date.dt.year
	options = options.merge(yearly_div, how = 'left', on = 'year')

	options['q'] = np.log(1+options.DIVAMT/options.Price)

	if plot:
		risk_free.plot()
		plt.title("Risk Free Rate")
		plt.show()
		plt.clf()

		stocks.plot()
		plt.title("Stock Price Development")
		plt.show()
		plt.clf()

		ax = dividends.pivot(columns='TICKER').dropna(how='all').plot(kind='bar')
		start, end = ax.get_xlim()
		ax.xaxis.set_ticks(np.arange(start, end, 10))
		plt.title("Dividend payouts")
		plt.show()

		options.loc[(options.ticker == "AAPL") & (options.cp_flag == "C"),["midpoint"]].hist()
		plt.title("Apple call prices")
		plt.show()

		print(options.head())

	return options, stocks

def plot_prediction_error(df, x_var = "Actual", y_var = "Predicted"):
	g = sns.pairplot(df, x_vars=[x_var], y_vars=[y_var], 
		hue="Ticker", height=5, aspect=2, plot_kws={'alpha':0.5})
	g.fig.suptitle("Predicted vs Actual prices")
	plt.show()

def plot_error_dist(diffcol, tickercol):
	sns.set(rc={"figure.figsize": (12, 6)})
	for t in ["WMT","AAPL","JPM","DIS"]:
		sns.distplot(diffcol[tickercol==t]).set_title("% difference distribution")


def write_errors(title, differences, actual):
	stats = dict()
	print("----------------------------------")
	print(title)
	print("----------------------------------")
	stats['mse'] = np.mean(differences**2)
	print("Mean Squared Error:      ", stats['mse'])

	stats['rmse'] = np.sqrt(stats['mse'])
	print("Root Mean Squared Error: ", stats['rmse'])

	stats['mae'] = np.mean(abs(differences))
	print("Mean Absolute Error:     ", stats['mae'])

	stats['mpe'] = 100 *np.sqrt(stats['mse'])/np.mean(actual)
	print("Mean Percent Error:      ", stats['mpe'])
	print("")
