from abc import ABC, abstractmethod

# Abstract class for volatility calculation
class AbstractVol(ABC):
    
    @abstractmethod
    def historical_volatility(self):
        pass



class TraditionalVol(AbstractVol):
  
    def historical_volatility(quotes, days): 
        "Return the annualized stddev of daily log returns of picked stock"
        logreturns = np.log(quotes / quotes.shift(1))
        # return square root * trading days * logreturns variance

        return np.sqrt(252*logreturns.var()) 
