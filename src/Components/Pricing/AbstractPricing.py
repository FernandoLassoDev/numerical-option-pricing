from abc import ABC, abstractmethod

# Abstract class for volatility calculation
class AbstractPricing(ABC):
    
    @abstractmethod
    def option_pricing(self):
        """
        S = Current stock price
        K = Strike price
        t = Time until option exercise (years to maturity)
        r = Risk-free interest rate
        v = Implied volatility
        """
        pass
