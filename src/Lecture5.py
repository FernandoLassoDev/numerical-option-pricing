# Written by Fernando Lasso to replicate the examples in lectures at ESADE


from math import exp
import numpy as np
# A node in the binomial tree
class Node(object):
    
    # Node contains theoretical value at time t
    def __init__(self, S, t, parent = None):
        self.S = S
        self.t = t
        self.parent = parent 
        self.up_child = None
        self.down_child = None 
        
    # Create children of node
    def create_children(self,u, d, dt, nephews):
        if len(nephews)<1:
            self.up_child   = Node(self.S * u, self.t +dt, self)
            self.down_child = Node(self.S * d, self.t +dt, self)
        else:
            self.up_child   = nephews[-1]
            self.down_child = Node(self.S * d, self.t +dt, self)
        
      
    # Return children of node
    def get_children(self,up = True, down = True):
        children = []
        if up: children.append(self.up_child)   
        if down: children.append(self.down_child) 
        return children
    
    # Bottom nodes are return if exercised
    # Other nodes are weighted expected value of children
    def calculate_value(self,K, r, dt, p, call = True, european = True):
        c_to_p = call*2-1
        if self.down_child is None:
            self.value = max(c_to_p * (self.S - K), 0)
        else:
            fu = self.up_child.value
            fd = self.down_child.value
            self.value = exp(-r*dt) * (p * fu + (1-p) * fd)
        
            if not european:
                self.value = max(self.value, c_to_p * (self.S - K))

    def flush():
        self.value = None
        
    def __repr__(self):
        return "Node()"
    
    def __str__(self):
        out = "Node at time {time} with value {stock}"
        return out.format(time = round(self.t,4), stock = round(self.S,4))
    
# The tree consists of nodes of potential stock prices in the future
class Binomial_Tree(object):
    
    # Iinitialize with current stock price, desired T, vol and n
    def __init__(self, S, T, o, n):
        """
        S = Current stock price
        T = Years to maturity
        o = Implied volatility
        n = Steps in the tree
        """
        self.dt = T/n
        self.o = o
        self.u = exp(o*np.sqrt(self.dt))
        self.d = 1/self.u
    
        self.build_tree(S, self.dt, self.u, self.d, n)
        
    # Build the tree
    def build_tree(self,S, dt, u, d, n):
        # Set origin
        self.nodes = [[Node(S, dt, None)]]
        
        # Build tree
        for i in range(n):
            level_nodes = []
            for n in self.nodes[i]:
                n.create_children(u,d, dt, level_nodes)
                
                if len(level_nodes) >0:
                    children = n.get_children(up = False)
                else:
                    children = n.get_children()
                level_nodes.extend(children)
            self.nodes.append(level_nodes)
      
    
    # calculate present value for given strike and rf rate
    def present_value(self, K, r, call = True, european = True, **kwargs):
        """
        K = Strike price
        r = Risk-free interest rate
        call = whether it is call or put
        """
        a = get_growth_factor(r, self.dt, **kwargs)
        p = (a-self.d) / (self.u-self.d) 
    
        # From last to first level, calculate value of tree
        for level in self.nodes[::-1]:
            for node in level:
                node.calculate_value(K, r, self.dt, p, call, european)
        
        # Return value of origin node
        return self.nodes[0][0].value

def get_growth_factor(r,dt, q = 0, rf = 0, underlying = ""):

    if underlying == "cont_div_stock":
        return exp((r-q)*dt)
    if underlying == "currency":
        return exp((r-rf)*dt)
    if underlying == "future":
        return 1

    return exp(r*dt)
