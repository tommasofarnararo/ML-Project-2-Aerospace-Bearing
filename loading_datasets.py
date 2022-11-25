import numpy as np

def load():
    """ Load training dataset and convert 'b' into 0 and 's' into 1
        Remove the 24th categorical feature
        Return the dataset and the removed feature """
   
    X = np.genfromtxt("X.csv", delimiter=",")
    Y = np.genfromtxt("Y.csv", delimiter=",")
    
    
    Fx = np.genfromtxt("FX.csv", delimiter=",")
    Fy = np.genfromtxt("FY.csv", delimiter=",")
    
    return X.T, Y.T, Fx.T, Fy.T
