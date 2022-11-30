import numpy as np
import pandas as pd
from loading_datasets import *
from feature_engineering import *
from linear_regression import *
import tqdm

#%% loading data
X, Y, Fx, Fy = load()
N = X.shape[0]
D = Y.shape[1]
dist = distance_from_rotor(X, Y)

#%%
dd = 50
df = np.zeros(dd*5)

for t in range(500):
    
    for n in range(N-dd):
        
        row = np.zeros(dd*5)
    
        for d in range(dd):
            
            row[5*d] = X[t, dd+n]
            row[5*d + 1] = Y[t, dd+n]
            row[5*d + 2] = dist[t, dd+n]
            row[5*d + 3] = Fx[t, dd+n]
            row[5*d + 4] = Fy[t, dd+n]
            
        df = np.r_[df,row]
            
        
        
        
        
        
