import numpy as np
from loading_datasets import *
from feature_engineering import *
from linear_regression import *
import tqdm

#%% loading data
X, Y, Fx, Fy = load()
N = X.shape[0]
D = Y.shape[1]

#%% setting parameters
dd_min = 0
dd_max = 200
F = Fx
M = distance_from_rotor(X, Y)
k_fold = 5

#%% LS AUTOREGRESSIVE
F_tilde_x, F_tilde_y, losses_x, losses_y = AUTOREGRESSIVE_TRUE(dd_min, dd_max, X, Y, Fx, Fy, k_fold)

#%%
d_to_plot = np.arange(dd_min, dd_max+1)
plt.plot(d_to_plot, np.sum(losses_x, axis=1)/2001, color='b')#, marker='*')
plt.xlabel("d")
plt.ylabel("loss_x")
plt.title("Test Loss for the delay parameter")
plt.ylim(0,0.01)

d_to_plot = np.arange(dd_min, dd_max+1)
plt.plot(d_to_plot, np.sum(losses_y, axis=1)/2001, color='r')#, marker='*')
plt.xlabel("d")
plt.ylabel("loss_y")
plt.title("Test Loss for the delay parameter")
#plt.ylim(0,2)

#%% plot the error for d = 200
time_steps = np.arange(2001)
plt.plot(time_steps, losses_x[150,:], color='b')#, marker='*')
plt.ylim(0, 0.1)
