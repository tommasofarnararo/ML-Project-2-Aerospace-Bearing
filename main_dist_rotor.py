import numpy as np
from loading_datasets import *
from feature_engineering import *
from linear_regression import *
import tqdm
import matplotlib

#%% loading data
X, Y, Fx, Fy = load()
N = X.shape[0]
D = Y.shape[1]

#%% setting parameters
dd_min = 0
dd_max = 400
M = distance_from_rotor(X, Y)
k_fold = 5

#%% LS REGRESSION
Fx_tilde, losses_x = LEAST_SQUARES_REGRESSION(dd_min, dd_max, Fx, M, k_fold)
Fy_tilde, losses_y = LEAST_SQUARES_REGRESSION(dd_min, dd_max, Fy, M, k_fold)

#%% plots
d_to_plot = np.arange(dd_min, dd_max+1)
plt.plot(d_to_plot, np.sum(losses_x, axis=1)/2001, color='b')#, marker='*')
plt.xlabel("d")
plt.ylabel("loss_x")
plt.title("Test Loss for the delay parameter")
plt.ylim(0,2)

d_to_plot = np.arange(dd_min, dd_max+1)
plt.plot(d_to_plot, np.sum(losses_y, axis=1)/2001, color='r')#, marker='*')
plt.xlabel("d")
plt.ylabel("loss_y")
plt.title("Test Loss for the delay parameter")
# plt.ylim(0,5)

#%% plot the error for d = 200
time_steps = np.arange(2001)
plt.plot(time_steps, losses_x[200,:], color='b')#, marker='*')
# plt.ylim(0,0.01)


