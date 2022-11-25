import numpy as np
from loading_datasets import *
from feature_engineering import *
from linear_regression import *
import tqdm

#%% loading data
X, Y, Fx, Fy = load()
N = X.shape[0]
D = Y.shape[1]

#%% computing the spatial steps
diff_X = velocity(X)
diff_Y = velocity(Y)

#%% adding the starting point
diff_X = np.c_[X[:,0], diff_X]
diff_Y = np.c_[Y[:,0], diff_Y]
M_diff = np.ravel([diff_X, diff_Y], 'F').reshape(N, 2*D)

#%% adding the distance from the rotor
dist_rotor = distance_from_rotor(X,Y)
M_dist = M_diff
j = 0
for i in range(M_diff.shape[1]):
    if ((i+1)%2 == 0):
        M_dist = np.insert(M_dist, i, dist_rotor[:, j], axis=1)
        j=j+1

#%% adding the distance from the rotor
# vel = velocity(X, Y)

#%% LS REGRESSION
dd_min = 1
dd_max = 200
F = Fx
M = distance_from_rotor(X, Y)
k_fold = 5
Fx_tilde, losses = LEAST_SQUARES_REGRESSION(dd_min, dd_max, F, M, k_fold) 

#%%
d_to_plot = np.arange(dd_min, dd_max)
plt.plot(d_to_plot, losses, color='b', marker='*')
plt.xlabel("d")
plt.ylabel("loss")
plt.title("Test Loss for the delay parameter")
# leg = plt.legend(loc="upper left", shadow=True)
# leg.draw_frame(False)
# plt.savefig("lambdas_for_RR_{deg}".format(deg=degree))

