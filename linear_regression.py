import numpy as np
from feature_engineering import *

import matplotlib.pyplot as plt

######################### UTILITIES LINEAR REGRESSION ########################
def least_squares(y, tx):
    """ The Least Squares algorithm (LS) 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
        Returns:
            w: shape=(D, ) optimal weights
            mse: scalar(float) """
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    mse = compute_mse(y, tx, w)
    
    return w, mse

def ridge_regression(y, tx, lambda_):
    """ The Ridge Regression algorithm (RR)
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            lambda_: scalar(float) penalization parameter
        Returns:
            w: shape=(D, ) optimal weights
            loss: scalar(float) """
            
    N = y.size
    D = tx[0,:].size
    I = np.identity(D)
    A = np.dot(tx.T,tx) + (lambda_*2*N)*I
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A, b)
    
    loss = compute_loss(y, tx, w)
    
    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and
        the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching 
        elements from `y` and `tx`. Data can be randomly shuffled to avoid 
        ordering in the original data messing with the randomness of the minibatches.
        Example of use :
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
                <DO-SOMETHING> """
    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    return

def compute_mse(y, tx, w):
    """ Calculate the mse for the vector e = y - tx.dot(w) """
    
    return 1 / 2 * np.mean((y-tx.dot(w)) ** 2)

def compute_mae(y, tx, w):
    """ Calculate the mae for vector e = y - tx.dot(w) """
    
    return np.mean(np.abs(y-tx.dot(w)))

def compute_loss(y, tx, w):
    """ Calculate the loss using either MSE or MAE 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            w: shape=(2,). The vector of model parameters.
        Returns:
            loss: scalar(float), corresponding to the input parameters w """
    
    return compute_mse(y, tx, w)


def build_poly(x, degree):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree """
    phi_x = np.zeros((x.size, degree+1))
    for i in range(len(x)):
        curr_row = np.array([x[i]**deg for deg in range(degree+1)])
        phi_x[i] = curr_row
        
    return phi_x

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold 
        Args:
            y: shape=(N,)
            k_fold: K in K-fold, i.e. the fold num
            seed: the random seed
        Returns:
            ret: shape=(k_fold, N/k_fold) with the data indices for each fold """
    num_row = y.shape[0]
    interval = int(num_row / k_fold) # Here it computes the number of intervals
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation_tx_LS(y, tx, k_indices, k):
    """ Return the loss of ridge regression for a fold corresponding to k_indices
        The regression matrix tx is given as an input 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            k_indices: 2D array returned by build_k_indices()
            k: scalar, the k-th fold
            lambda_: scalar, cf. ridge_regression()
        Returns:
            loss_tr: scalar(float), rmse = sqrt(2 mse) of the training set
            loss_te: scalar(float), rmse = sqrt(2 mse) of the testing set """

    # get k'th subgroup in test, others in train
    k_fold = len(k_indices)
    
    y_test = y[k_indices[k]]
    tx_test = tx[k_indices[k],:]
    
    ind_train = []
    ind_train = np.append(ind_train, k_indices[np.arange(k_fold)!=k])
    ind_train = [int(ind_train[i]) for i in range(len(ind_train))]
                 
    y_train = y[ind_train]
    tx_train = tx[ind_train,:]
    
    # ridge regression
    w_k = least_squares(y_train, tx_train)[0]
    
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*compute_mse(y_train, tx_train, w_k))
    loss_te = np.sqrt(2*compute_mse(y_test, tx_test, w_k))
    
    return loss_tr, loss_te, w_k


def cross_validation_demo_tx_LS(y, tx, k_fold):
    """ Cross validation over regularisation parameter lambda 
        The regression matrix tx is given as an input 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            degree: integer, degree of the polynomial expansion
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_lambda : scalar, value of the best lambda
            best_rmse : scalar, the associated root mean squared error for the best lambda """
    
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    
    r_tr = []
    r_te = []
    ww = []
    ww = np.asarray(ww)
    
    for k in range(k_fold): # we do this to perform the training using all the data
        tr, te, w_k = cross_validation_tx_LS(y, tx, k_indices, k) 
        r_tr.append(tr)  
        r_te.append(te)
        if k==0:
            ww = w_k
        else:
            ww = np.c_[ww, w_k]
    
    rmse_tr = np.mean(r_tr)
    rmse_te = np.mean(r_te)
    w = np.mean(ww, axis=1)
    
    return rmse_tr, rmse_te, w

def best_degree_selection_x_lin(x, y, degrees, k_fold, lambdas, seed = 1):
    """ Cross validation over regularisation parameter lambda and degree 
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_degree: scalar, value of the best degree
            best_lambda : scalar, value of the best lambda
            best_rmse : scalar, the associated root mean squared error for the best lambda """
            
    # define lists to store the loss of training data and test data
    best_lambdas = []
    best_rmses = []
    
    for deg in degrees:
        
        best_lambdas.append(cross_validation_demo_x_lin(x, y, deg, k_fold, lambdas)[0])
        best_rmses.append(cross_validation_demo_x_lin(x, y, deg, k_fold, lambdas)[1])
    
    best_ind = np.argmin(best_rmses)
    best_degree = degrees[best_ind]
    best_lambda = best_lambdas[best_ind]
    best_rmse = best_rmses[best_ind]
    
    return best_degree, best_lambda, best_rmse


def LEAST_SQUARES_REGRESSION(dd_min, dd_max, F, M, k_fold):
    
    N = M.shape[1]
    D = 2001
    f = int(N/D) # number of features associated to each time stamp
    losses = np.zeros(dd_max-dd_min)

    for d in range(dd_min, dd_max+1):
        
        F_tilde = np.zeros(F.shape)
        loss = 0
        
        for j in range(D):
            
            if (j+1) - d < 0:
                tx = M[:, : f*(j+1)]
            else:
                tx = M[:, f*(j+1-d) : f*(j+1)]
               
            y = F[:, j]
            # tx = standardize(tx)[0]
            rmse_tr, rmse_te, w = cross_validation_demo_tx_LS(y, tx, k_fold)
            prediction = np.dot(tx, w)
            F_tilde[:, j] = prediction
            curr_loss = compute_loss(y, tx, w)
            loss += curr_loss
        # loss/= D 
        
        losses[d-dd_min-1] = loss
        
    return F_tilde, losses

