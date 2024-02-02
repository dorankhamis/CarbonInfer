import xarray as xr
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SQRT2PI = np.sqrt(2*np.pi)

def score(y_pred, y_true):
    u = ((y_true - y_pred)**2).sum() # residual sum of squares
    v = ((y_true - y_true.mean())**2).sum() # total sum of squares
    r2 = 1. - u/v
    return r2

def resample_sd(x):
    # error in the mean, assuming uncorrelated values
    # (which is likely not true.)
    # https://www.public.asu.edu/~laserweb/woodbury/classes/chm467/bioanalytical/data%20reduction%20and%20error%20analysis/data%20reduction%20and%20error%20analysis.html
    # in correlated case, sigma_X+Y = sqrt(sigma_X^2 + sigma_Y^2 + 2 rho sigma_X sigma_Y)
    # where rho is the correlation, or for multiple variables the covariance matrix
    # https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
    # https://en.wikipedia.org/wiki/Variance#Sum_of_correlated_variables    
    return np.sqrt(np.sum(x*x) / len(x))

def gaussian_loglikelihood(x, mu, sigma):
    return - 0.5 * ((x-mu)/sigma)**2 - np.log(sigma * SQRT2PI)
