from __future__ import division
from scipy.special import polygamma
from scipy.special import digamma
from math import exp
import math
import pdb

# Reference : Minka - Estimation of dirichlet parameters
# Using newton raphson method

def inverse_digamma(y):

    gammac = -digamma(1)
    
    # Initialization of x
    if y >= -2.22: 
        x_old = exp(y) + 0.5
    else:
        x_old = -(1/(y+gammac))
        
    # Limiting iteration to 5 since initialization is pretty good.    
    for i in range(5):    
        x_new = x_old - ((digamma(x_old) - y)/polygamma(1,x_old))
        x_old = x_new
    
    return x_new
