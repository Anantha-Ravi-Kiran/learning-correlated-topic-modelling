from __future__ import division
from itertools import groupby
from scipy.sparse import *
from scipy import *
from scipy.special import digamma
import sys
import scipy.io
import numpy as np
import pdb
import operator
import gibbs_sampling as gs
import inverse_digamma as inv_d

def Expectation(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs):
    # Compute each of the probability coefficients with the hyper
    # parameters fixed constant. Done using gibbs sampling. 
    
    print('Running Expectation Module')
    z_count, p_md, E_theta, E_m_d_theta = \
        gs.gibbs_sampling(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs)    
    
    return (p_md,E_m_d_theta)
    
def Maximization(p_md, E_log_theta, alpha):
    
    # Pi maximization
    print('- Maximization - Running')
    
    Pi = p_md.sum(axis=0)
    Pi = Pi/Pi.sum()

    # alpha maximization
    K = alpha.shape[0]
    no_of_topics = alpha.shape[1]
    E_d = E_log_theta.sum(axis=0)
    p_d = p_md.sum(axis=0)
    bar_p = np.empty([K,no_of_topics])    
    alpha_new = np.empty([K,no_of_topics])

    for i in range(K):
        bar_p[i] = E_d[i]/p_d[i]

    alpha_old = alpha
    for m in range(1000):
        salpha_old = alpha_old.sum(axis=1) #Summing up along the topics    
        for k in range(K):
            alpha_new[k] = digamma(salpha_old[k]) + bar_p[k]
        alpha_new = [[inv_d.inverse_digamma(x) for x in row] for row in alpha_new]
        if(abs((alpha_new - alpha_old).max()) < 10**(-6)): # Convergence Condition
            break
        else:
            alpha_old = np.array(alpha_new)    

    print('--Done')
    return Pi, np.array(alpha_new)

def pick_top_index(alpha,top_few):
    alpha_top_index = []
    for x in alpha:
        top_index = []
        m = sorted(enumerate(x),key=operator.itemgetter(1),reverse=True)
        for i in m[:top_few]:
            top_index.append(i)
        alpha_top_index.append(top_index)
    
    return np.array(alpha_top_index)
        
def count_occurence(a,K):
    count_ = np.zeros(K)
    for i in a:
        count_[i] = count_[i] + 1 
    return count_
    
def remove_zero(x):
    if (x < 10**(-5)):
        return 10**(-5)
    else:
        return x

