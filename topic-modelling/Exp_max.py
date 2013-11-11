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

def Expectation(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs,z_init,gibbs_itrn):
    # Compute each of the probability coefficients with the hyper
    # parameters fixed constant. Done using gibbs sampling. 
    
    print('Running Expectation Module')
    z_count, p_md, E_theta, E_m_d_theta = \
        gs.gibbs_sampling(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs,z_init,gibbs_itrn)    
    
    return (p_md,E_m_d_theta,z_count)
    
def Maximization(p_md, E_log_theta, alpha, z_count, model_type, A, gibbs_itrn, beta_smooth,itrn):
    
    # Pi maximization
    print('-Maximization- Running')

    K = alpha.shape[0]
    no_of_topics = alpha.shape[1]
    
    Pi = np.array(p_md.sum(axis=0))
    Pi = Pi/Pi.sum()
    Pi,mask = regularize(Pi,K)
    
    # alpha maximization
    E_d = E_log_theta.sum(axis=0)
    p_d = p_md.sum(axis=0)
    bar_p = np.empty([K,no_of_topics])    
    alpha_new = np.empty([K,no_of_topics])
    
    for i in range(K):
        if(mask[i] == 0):
            bar_p[i] = E_d[i]/p_d[i]

    alpha_old = alpha
    for m in range(1000):
        salpha_old = alpha_old.sum(axis=1) #Summing up along the topics    
        for k in range(K):
            # Alpha's are not updated for p_d[i] == 0
            if(mask[i] == 0): 
                alpha_new[k] = digamma(salpha_old[k]) + bar_p[k]
            else:
                alpha_new[k] = digamma(alpha_old[k])                    
        alpha_new = [[inv_d.inverse_digamma(x) for x in row] for row in alpha_new]
        if(abs((alpha_new - alpha_old).max()) < 10**(-6)): # Convergence Condition
            break
        else:
            alpha_old = np.array(alpha_new)    

	# Beta Maximization
    print(model_type)
    print(itrn)
    if((model_type != 0) & (itrn > 20)):
        # Computing p_z from the z_count
        print("Maximizing Beta");
        B = np.zeros([z_count.shape[1],z_count.shape[2]]) + beta_smooth
        for d in range(z_count.shape[0]):
            B += z_count[d]
        B = B/gibbs_itrn
        B = [[remove_zero(x) for x in row] for row in B]    
        B = np.array(B)        
        for i in range(B.shape[1]):
            A[:,i] = B[:,i]/B[:,i].sum()
        
    print('--Done')
    return Pi, np.array(alpha_new), A

def regularize(Pi,K):
    mask = np.zeros([Pi.shape[0]])
    pi_sum = 1;
    part_sum = 0;
    for i in range(K):
        if(Pi[i] < (1/(10*K))):
            Pi[i] = 1/(10*K)
            mask[i] = 1
            pi_sum = pi_sum - 1/(10*K)
        else:
            part_sum = part_sum + Pi[i]
            
    for i in range(K):
        if(mask[i] == 0):
            Pi[i] = (Pi[i]*pi_sum)/part_sum
    
    return (Pi,mask)
        
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

