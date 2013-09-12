from __future__ import division
from numpy.random.mtrand import dirichlet
from scipy.sparse import *
from scipy import *
import numpy as np
import math
import operator
import pdb
import threading
import multiprocessing as mp
import time
import functools
import sys
import time
import ctypes

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def remove_zero(x):
    if (x == 0):
        return sys.float_info.min
    else:
        return x

# Considering only proportionality and ignoring the constant term
def dirichlet_log_prob(x, alpha):
    p = 0
    for i in range(len(alpha)):
        p += (alpha[i]-1.0) * math.log(x[i])
    return (p - log_dirichlet_const(alpha))

def compute_log_sum(ind_log):
    x = ind_log[0]
    for i in range(ind_log.shape[0]-1):
        y = ind_log[i+1]
        x = x + math.log(math.exp(y-x)+1)
    return x
    
def log_dirichlet_const(alpha):
    B = 0
    for a in alpha:
        B += math.lgamma(a)
    return (B - math.lgamma(sum(alpha)))

def count_topics(z_d, word_index, no_of_topics):
    topics_count = np.zeros(no_of_topics)
    for i in word_index:
        topics_count[z_d[i]] += 1 
    return topics_count

def log_np_array(Pi):
    log_Pi = np.empty(Pi.shape[0])
    for i in range(Pi.shape[0]):
        log_Pi[i] = math.log(Pi[i])
    return log_Pi

def gibbs_sampling(alpha,Pi,A,word_list,doc_list,vocabSize,numdocs):
        
    print('-Gibbs Sampling')
    no_of_itr = 1000
    no_of_topics = A.shape[1] 
    X = 50 # Burning-in 
    K = alpha.shape[0] # No of mixture components

    max_procs = mp.cpu_count()

    # Creating shared memory for multiprocesses and converting them
    # to np.array types
    p_md_base = mp.Array(ctypes.c_double, numdocs*K)
    p_md = np.ctypeslib.as_array(p_md_base.get_obj())
    p_md = p_md.reshape(numdocs,K)

    E_theta_base = mp.Array(ctypes.c_double, numdocs*no_of_topics)
    E_theta = np.ctypeslib.as_array(E_theta_base.get_obj())
    E_theta = E_theta.reshape(numdocs,no_of_topics)

    E_m_d_theta_base = mp.Array(ctypes.c_double, numdocs*K*no_of_topics)
    E_m_d_theta = np.ctypeslib.as_array(E_m_d_theta_base.get_obj())
    E_m_d_theta = E_m_d_theta.reshape(numdocs,K,no_of_topics)

    z_count_base = mp.Array(ctypes.c_int, numdocs*vocabSize*no_of_topics)
    z_count = np.ctypeslib.as_array(z_count_base.get_obj())
    z_count = z_count.reshape(numdocs,vocabSize,no_of_topics)

    # No copy was made
    assert z_count.base.base is z_count_base.get_obj()
    assert p_md.base.base is p_md_base.get_obj()
    assert E_theta.base.base is E_theta_base.get_obj()
    assert E_m_d_theta.base.base is E_m_d_theta_base.get_obj()

    procs = []
    for proc_id in range(max_procs):
        p = mp.Process(target = gibbs_sep_doc,
                       args = (alpha,Pi,A,word_list[proc_id],doc_list[proc_id],
                       z_count,p_md,E_theta,E_m_d_theta,
                       X,no_of_itr,vocabSize,proc_id))

        p.start()
        procs.append(p)

    # Wait for all procs to complete
    for p in procs:
        p.join()

    print('--Done')
    return (z_count, p_md, E_theta, E_m_d_theta)
 
# This function is called only for a set of documents among the doc_list.
def gibbs_sep_doc(alpha,Pi,A,word_doc,doc_list,\
                  z_count_all,p_md_all,E_theta_all,E_m_d_theta_all,\
                  X,no_of_itr,vocabSize,proc_id):

    no_of_topics = A.shape[1] 
    K = alpha.shape[0] # No of mixture components
    multi_rand = np.random.multinomial(1,[1/no_of_topics]*no_of_topics,\
                                        size=(vocabSize))
    z_init = np.array([mat.argmax() for mat in multi_rand])
    theta = np.ones([no_of_topics])*(1/no_of_topics)

    numdocs = doc_list.shape[0]
    p_M = np.zeros([K])
    E_theta = np.zeros([no_of_topics])
    E_m_d_theta = np.zeros([K,no_of_topics])
    z_count = np.zeros([vocabSize,no_of_topics])
        
    # iterating through every document  
    idx = 0
    for doc_index in doc_list:
        word_indices = word_doc[idx]
        z_d = z_init
        E_theta.fill(0)
        E_m_d_theta.fill(0)
        p_M.fill(0)
        z_count.fill(0)
        for i in range(no_of_itr + X):    

            # Sampling the mixture component
            p_theta_gm = np.empty(K)
            for k in range(K):
                p_theta_gm[k] = dirichlet_log_prob(theta,alpha[k])
            log_p_md = log_np_array(Pi) + p_theta_gm
            norm_log_pmd = (log_p_md - compute_log_sum(log_p_md))
            p_md = np.array([math.exp(m) for m in norm_log_pmd])
            M = np.random.multinomial(1,p_md,size=1).argmax()

            # Sampling theta
            alpha_d = alpha[M]
            topics_count = count_topics(z_d, word_indices, no_of_topics)
            alpha_p = alpha_d + topics_count
            local_theta = dirichlet(alpha_p, size=1)
            local_theta = np.array([remove_zero(m) for m in local_theta[0]])
            theta = local_theta/local_theta.sum()
            
            for w_index in word_indices:
                # iterating through every word in document to sample z
                p_zd = A[w_index] * theta
                p_zd = p_zd/p_zd.sum()
                word_topic = np.random.multinomial(1,p_zd,size=1).argmax()
                z_d[w_index] = word_topic

                if (i >= X):                  
                    z_count[w_index, word_topic] += 1 
                
            # Saving only theta and M samples from every iteration 
            # only burning-in
            if (i >= X):
                p_M[M] += 1 
                E_theta += theta 
                E_m_d_theta[M] += log(theta)

        idx += 1

        # Communicating the array's back to the parent process
        z_count_all[doc_index] = z_count
        p_md_all[doc_index] = p_M/no_of_itr
        E_theta_all[doc_index] = E_theta/no_of_itr
        E_m_d_theta_all[doc_index] = E_m_d_theta/no_of_itr
    
