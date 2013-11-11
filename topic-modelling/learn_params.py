from __future__ import division
import IPython
from itertools import groupby
from scipy.sparse import *
from scipy import *
from scipy.special import digamma
import multiprocessing as mp
import sys
import scipy.io
import numpy as np
import pdb
import time
import operator
import gibbs_sampling as gs
import inverse_digamma as inv_d
import Exp_max as em
import os
import random

# K - No of dirichlet mixtures which is passed as an argument
# A = matrix(zeros((V,K)))
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

if len(sys.argv) < 8:
    print("Usage: <input_doc> <A_matrix> <top_words> <output_file> <num_comp> <num_itrn> <corpus_string> <gibbs_itrn> <model_type> <seed>")
    sys.exit()
    
input_docs_file=sys.argv[1]
A_file=sys.argv[2]
top_words_in=sys.argv[3]
output_file=sys.argv[4]
K = int(sys.argv[5])
num_itrn = int(sys.argv[6])
corpus = sys.argv[7]
gibbs_itrn = int(sys.argv[8])
model_type=int(sys.argv[9])
beta_smooth=float(sys.argv[10])
seed_num=int(sys.argv[11])

random.seed(seed_num)

if(model_type == 3):
    noise_prop = float(sys.argv[12])

dirname,f_temp = os.path.split(sys.argv[2])

# Reading the input files
input_docs = scipy.io.loadmat(input_docs_file)['M']
A = np.loadtxt(A_file) 
topwords_file = open(top_words_in, "r")

# Initialization
vocabSize = input_docs.shape[0]
numdocs = input_docs.shape[1]
no_of_topics = A.shape[1]

# Read Top words from the file
topwords = []
try:
	for line in topwords_file.read().split('\n'):
		topwords.append(line)
finally:
	topwords_file.close()

print("No of topics: %d" %(no_of_topics))

# Setting zero values in A to smallest possible float and 
# Renormalization    
A = [[em.remove_zero(x) for x in row] for row in A]    
A = np.array(A)
for i in range(A.shape[1]):
	A[:,i] = A[:,i]/A[:,i].sum()

if (model_type == 1):
    # Initializing A matrix with uniform distribution
    A = [[(random.random() + 1.0/100) for x in row] for row in A]    
    A = np.array(A)
    for i in range(A.shape[1]):
        A[:,i] = A[:,i]/A[:,i].sum()
elif (model_type == 3):
    # Init_noise_mode
    B = np.empty_like(A)
    B[:] = A
    B = [[(random.random() + 1.0/100) for x in row] for row in B]    
    B = np.array(B)
    for i in range(B.shape[1]):
        B[:,i] = B[:,i]/B[:,i].sum()
    A = np.array((noise_prop*B) + ((1-noise_prop)*A))
    for i in range(A.shape[1]):
        A[:,i] = A[:,i]/A[:,i].sum()

# Splitting the documents for performing gibbs sampling on different
# processors     
max_procs = mp.cpu_count()
docs_list = np.array(range(numdocs))
docs_per_proc = int(numdocs/max_procs)
left_docs = numdocs - max_procs*int(numdocs/max_procs)
doc_list = []
i = 0
while (max_procs > 0):
	doc_list.append(docs_list[i:i+docs_per_proc+(left_docs>0)])
	i = i+docs_per_proc+(left_docs>0)
	left_docs -= 1        
	max_procs -= 1
#doc_list = np.array(doc_list)

# Maximum probability topic assignment for each word
max_topic_assgn = np.empty(vocabSize)
for i in range(vocabSize):
    max_topic_assgn[i] = A[i,:].argmax()
max_topic_assgn = np.array(max_topic_assgn)

# Collecting words in each document for multiprocessing efficiently 
# Also does gibbs sampling z initialization
word_list = []
z_init = []
for docs in doc_list:
    word_doc = []
    z_doc = []
    for doc_index in docs:
        start = input_docs.indptr[doc_index]
        end = input_docs.indptr[doc_index + 1]
        word_indices = input_docs.indices[start:end]
        word_doc.append(word_indices)    
        z_doc.append(max_topic_assgn[word_indices])
    word_list.append(word_doc)
    z_init.append(z_doc)
    #word_list.append(np.array(word_doc))
#word_list = np.array(word_list) #YONI COMMENTED OUT HERE

# Writting the report to the output - bufsize = 0 for unbuffering.
#out_file_name = "%s_%s_%d.html" %(output_file,corpus,no_of_topics)
out_alpha = "%s/%s_%s_%d.txt" %(dirname,"alpha",corpus,no_of_topics)
out_pi = "%s/%s_%s_%d.txt" %(dirname,"pi",corpus,no_of_topics)
out_beta = "%s/%s_%s_%d.txt" %(dirname,"topics",corpus,no_of_topics)

#f = open(out_file_name,'w')
f_alpha = open(out_alpha,'w')
f_pi = open(out_pi, 'w')
f_beta = open(out_beta,'w')

# Size of the vocabulary written into the A matrix first line
f_beta.write("%d\n"%(vocabSize))

#for K in range(1,int(no_of_topics/5) + 1):
	# Initializing the dirichlet mixtures with equal probability

Pi = np.ones(K)*(1/K)     
alpha = np.ones([K,no_of_topics])*(1/no_of_topics)

alpha = np.empty([K,no_of_topics])
alpha = [[random.random() for x in row ] for row in alpha]
alpha = np.array(alpha)
alpha_sum = alpha.sum(1)
for row in range(K):
    alpha[row] = 0.1 * alpha[row]/alpha_sum[row]

# Pending - Initialize alpha using the moments from the data instead.
p_md,E_log_theta,z_count = em.Expectation(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs,z_init,gibbs_itrn)
#while True:
for i in range(num_itrn):
    print("iteration",i)
    print(Pi)
    print(alpha)
    Pi, alpha,A = em.Maximization(p_md,E_log_theta,alpha,z_count,model_type,A,gibbs_itrn,beta_smooth,i)
    p_md,E_log_theta,z_count = em.Expectation(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs,z_init,gibbs_itrn)

	# Write alpha and pi out for likelihood computation
    for m in range(K):
        for topics in range(no_of_topics):	
            f_alpha.write("%f\t"%(alpha[m,topics]))
        f_alpha.write("\n")
        
    # Writting the A matrix compatible with mallet
    for vocab in range(vocabSize):
        for topic in range(no_of_topics):
            f_beta.write("%d:%f\t"%(topic,A[vocab,topic]))
        f_beta.write("\n")

    for p in Pi:
        print p
        f_pi.write("%f\n"%p)

    print(alpha)	

    # For flushing the buffers after every iterations
    f_alpha.close()
    f_pi.close()
    f_beta.close()
    f_alpha = open(out_alpha,'a')
    f_pi = open(out_pi, 'a')
    f_beta = open(out_beta,'a')


# Saving the A file
if(model_type != 0):
    A = array(A)
    np.savetxt(A_file, A)

# Print Top topics corresponding to the learned alphas    
top_index = em.pick_top_index(alpha,5)
for i in range(top_index.shape[0]):
	print("Top Alpha Topics in cluster",i)
	ind_alpha_index = top_index[i]
	for index in ind_alpha_index:
		print(index[1],':',topwords[int(index[0])])

#ht = ch.create_html_report(topwords,top_index,Pi)
#print(ht,file=f)
#print >>f, ht

#f.close()
f_beta.close()
f_alpha.close()
f_pi.close()
