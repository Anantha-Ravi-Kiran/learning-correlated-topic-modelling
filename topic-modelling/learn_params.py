from __future__ import division
import IPython
from itertools import groupby
from scipy.sparse import *
from scipy import *
from scipy.special import digamma
from html import HTML
import  multiprocessing as mp
import sys
import scipy.io
import numpy as np
import pdb
import time
import operator
import gibbs_sampling as gs
import inverse_digamma as inv_d
import create_html as ch
import Exp_max as em
import os

# K - No of dirichlet mixtures which is passed as an argument
# A = matrix(zeros((V,K)))
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

if len(sys.argv) < 4:
    print("usage: input_doc A_matrix top_words output_file")
    sys.exit()

input_docs_file=sys.argv[1]
A_file=sys.argv[2]
top_words_in=sys.argv[3]
output_file=sys.argv[4]

dirname,f_temp = os.path.split(sys.argv[2])

# Reading the input files
input_docs = scipy.io.loadmat(input_docs_file)['M']
A = np.loadtxt(A_file) 
topwords_file = open(top_words_in, "r")

# Read Top words from the file
topwords = []
try:
	for line in topwords_file.read().split('\n'):
		topwords.append(line)
finally:
	topwords_file.close()

# Initialization
input_docs = input_docs[:,1:10]
vocabSize = input_docs.shape[0]
numdocs = input_docs.shape[1]
no_of_topics = A.shape[1]

print("No of topics: %d" %(no_of_topics))

# Setting zero values in A to smallest possible float and 
# Renormalization    
A = [[em.remove_zero(x) for x in row] for row in A]    
A = np.array(A)
for i in range(A.shape[1]):
	A[:,i] = A[:,i]/A[:,i].sum()

# Writting the A matrix compatible with mallet
out_beta = "%s/%s%d.txt" %(dirname,"topics-",no_of_topics)
f_beta = open(out_beta,'w')
for vocab in range(vocabSize):
	for topic in range(no_of_topics):
		f_beta.write("%d:%f\t"%(topic,A[vocab,topic]))
	f_beta.write("\n")
f_beta.close()

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

# Collecting words in each document for multiprocessing efficiently 
word_list = []
for docs in doc_list:
	word_doc = []
	for doc_index in docs:
		start = input_docs.indptr[doc_index]
		end = input_docs.indptr[doc_index + 1]
		word_indices = input_docs.indices[start:end]
		word_doc.append(word_indices)    
	word_list.append(word_doc)
	#word_list.append(np.array(word_doc))
#word_list = np.array(word_list) #YONI COMMENTED OUT HERE

# Writting the report to the output - bufsize = 0 for unbuffering.
out_file_name = "%s_%d.html" %(output_file,no_of_topics)
out_alpha = "%s/%s_%d.txt" %(dirname,"alpha",no_of_topics)
out_pi = "%s/%s_%d.txt" %(dirname,"pi",no_of_topics)

f = open(out_file_name,'w')
f_alpha = open(out_alpha,'w')
f_pi = open(out_pi, 'w')

#for K in range(1,int(no_of_topics/5) + 1):
for K in range(5,6):    
	# Initializing the dirichlet mixtures with equal probability

	Pi = np.ones(K)*(1/K)     
	alpha = np.ones([K,no_of_topics])*(1/no_of_topics)

	# Pending - Initialize alpha using the moments from the data instead.
	p_md,E_log_theta = em.Expectation(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs)
	#while True:
	for i in range(1,3):
		Pi, alpha = em.Maximization(p_md,E_log_theta,alpha)
		p_md,E_log_theta = em.Expectation(alpha, Pi, A, word_list, doc_list,vocabSize,numdocs)
		
		# Write alpha and pi out for likelihood computation
		for m in range(K):
			for topics in range(no_of_topics):	
				f_alpha.write("%f\t"%(alpha[m,topics]))
			f_alpha.write("\n")

		for p in Pi:
			print p
			f_pi.write("%f\n"%p)

	# Print Top topics corresponding to the learned alphas    
	top_index = em.pick_top_index(alpha,5)
	for i in range(top_index.shape[0]):
		print("Top Alpha Topics in cluster",i)
		ind_alpha_index = top_index[i]
		for index in ind_alpha_index:
			print(index[1],':',topwords[int(index[0])])
   
	print(alpha)
	print(Pi)

	ht = ch.create_html_report(topwords,top_index,Pi)
	#print(ht,file=f)
	print >>f, ht

	# For flushing the buffers after every iterations
	f.close()
	f_alpha.close()
	f = open(out_file_name,'a')
	f_alpha = open(out_alpha,'a')
	f_pi = open(out_pi, 'a')


f.close()
f_alpha.close()
f_pi.close()
