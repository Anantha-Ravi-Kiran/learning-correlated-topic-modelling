#takes in matrix in UCI repository format and outputs a scipy sparse matrix file

import sys
import scipy.io
import scipy
import numpy as np
import random
import math as math
import os

if len(sys.argv) < 3:
    print("usage: input_matrix(scipy format) vocab_file corpus")
    sys.exit()
    
input_matrix = sys.argv[1]
vocab_file = sys.argv[2]
output_file_name = sys.argv[3]

dirname,f = os.path.split(input_matrix)
input_docs = scipy.io.loadmat(input_matrix)['M']
vocab = file(vocab_file).read().strip().split()

vocabSize = input_docs.shape[0]
numdocs = input_docs.shape[1]

train_scipy = dirname + '/' + output_file_name + '_train.scipy'
test_scipy = dirname + '/' + output_file_name + '_test.scipy'
train_mallet = open(dirname + '/' + output_file_name + '_train.mallet',"w")
test_mallet = open(dirname + '/' + output_file_name + '_test.mallet',"w")
train_ctm = open(dirname + '/' + output_file_name + '_train.ctm',"w")
test_ctm = open(dirname + '/' + output_file_name + '_test.ctm',"w")

doc_items = np.array(xrange(numdocs))
#random.shuffle(doc_items)
train_docs = doc_items[:math.ceil(numdocs*0.75)]
test_docs = doc_items[math.ceil(numdocs*0.75):]
train_matrix = scipy.sparse.lil_matrix((vocabSize, train_docs.shape[0]))
test_matrix = scipy.sparse.lil_matrix((vocabSize, test_docs.shape[0]))

temp_index = 0
for doc_index in train_docs:
	# Training data - mallet format and scipy format
    start = input_docs.indptr[doc_index]
    end = input_docs.indptr[doc_index + 1]
    word_indices = input_docs.indices[start:end]
    data = input_docs.data[start:end]
    zipped=zip(word_indices,data)
    train_mallet.write('%d\tX\t'%doc_index)
    train_ctm.write('%d\t'%word_indices.shape[0])
    for word_index,d in zipped:
        for i in xrange(int(d)):
            train_mallet.write(vocab[word_index])
            train_mallet.write(' ')
        train_matrix[word_index, temp_index] = d
        train_ctm.write('%d:%d '%(word_index,d))
    temp_index += 1
    train_mallet.write('\n')    
    train_ctm.write('\n')    

temp_index = 0
for doc_index in test_docs:
	# Training data - mallet format
    start = input_docs.indptr[doc_index]
    end = input_docs.indptr[doc_index + 1]
    word_indices = input_docs.indices[start:end]
    data = input_docs.data[start:end]
    zipped=zip(word_indices,data)
    test_mallet.write('%d\tX\t'%doc_index)
    test_ctm.write('%d\t'%word_indices.shape[0])
    for word_index,d in zipped:
        for i in xrange(int(d)):
            test_mallet.write(vocab[word_index])
            test_mallet.write(' ')
        test_matrix[word_index, temp_index] = d
        test_ctm.write('%d:%d '%(word_index,d))
    temp_index += 1    
    test_mallet.write('\n')
    test_ctm.write('\n')    
    
scipy.io.savemat(train_scipy, {'M' : train_matrix}, oned_as='column')
scipy.io.savemat(test_scipy, {'M' : test_matrix}, oned_as='column')
