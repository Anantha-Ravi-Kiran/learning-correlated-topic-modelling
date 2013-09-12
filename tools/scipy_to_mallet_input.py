#takes in matrix in UCI repository format and outputs a scipy sparse matrix file

import sys
import scipy.io
import scipy
import numpy as np

if len(sys.argv) < 3:
    print("usage: input_matrix(scipy format) vocab_file output_file(txt)")
    sys.exit()
    
input_matrix = sys.argv[1]
vocab_file = sys.argv[2]
output_file = sys.argv[3]
    
input_docs = scipy.io.loadmat(input_matrix)['M']
vocab = file(vocab_file).read().strip().split()
outfile = open(output_file,"w")    

vocabSize = input_docs.shape[0]
numdocs = input_docs.shape[1]

for doc_index in xrange(numdocs):
    start = input_docs.indptr[doc_index]
    end = input_docs.indptr[doc_index + 1]
    word_indices = input_docs.indices[start:end]
    data = input_docs.data[start:end]
    zipped=zip(word_indices,data)
    outfile.write('%d\tX\t'%doc_index)
    for word_index,d in zipped:
        for i in xrange(int(d)):
            outfile.write(vocab[word_index])
            outfile.write(' ')
    outfile.write('\n')    
