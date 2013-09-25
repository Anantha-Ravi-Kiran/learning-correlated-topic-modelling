from __future__ import division
import numpy as np


def Projection_Find(M_orig, r, candidates, logfile):

    if logfile:
        print >>logfile, "beginning projection find"

    n = M_orig[:, 0].size
    dim = M_orig[0, :].size

    M = M_orig.copy()
    
    # stored recovered anchor words
    anchor_words = np.zeros((r, dim))
    anchor_indices = np.zeros(r, dtype=np.int)

    # store the basis vectors of the subspace spanned by the anchor word vectors
    basis = np.zeros((r-1, dim))


    # find the farthest point p1 from the origin
    max_dist = 0
    #for i in range(0, n):
    for i in candidates:
        dist = np.dot(M[i], M[i])
        if dist > max_dist:
            max_dist = dist
            anchor_words[0] = M_orig[i]
            anchor_indices[0] = i

    if logfile:
        print >>logfile, "max_dist", max_dist
    # let p1 be the origin of our coordinate system
    #for i in range(0, n):
    for i in candidates:
        M[i] = M[i] - anchor_words[0]


    # find the farthest point from p1
    max_dist = 0
    #for i in range(0, n):
    for i in candidates:
        dist = np.dot(M[i], M[i])
        if dist > max_dist:
            max_dist = dist
            anchor_words[1] = M_orig[i]
            anchor_indices[1] = i
            basis[0] = M[i]/np.sqrt(np.dot(M[i], M[i]))
    
    if logfile:
        print >>logfile, "max_dist", max_dist

    # stabilized gram-schmidt which finds new anchor words to expand our subspace
    for j in range(1, r - 1):

        # project all the points onto our basis and find the farthest point
        max_dist = 0
        #for i in range(0, n):
        for i in candidates:
            M[i] = M[i] - np.dot(M[i], basis[j-1])*basis[j-1]
            dist = np.dot(M[i], M[i])
            if dist > max_dist:
                max_dist = dist
                anchor_words[j + 1] = M_orig[i]
                anchor_indices[j + 1] = i
                basis[j] = M[i]/np.sqrt(np.dot(M[i], M[i]))

        if logfile:
            print >>logfile, "max_dist", max_dist

    # convert numpy array to python list
    anchor_indices_list = []
    for i in range(r):
        anchor_indices_list.append(anchor_indices[i])
    
    print "done finding anchors"
    return (anchor_words, anchor_indices_list)



