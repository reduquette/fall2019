#Instead of trying to do calculations for a particular input / output, here we're just going to operate on a matrix
#saves redoing calculations -- and then we have an answer for every single possible referent

#m messages (rows) x n objects (columns)

import numpy as np
from sklearn.preprocessing import normalize


def rsa_test(vocab):
    l0 = normalize(vocab, axis=1, norm='l1')
    print(l0)
    s1 = normalize(l0, axis=0, norm='l1')
    print(s1)
    s1 = normalize(s1, axis=1, norm='l1')
    print(s1)


def rsa(vocab, num_recurse):
    # num_recurse is the order of the speaker for which the matrix is returned
    prob_dist = vocab.copy()
    for i in range(num_recurse):
        #axis = 1: by row: by message
        prob_dist = normalize(prob_dist, axis=1, norm='l1', copy=False)
        print("Listener", str(i), "\n", prob_dist)
        #axis = 0: by column: by object
        prob_dist = normalize(prob_dist, axis=0, norm='l1', copy=False)
        print("Speaker", str(i + 1), "\n", prob_dist)
    return prob_dist


def ref_game():
    m = 3  #|V|
    n = 3  #|C|
    # x  = np.empty(shape(m, n))
    print("Original vocabulary: literal meanings")
    vocab = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])

    model = rsa(vocab, 2)


ref_game()
