import numpy as np
import scipy
from numba import jit, int64

from LCTM import utils
from LCTM.dtw import DTW

def accuracy(P, Y):
    def acc_(p,y):
        return np.mean(p==y)*100
    if type(P) == list:
        return np.mean([np.mean(P[i]==Y[i]) for i in range(len(P))])*100
    else:
        return acc_(P,Y)

def classification_accuracy(P, Y):
    def clf_(p, y):
        segs_p, segs_y = utils.segment_data(p[:,None].T, y)
        segs = np.array([scipy.stats.mode(s)[0][0] for s in segs_p])
        return np.mean(segs == segs_y)*100

    if type(P) == list:
        return np.mean([clf_(P[i], Y[i]) for i in range(len(P))])
    else:
        return clf_(P, Y)


@jit("float64(int64[:], int64[:])")
def edit_(p,y):
    n_col = len(p)
    n_row = len(y)
    D = np.zeros([n_row+1, n_col+1], np.float)
    for i in range(n_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i    

    for j in range(1, n_col+1):
        for i in range(1, n_row+1):
            if p[i-1] == y[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j], D[i,j-1], D[i-1,j-1]) + 1

    return (1 - D[-1,-1]/max(len(p), len(y))) * 100


def edit_score(P, Y):
    # def edit_(p,y):
    #     score = DTW(p[:,None], y[:,None])
    #     score = 1. - score/max(len(p), len(y)) 
    #     return score*100

    if type(P) == list:
        tmp = []
        for i in range(len(P)):
            P_ = utils.segment_labels(P[i])
            Y_ = utils.segment_labels(Y[i])            
            tmp += [edit_(P_,Y_)]
        return np.mean(tmp)
    else:
        P_ = utils.segment_labels(P)
        Y_ = utils.segment_labels(Y)
        return edit_(P_, Y_)

# def overlap_score(P, Y):
#   def overlap_(p,y):
#       segs_p, segs_y = segment_data(p[:,None].T, y)
#       score = DTW(p[:,None], y[:,None])
#       score = 1. - score/max(len(p), len(y)) 
#       return score*100

#   if type(P) == list:
#       return np.mean([edit_(P[i],Y[i]) for i in range(len(P))])*
#   else:
#       return edit_(P, Y)


def midpoint_(p,y):
    """
    As suggested in Rohrbach et al (IJCV15) for action detection
    """
    segs_p = utils.segment_intervals(p)
    segs_y = utils.segment_intervals(y)
    class_p = utils.segment_labels(p).tolist()
    class_y = utils.segment_labels(y).tolist()
    unused = np.ones(len(class_y), np.bool)
    n_true = len(segs_y)

    TP, FP, FN = 0, 0, 0
    # Go through each segment and check if it's correct.
    for i in range(len(segs_p)):
        midpoint = np.mean(segs_p[i])
        # Check each corresponding true segment
        for j in range(n_true):
            # If the midpoint is in this true segment
            if segs_y[j][0] <= midpoint <= segs_y[j][1]:
                # If yes and it's correct
                if (class_p[i] == class_y[j]):
                    # Only a TP if it's the first occurance. Otherwise FP
                    if unused[j]:
                        TP += 1
                        unused[j] = 0
                    else:
                        FP += 1
                # FN if it's wrong class
                else:
                    FN += 1
            elif midpoint < segs_y[j][0]:
                break


    prec = float(TP) / (TP+FP) * 100
    recall = float(TP) / (TP+FN) * 100

    return prec, recall

def midpoint_precision(P, Y):
    if type(P) == list:
        return np.mean([midpoint_(P[i],Y[i])[0] for i in range(len(P))])
    else:
        return midpoint_(P, Y)[0]


def midpoint_recall(P, Y):
    if type(P) == list:
        return np.mean([midpoint_(P[i],Y[i])[1] for i in range(len(P))])
    else:
        return midpoint_(P, Y)[1]












