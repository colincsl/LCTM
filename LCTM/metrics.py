import numpy as np
import scipy
from numba import jit, int64, boolean

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


# @jit("float64(int64[:], int64[:])")
# def edit_(p,y):
#     m_row = len(p)    
#     n_col = len(y)
#     D = np.zeros([m_row+1, n_col+1], np.float)
#     for i in range(m_row+1):
#         D[i,0] = i
#     for i in range(n_col+1):
#         D[0,i] = i    

#     for j in range(1, n_col+1):
#         for i in range(1, m_row+1):
#             if  y[j-1]==p[i-1]:
#                 D[i,j] = D[i-1,j-1]
#             else:
#                 D[i,j] = min(D[i-1,j], D[i,j-1], D[i-1,j-1]) + 1
#     score = (1 - D[-1,-1]/max(len(p), len(y))) * 100
    
#     return score


def border_(p, y, intervals=30, max_dur=None):
    # True borders = 
    p_labels = utils.segment_labels(p)
    y_labels = utils.segment_labels(y)
    
    p_starts = np.array([s[0] for s in utils.segment_intervals(p)])
    y_starts = np.array([s[0] for s in utils.segment_intervals(y)])

    dists = []
    n_true = len(y_labels)
    for i in range(n_true):
        start = y_starts[i]
        label = y_labels[i]
        idxs_same_labels = np.nonzero(p_labels==label)[0]
        if len(idxs_same_labels)>0:
            closest_correst = np.abs(p_starts[idxs_same_labels] - start).min()
        else:
            closest_correst = np.inf
        dists += [closest_correst]
    dists = np.array(dists)

    if max_dur is None:
        max_dur = np.max(dists)

    n_bins = int(np.ceil(max_dur / float(intervals)))
    scores = np.array([np.mean(dists<=i*intervals) for i in range(1, n_bins+1)])*100

    return scores

def border_distance(P, Y, intervals=30, max_dur=None):
    if type(P) == list:
        tmp = []
        for i in range(len(P)):
            tmp += [border_(P[i],Y[i], intervals, max_dur)]
        return np.mean(tmp, 0)
    else:
        return border_(P, Y, intervals, max_dur)

@jit("float64(int64[:], int64[:], boolean)")
def levenstein_(p,y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1] 
            else:
                D[i,j] = min(D[i-1,j]+1,
                             D[i,j-1]+1,
                             D[i-1,j-1]+1)
    
    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col) ) * 100
    else:
        score = D[-1,-1]

    return score

@jit("float64(int64[:], int64[:], boolean)")
def lcs_(p,y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for i in range(1, m_row+1):
        for j in range(1, n_col+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j], D[i,j-1])+1

    if norm:
        score = (1 - D[-1,-1]/(m_row+n_col) ) * 100
    else:
        score = D[-1,-1]

    return score


def edit_score(P, Y, norm=True):
    if type(P) == list:
        tmp = []
        for i in range(len(P)):
            P_ = utils.segment_labels(P[i])
            Y_ = utils.segment_labels(Y[i])            
            tmp += [levenstein_(P_,Y_,norm)]
        return np.mean(tmp)
    else:
        P_ = utils.segment_labels(P)
        Y_ = utils.segment_labels(Y)
        return levenstein_(P_, Y_, norm)

def lcs_score(P, Y):
    if type(P) == list:
        tmp = []
        for i in range(len(P)):
            P_ = utils.segment_labels(P[i])
            Y_ = utils.segment_labels(Y[i])            
            tmp += [lcs_(P_,Y_)]
        return np.mean(tmp)
    else:
        P_ = utils.segment_labels(P)
        Y_ = utils.segment_labels(Y)
        return lcs_(P_, Y_)




def overlap_score(P, Y):
    # @jit("float64(int64[:], int64[:])")
    def overlap_(p,y):
        true_intervals = np.array(utils.segment_intervals(y))
        true_labels = utils.segment_labels(y)
        pred_intervals = np.array(utils.segment_intervals(p))
        pred_labels = utils.segment_labels(p)

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, np.float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i]==pred_labels[j]:
                    intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0], true_intervals[i][0])
                    union        = max(pred_intervals[j][1], true_intervals[i][1]) - min(pred_intervals[j][0], true_intervals[i][0])
                    score_ = float(intersection)/union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean()*100

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i]) for i in range(len(P))])
    else:
        return overlap_(P, Y)


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












