

import numpy as np
from numba import float64, jit, int16, boolean, autojit
import scipy
import sklearn.neighbors as sn
from LCTM import utils

# LB_Keogh = sqrt(sum([[Q > U].* [Q-U]; [Q < L].* [L-Q]].^2))
# It looks like this is intended for 1D data
# def LB_Keogh(x, y, pct=0.3):
#     Tx = x.shape[1]
#     Ty = y.shape[1]
#     min_len = min(Tx, Ty)

#     pct_offset = int(pct*Tx)
#     U = nd.maximum_filter(x, [1, pct_offset])
#     L = nd.minimum_filter(x, [1, pct_offset])
    

#     cost_U = (y[:,min_len]-U[:,min_len])**2 * (y[:,min_len]>U[:,min_len])
#     cost_L = (y[:,min_len]-L[:,min_len])**2 * (y[:,min_len]<L[:,min_len])

#     cost = cost_U + cost_L

@jit("int32(float32[:,:])")
def _traceback(D):
    n_ref, n_in = D.shape
    correspondences = np.zeros(n_in, np.int)
    correspondences[-1] = n_ref-1

    c = n_ref-1
    i = n_in-1
    while i > 0 and c > 0:
        a = np.argmin([
                       D[c-1, i-1],
                       D[c-1, i],
                       D[c, i-1]
                       ])
        if a==0 or a==1:
            c -= 1
        if a==0 or a==2:
            i -= 1
        correspondences[i] = c

    return correspondences

# @jit("(float32, int32[:])(float32[:,:], float32[:,:], float32)")
@autojit
def DTW(x, y, max_value=np.inf, output_correspondences=False, output_costs=False):
    # Should be of shape FxT
    Tx = x.shape[1]
    Ty = y.shape[1]

    cost = np.zeros([Tx, Ty], np.float32) + np.inf
    # Compute first row
    # cost[0,:] = ((x[:,0][:,None]-y)**2).sum(0).cumsum(0)
    cost[0,:] = (np.abs(x[:,0][:,None]-y)).sum(0).cumsum(0)
    # cost[0,:] = ((x[:,0][:,None]!=y)).sum(0).cumsum(0)
    
    # Compute rest of the rows
    tx = 1
    while tx < Tx:
        ty = 1
        while ty < Ty:
            topleft = cost[tx-1, ty-1]
            top = cost[tx-1, ty]
            left = cost[tx, ty-1]

            # current = np.sum((x[:,tx]-y[:,ty])**2)
            current = np.sum(np.abs(x[:,tx]-y[:,ty]))
            # current = np.sum((x[:,tx]!=y[:,ty]))
            cost[tx, ty] = min(topleft, left, top) + current

            # if cost[tx, ty] > max_value:
            #     cost[-1,-1] = np.inf 
            #     tx = Tx
            #     ty = Ty
            ty += 1
        tx += 1

    out = [cost[-1,-1]]
    if output_correspondences:
        c = _traceback(cost)
        out += [c]
    if output_costs:
        cost[np.isinf(cost)] = 0
        out += [cost]

    return out

# path = coors[0]
# table = tables[0]

@autojit
def draw_path(table, path):
    table_new = table.copy()
    max_val = table.max()
    for i in range(len(path)):
        table_new[path[i], i] = max_val
        if path[i]>0: table_new[path[i]-1, i] = max_val
        # if path[i]<table.shape[1]-1: table_new[path[i]+1, i] = max_val
    return table_new


# def DTW(x, y, max_value=np.inf):
#     Tx = x.shape[1]
#     Ty = y.shape[1]    
#     idx_x = np.linspace(0, Tx-1, Tx).astype(np.int)
#     idx_y = np.linspace(0, Ty-1, Tx).astype(np.int)
#     return ((x[:,idx_x]-y[:,idx_y])**2).sum()

# x = np.random.random([10,1000])
# y = np.random.random([10,1000])

# %timeit DTW(x,y)
# 146 ms / loop




def normalize(x):
    x -= x.mean(1)[:,None]
    std = x.std(1)[:,None]
    std[std==0] = 1
    x /= std
    return x


class DTWClassifier:

    def __init__(self, sample_rate=1, n_neighbors=5, normalize=False):
        self.sample_rate = sample_rate
        self.n_neighbors = n_neighbors
        self.normalize = normalize

    def fit(self, X, Y):
        Xs_train = []
        Ys_train = []
        for i in range(len(X)):
            Xs, Ys = utils.segment_data(X[i], Y[i])
            if self.normalize:
                Xs = [normalize(x) for x in Xs]
            Xs_train += Xs
            Ys_train += Ys

        self.Xs_train = Xs_train
        self.Ys_train = Ys_train
        self.n_train = len(Xs_train)
        self.n_classes = len(np.unique(Ys_train))

    def predict(self, X, Y, return_prob=False):
        if type(X) is list:
            out = []
            for i in range(len(Xi)):
                out += [self.predict(model, X[i], Y[i], return_prob)]
            return out

        Xs, Ys = utils.segment_data(X, Y)
        n_segs = len(Xs)
        if self.normalize:
            Xs = [normalize(x) for x in Xs]

        Ps = np.zeros(n_segs, np.int)
        Ps_prob = np.zeros([self.n_classes, n_segs], np.float)
        top_scores = [np.inf]*self.n_neighbors
        for k in range(n_segs):
            scores = np.zeros(self.n_train, np.float)
            worst_best_score = np.inf
            for j in range(self.n_train):
                scores[j] = DTW(Xs[k][:,::self.sample_rate], self.Xs_train[j][:,::self.sample_rate], worst_best_score)
                
                if scores[j] < np.min(top_scores):
                    idx = np.argmin(top_scores)
                    top_scores[idx] = scores[j]
                    worst_best_score = np.max(top_scores)
            
            argsorted = scores.argsort()
            neighbors = [self.Ys_train[a] for a in argsorted[:self.n_neighbors]]
            neighbors_scores = np.array([scores[a] for a in argsorted[:self.n_neighbors]])
            Ps_prob[:,k] = np.histogram(neighbors, self.n_classes, [0, self.n_classes])[0]*1.
            Ps_prob[:,k] /= Ps_prob[:,k].sum()

            Ps[k] = scipy.stats.mode(neighbors)[0][0]
        
        if return_prob:
            return Ps_prob
        else:
            return Ps

    def predict_proba(self, X, Y):
        return self.predict(X, Y, return_prob=True)



# Xs_test = []
# Ys_test = []
# Ps_test = []
# for i in range(len(X_test)):
#     Xs, Ys = segment_data(X_test[i], y_test[i])
#     # Xs = [normalize(x) for x in Xs]

#     Ps = np.zeros(len(Xs), np.int)
#     for k in range(len(Xs)):
#         scores = np.zeros(n_train, np.float)
#         best_score = np.inf
#         for j in range(n_train):
#             # scores[j] = DTW(Xs[k], Xs_train[j])
#             scores[j] = DTW(Xs[k][:,::5], Xs_train[j][:,::5])
#             # if scores[j] < best_score:
#                 # best_score = scores[j]
#         # scores = np.array([DTW(Xs[k][:,::2], Xs_train[j][:,::2]) for j in range(n_train)])
        
#         # clf = sn.KNeighborsClassifier(5, )
#         argsorted = scores.argsort()
#         neighbors = [Ys_train[a] for a in argsorted[:5]]
#         Ps[k] = scipy.stats.mode(neighbors)[0][0]
#         print("Truth={}, Pred={}".format(Ys[k], Ps[k]))

#     Xs_test += Xs
#     Ys_test += Ys
#     Ps_test += Ps.tolist()
#     print(np.mean(Ys==Ps)*100)

# acc = np.mean([np.mean(Ys_test[i]==Ps_test[i]) for i in range(n_test)])*100
# print("Avg={:.4}%".format(acc))



