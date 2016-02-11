import numpy as np
from numba import float64, jit, int16, int32, int64


class CorePotential:
    def __init__(self, name=""):
        self.name = name
    def init_weights(self, model):
        pass
    def cost_fcn(self, model, X, Y):
        pass
    def compute(self, model, X, score):
        pass                

# -----------------------------------------
class pretrained_unary(CorePotential):
    def __init__(self, name='pre'):
        self.name = name

    def init_weights(self, model):
        return np.ones(1)

    def cost_fcn(self, model, Xi, Yi):
        return np.ones(1)

    def compute(self, model, Xi, score):
        return score + Xi


# -----------------------------------------
@jit("float64[:,:](float64[:,:], float64[:,:], int64[:])")
def framewise_cost(cost, Xi, Yi):
    T = Yi.shape[0]
    for t in range(T):
        cost[Yi[t],:] += Xi[:, t]

    return cost

class framewise_unary(CorePotential):
    def __init__(self, name='unary'):
        self.name = name

    def init_weights(self, model):
        return np.random.randn(model.n_nodes, model.n_features)

    def cost_fcn(self, model, Xi, Yi):
        _, T = Xi.shape
        cost = np.zeros([model.n_nodes, model.n_features], np.float64)
        cost = framewise_cost(cost, Xi, Yi)

        return cost / T

    def compute(self, model, Xi, score):
        return score + (model.ws[self.name] @ Xi)


# -----------------------------------------
@jit("float64[:,:](float64[:,:], int32)")
def buffer_data(Xi, new_len):
    # Add buffer to end of data
    n_features, T = Xi.shape
    Xi_tmp = np.zeros([n_features, new_len], np.float64)
    Xi_tmp[:,:T] = Xi
    Xi_tmp[:,T:] += Xi[:,-1][:,None]

    return Xi_tmp

@jit("float64[:,:,:](float64[:,:,:], float64[:,:], int32[:])")
def conv_cost(cost, Xi, Yi):
    T = Xi.shape[1]
    n_classes, n_features, conv_len = cost.shape

    Xi_tmp = buffer_data(Xi, T+conv_len)

    for t in range(T):
        cost[Yi[t], :] += Xi_tmp[:, t:t+conv_len]

    cost /= conv_len
    return cost

# %timeit conv_cost(cost, Xi, Yi)

@jit("float64[:,:](float64[:,:], float64[:,:,:])")
def convolve1d(Xi, ws):
    # ws : CxFxT
    n_features, T = Xi.shape
    n_classes, _, conv_len = ws.shape

    # Add buffer to end of data for convolving
    Xi_tmp = buffer_data(Xi, T+conv_len)
    
    # Convolve using dot products
    score = np.zeros([n_classes, T], np.float64)
    for t in range(conv_len):
        score += np.dot(ws[:,:,t], Xi_tmp[:,t:-conv_len+t])
        
    return score

# %timeit convolve1d(Xi, model.ws['conv'])


# import tensorflow as tf
# @jit("float64[:,:](float64[:,:], float64[:,:,:])")
# def tf_convolve1d(Xi, ws):
#     # ws : CxFxT
#     n_features, T = Xi.shape
#     n_classes, _, conv_len = ws.shape

#     # Add buffer to end of data for convolving
#     Xi_tmp = buffer_data(Xi, T+conv_len)
    
#     # Convolve using dot products
#     score = np.zeros([n_classes, T], np.float64)
#     for t in range(conv_len):
#         score += np.dot(ws[:,:,t], Xi_tmp[:,t:-conv_len+t])
        
#     return score


class conv_unary(CorePotential):
    def __init__(self, conv_len=1, name="conv"):
        self.name = name
        self.conv_len = conv_len

    def init_weights(self, model):
        return np.zeros([model.n_nodes, model.n_features, self.conv_len], dtype=np.float64)
    
    def cost_fcn(self, model, Xi, Yi):
        _, T = Xi.shape
        conv_len = self.conv_len

        cost = np.zeros([model.n_nodes, model.n_features, self.conv_len], np.float64)
        cost = conv_cost(cost, Xi, Yi)
        cost /= conv_len

        return cost

    def compute(self, model, Xi, score):
        T = Xi.shape[1]
        return score + convolve1d(Xi, model.ws[self.name]) / self.conv_len #/ T


# -----------------------------------------

# class time_invariant_unary(CorePotential):
#     def __init__(self, name='pw', conv_len=1):
#         self.conv_len = conv_len
#         self.name = name

#     def init_weights(self, model):
#         return np.zeros([model.n_classes, model.n_features, self.conv_len], dtype=np.float64)
#         # return np.random.randn(model.n_classes, model.n_classes)
    
#     def cost_fcn(self, model, Xi, Yi):
#         return pw_cost(Yi, model.n_classes, self.skip)

#     def compute(self, model, Xi, score):
#         return compute_pw(score, model.ws[self.name], self.skip)


#     T = Xi.shape[2]
#     filter_lengths = model.conv_length

#     # Split labels into segments
#     segmented = model.segment_model.transform(model.segment_model, x)
#     segs = split_by_labels(segmented, x)
#     segs_labels = split_by_labels(segmented, y)
#     segs_labels = [int(median(s)) for s in segs_labels]
#     n_segs = size(segs, 1)

#     # Pre-allocate matrices
#     cost = zeros(Float64, (model.n_features, filter_lengths, model.n_classes))
#     for i in 1:n_segs
#         # Resize the segment so it's the same length as the weights
#         s = resize_segment(segs[i], filter_lengths)
#         # Add to cost
#         cost[:, :, segs_labels[i]] += s
#     end
#     cost /= filter_lengths


# function compute_segmental_unary!(model, score::Array{Float64,2}, segs)
#     """
#     This looks at every N frames and adds to cost.
#     Oversmooths data
#     """

#     filter_lengths = model.filter_lengths
#     n_segs = size(segs, 1)

#     score_new = zeros(score)
#     for i in 1:n_segs
#         # Resize the segment so it's the same length as the weights
#         s = resize_segment(segs[i], filter_lengths)
#         # Apply action filter to data
#         score_new[:,i] = sum(sum(model.ws.conv .* s, 1), 2)[:]
#     end
#     score_new /= filter_lengths
#     score[:] += score_new[:]

# end
