import numpy as np
from numba import float64, jit, int16, int32, int64

from LCTM import utils

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
@jit("float64[:,:](int32[:], int32, int32)")
def pw_cost(Yi, n_classes, skip=1):
    T = Yi.shape[0]
    cost = np.zeros([n_classes, n_classes], np.float64)
    for t in range(skip, T):
        cost[Yi[t-skip], Yi[t]] += 1
    cost /= T-skip
    return cost

def segmental_pw_cost(Yi, n_classes, skip=1):
    Yi_ = utils.segment_labels(Yi)
    return pw_cost(Yi_, n_classes, skip)

@jit("float64[:,:](float64[:,:], float64[:,:], int32)")
def compute_pw(scores, ws, skip=1): 
    T = scores.shape[1]
    # Forward step
    for t in range(skip, T):
        prev_class = scores[:,t-skip].argmax()
        scores[:,t] += ws[prev_class]
    
    # Backward step
    for t in range(T-skip, -1, -1):
        prev_class = scores[:,t+skip].argmax()
        scores[:,t] += ws[:,prev_class]

    return scores


class pairwise(CorePotential):
    def __init__(self, skip=1, name='pw'):
        self.skip = skip
        self.name = name


    def init_weights(self, model):
        return np.random.randn(model.n_nodes, model.n_nodes)
        # return np.eye(n_nodes, dtype=np.float64)
    
    def cost_fcn(self, model, Xi, Yi):
        return pw_cost(Yi, model.n_nodes, self.skip)

    def compute(self, model, Xi, score):
        return compute_pw(score, model.ws[self.name], self.skip)


class segmental_pairwise(CorePotential):
    def __init__(self, name='pw'):
        self.name = name

    def init_weights(self, model):
        # return np.zeros([model.n_classes, model.n_classes], dtype=np.float64)
        return np.random.randn(model.n_classes, model.n_classes)
        # return np.eye(model.n_classes, dtype=np.float64)
    
    def cost_fcn(self, model, Xi, Yi):
        return segmental_pw_cost(Yi, model.n_classes)

    def compute(self, model, Xi, score):
        # We're going to add this into segmental inference... so keep cost the same
        return score
