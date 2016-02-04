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
class class_prior(CorePotential):
    def __init__(self, name='class_prior'):
        self.name = name

    def init_weights(self, model):
        return np.random.randn(model.n_nodes, 1)

    def cost_fcn(self, model, Xi, Yi):
        cost = np.histogram(Yi, model.n_nodes, [0, model.n_nodes])[0][:,None].astype(np.float64)
        cost /= cost.sum()
        return cost

    def compute(self, model, Xi, score):
        return score + model.ws[self.name]

@jit("float64[:,:](float64[:,:], float64[:], int64)")
def temporal_cost(cost, Yi, length):
    T = Yi.shape[0]
    idxs = np.linspace(0, length-1, T).astype(np.int)
    for t in range(T):
        cost[Yi[t], idxs[t]] += 1
    return cost    

@jit("float64[:,:](float64[:,:], float64[:,:], int64)")
def temporal_compute(score, ws, length):
    T = score.shape[1]
    idxs = np.linspace(0, length-1, T).astype(np.int)
    for t in range(T):
        score[:, t] += ws[:,idxs[t]]
    return score

class temporal_prior(CorePotential):
    def __init__(self, length=30, name='temporal_prior'):
        self.name = name
        self.length = length

    def init_weights(self, model):
        return np.random.randn(model.n_nodes, self.length)

    def cost_fcn(self, model, Xi, Yi):
        T = Xi.shape[1]
        cost = np.zeros([model.n_nodes, self.length], np.float)
        # cost /= cost.sum(1)[:,None]
        
        return temporal_cost(cost, Yi, self.length) / T

    def compute(self, model, Xi, score):
        return temporal_compute(score, model.ws[self.name], self.length)


class start_prior(CorePotential):
    def __init__(self, name='start'):
        self.name = name

    def init_weights(self, model):
        return np.random.randn(model.n_nodes, 1)

    def cost_fcn(self, model, Xi, Yi):
        cost = np.histogram(Yi[:1], model.n_nodes, [0, model.n_nodes])[0][:,None].astype(np.float64)
        cost /= cost.sum()
        return cost

    def compute(self, model, Xi, score):
        return score

class end_prior(CorePotential):
    def __init__(self, name='end'):
        self.name = name

    def init_weights(self, model):
        return np.random.randn(model.n_nodes, 1)

    def cost_fcn(self, model, Xi, Yi):
        cost = np.histogram(Yi[-1], model.n_nodes, [0, model.n_nodes])[0][:,None].astype(np.float64)
        cost /= cost.sum()
        return cost

    def compute(self, model, Xi, score):
        return score        

