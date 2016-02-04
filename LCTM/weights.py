import numpy as np
from collections import OrderedDict
from copy import deepcopy

class Weights(OrderedDict):

    def op(self, operation, ws=None):
        ws_new = deepcopy(self)
        for key in self.keys():
            if ws is None:
                ws_new[key] = operation(self[key])
            else:
                if type(ws) is Weights:
                    ws_new[key] = operation(self[key], ws[key])
                else:
                    ws_new[key] = operation(self[key], ws)
    
        return ws_new

    def __add__(self, ws):
        return self.op(np.add, ws)
    def __sub__(self, ws):
        return self.op(np.subtract, ws)
    def __mul__(self, ws):
        return self.op(np.multiply, ws)
    def __truediv__(self, ws):
        return self.op(np.divide, ws)
    
    def sqrt(self):
        return self.op(np.sqrt)

    def init_weights(self, model):
        for key in model.potentials:
            init = model.potentials[key].init_weights(model)
            if init.ndim==2:
                U, S, Vt = np.linalg.svd(init, False)
                init = U@Vt
            self[key] = init
