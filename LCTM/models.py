import os
import numpy as np
from collections import OrderedDict
from functools import reduce

import sklearn.metrics as sm
import scipy.ndimage as nd
import matplotlib.pylab as plt

# os.chdir("/Users/colin/CVPR2016/src/")
from LCTM import weights
from LCTM.energies import priors
from LCTM.energies import unary
from LCTM.energies import pairwise as pw
from LCTM.infer import segmental_inference
from LCTM.infer_known_order import infer_known_ordering
from LCTM import learn
from LCTM import ssvm


class Logger:
    def __init__(self):
        self.objectives = {}

class CoreModel:
    name = ""
    debug = False

    # Model
    n_classes = None
    n_features = None    
    inference_type = "framewise"

    # Learning
    fit = learn.subgradient_descent
    loss_fcn = ssvm.hamming_loss
    objective = ssvm.objective_01

    # TODO: Add other regularlizers
    regularizer = "" 

    def __init__(model, name="", debug=False, inference="framewise"):
        model.name = name
        model.debug = debug        
        model.inference_type = inference

        model.potentials = OrderedDict()
        model.ws = weights.Weights()
        model.logger = Logger()        

        assert inference in ["framewise", "segmental"], \
            "'inference' must be framewise or segmental"

    @property
    def is_latent(model):
        if hasattr(model, "n_latent") and model.n_latent > 1:
            return True
        else:
            return False

    @property
    def n_nodes(model):
        if model.is_latent:
            n_nodes = model.n_classes*model.n_latent
        else:
            n_nodes = model.n_classes
        return n_nodes

    def get_weights(model):
        return list(model.ws.values())

    def predict_latent(model, Xi):
        return model.predict(Xi, output_latent=True)

    def predict(model, Xi, Yi=None, is_training=False, output_latent=False, inference=None, known_order=None):
        # This function is applicable to normal and latent models
        if type(Xi) is list:
            out = []
            for i in range(len(Xi)):
                Yi_ = None if Yi is None else Yi[i]
                out += [model.predict(Xi[i], Yi_, is_training, output_latent, inference, known_order)]
            return out

        # Check that Xi is of size FxT
        if Xi.shape[0] > Xi.shape[0]:
            Xi = Xi.T

        if Yi is not None:
            assert Xi.shape[1]==Yi.shape[0], "Error: Xi and Yi are of shapes {} and {}".format(Xi.shape[1],Yi.shape[0])

        _, n_timesteps = Xi.shape
        n_nodes = model.n_nodes

        # Initialize score
        score = np.zeros([n_nodes, n_timesteps], np.float64)

        # loss augmented inference (if training)
        if is_training:
            if model.is_latent:
                score += ssvm.latent_loss_augmented_unaries(score, Yi, model.n_latent)
            else:
                score += ssvm.loss_augmented_unaries(score, Yi)

        # Add potentials to score
        for key in model.potentials:
            score = model.potentials[key].compute(model, Xi, score)

        if model.is_latent and (not is_training and not output_latent):
            score = ssvm.reduce_latent_states(score, model.n_latent, model.n_classes)

        # Get predictions
        inference_type = model.inference_type if inference is None else inference
        if inference_type is "framewise":
            path = score.argmax(0)

        elif inference_type is "filtered":
            assert hasattr(model, "filter_len"), "filter_len must be set"
            path = score.argmax(0)
            path = nd.median_filter(path, model.filter_len)

        elif inference_type is "segmental":

            if known_order is not None:
                path = infer_known_ordering(score.T, known_order)
            else:
                assert hasattr(model, "max_segs"), "max_segs must be set"
                # Check if there is a segmental pw.pairwise term
                seg_term = [p.name for p in model.potentials.values() if type(p) is pw.segmental_pairwise]                
                if len(seg_term) >= 1:
                    path = segmental_inference(score.T, model.max_segs, pw=model.ws[seg_term[0]])
                else:
                    path = segmental_inference(score.T, model.max_segs)

        return path 


class CoreLatentModel(CoreModel):
    n_latent = 1
    def __init__(self, n_latent, **kwargs):
        CoreModel.__init__(self, **kwargs)
        self.n_latent = n_latent

class ChainModel(CoreModel):
    def __init__(self, skip=1, **kwargs):
        CoreModel.__init__(self, name="SC-Model", **kwargs)

        # self.potentials["class_prior"] = class_prior()
        # self.potentials["prior.temporal_prior"] = prior.temporal_prior(length=30)
        self.potentials["unary"] = unary.framewise_unary()
        
        # self.potentials["pw2"] = pw.pairwise("pw2", skip*2)
        if skip: self.potentials["pw"] = pw.pairwise(skip, name="pw")
        # self.potentials["pw1"] = pw.pairwise("pw0", 1)

class ConvModel(CoreModel):
    def __init__(self, skip=1, conv_len=100, **kwargs):
        CoreModel.__init__(self, name="Conv-Model", **kwargs)
        self.debug = debug

        # self.potentials["prior.temporal_prior"] = prior.temporal_prior(length=30)
        self.potentials["conv"] = unary.conv_unary(conv_len=conv_len)
        # self.potentials["class_prior"] = class_prior()
        # self.potentials["pw2"] = pw.pairwise("pw2", skip*2)
        if skip: self.potentials["pw"] = pw.pairwise(skip, name="pw")
        # if skip: self.potentials["pw0"] = pw.pairwise("pw0", 1)

class LatentChainModel(CoreLatentModel):
    def __init__(self, n_latent, skip=1, **kwargs):
        # CoreLatentModel.__init__(self, n_latent, name="Latent Skip Chain Model", **kwargs)
        super(CoreLatentModel,self).__init__(name="Latent Skip Chain Model", **kwargs)

        self.potentials["prior.temporal_prior"] = priors.temporal_prior(length=30)
        self.potentials["unary"] = unary.framewise_unary()
        
        # if skip: self.potentials["pw2"] = pw.pairwise("pw2", skip*2)
        if skip: self.potentials["pw"] = pw.pairwise(skip, name="pw")

class LatentConvModel(CoreModel):
    def __init__(self, n_latent, conv_len=100, skip=1, prior=False, **kwargs):
        CoreLatentModel.__init__(self, n_latent, name="Latent Convolutional Model", **kwargs)

        if prior: self.potentials["temporal_prior"] = priors.temporal_prior(length=30)
        self.potentials["conv"] = unary.conv_unary(conv_len=conv_len)
        if skip: self.potentials["pw"] = pw.pairwise(skip, name="pw")

class SegmentalModel(CoreModel):
    def __init__(self, pretrained=False, **kwargs):
        CoreModel.__init__(self, name="Seg-Model", **kwargs)

        self.potentials["temporal_prior"] = priors.temporal_prior(length=30)
        if pretrained:
            self.potentials["pre"] = unary.pretrained_unary()
        else:
            self.potentials["unary"] = unary.framewise_unary()
        # self.potentials["start"] = start_prior()
        # self.potentials["end"] = end_prior()
        # self.potentials["pw_"] = pw.pairwise("pw", skip)
        self.potentials["pw"] = pw.segmental_pairwise(name="pw")

class PretrainedModel(CoreModel):
    def __init__(self, skip=1, **kwargs):
        CoreModel.__init__(self, name="Pretrained-Model", **kwargs)

        self.potentials["pre"] = unary.pretrained_unary()
        self.potentials["pw"] = pw.pairwise(skip=skip)

if 0:
    Xi = X_test[0]
    Yi = y_test[0]
    X, Y = X_train, y_train
    is_training=False

    # model = ChainModel(skip=100, debug=True)
    skip = 100
    model = LatentChainModel(n_latent=1, skip=skip, debug=True)
    # model = LatentConvModel(n_latent=1, conv_len=50, skip=0, debug=True)
    # model = SegmentalModel(skip=200, debug=True)
    # model = ConvModel(skip=200, conv_len=200, debug=True)
    model.fit(X_train, y_train, n_iter=200, learning_rate=.1, pretrain=True)

    # Framewise inference
    model.inference_type = "framewise"
    P_test = model.predict(X_test)
    acc = utils.accuracy(P_test, y_test)
    print("Frame Inference Acc: {:.3}%".format(acc))

    # Framewise inference
    model.inference_type = "filtered"
    model.filter_len = skip//2
    P_test_filtered = model.predict(X_test)
    acc = utils.accuracy(P_test_filtered, y_test)
    print("Filtered Inference Acc: {:.3}%".format(acc))

    # Segmental inference
    model.inference_type = "segmental"
    model.max_segs = utils.max_seg_count(y_train)
    P_test_seg = model.predict(X_test)
    acc = utils.accuracy(P_test_seg, y_test)
    print("Seg Inference Acc: {:.3}%".format(acc))

    n_test = len(y_test)
    imgs = [np.vstack([y_test[i], P_test[i], P_test_filtered[i], P_test_seg[i]]) for i in range(n_test)]

    for i in range(n_test):
        plt.subplot(n_test,1,i+1)
        utils.imshow_(imgs[i])


    # for i in range(model.n_classes):
    #     subplot(4,3,i+1)
    #     imshow_(np.squeeze(model.ws['conv'][i]))
    #     # axis("off")

    # show()
