# Latent Convolutional Time-series Models

### Overview

LCTM was developed for learning structured prediction models including Conditional Random Fields, Structured SVMs, and Segmental models. We apply this to time-series models, however, it can be easily extended to other data like images. This package was used for the following papers:

[Learning Convolutional Action Primitives for Fine-grained Action Recognition](http://colinlea.com/docs/pdf/2016_ICRA_CLea.pdf). Colin Lea, Rene Vidal, Greg Hager. ICRA 2016.

[Segmental Spatio-Temporal CNNs for Fine-grained Action Segmentation and Classification](http://arxiv.org/abs/1602.02995). Colin Lea, Austin Reiter, Rene Vidal, Greg Hager. arXiv 2016.

LCTM is a **potential**-focused framework. Energy potentials (e.g. unary, pairwise, or priors) are the heart of any given model and vary substantially between domains. Our library makes it easy to develop new potentials. Each potential only requires two functions: a score (given raw data and labels) and inference (given the data and weights). In addition, we have a special data structure for all weights that works seemlessly with learning regardless of if they are vectors, matrices, or tensors.

We speed up LCTM using [Numba](http://numba.pydata.org/), a recent library for compiling Python code with LLVM, which is crucial for some of our inference algorithms. Numba makes it trivial to achieve C++ like performance from Python code.

This library is most directly comparable to [pyStruct](https://pystruct.github.io/). While pyStruct is great in certain ways, we found limitations from their coupling between learning and inference and due to their lack of recent learning algorithms. We find our potential-focused design makes it a lot easier to develop new models and experiment with different loss functions and learning algorithms.

For any questions/comments/concerns email [Colin](mailto:colincsl@gmail.com).

### Usage

There are two use cases of our code: running our ICRA model or developing your own. In order to use our ICRA model I encourage you to look through the ``eval_ICRA.py`` example code where you will see the API is similar to Scikit-Learn and pyStruct. This can be used to recreate the results from the paper. See more details about the datasets below and more details about the model in the corresponding paper. 

#### Implemented Models

Example temporal models located in ```LCTM/models.py``` and their corresponding energies/potentials in ```LCTM/energies```. The example models are: 

* ```ChainModel``` (a Skip-Chain Conditional Random Field (SC-CRF))
* ```ConvModel``` (an SC-CRF with temporal convolutional weights)
* ```LatentChainModel``` and ```LatentConvModel``` (Latent generalizations of the previous)
* ```SegmentalModel``` (Based on a Semi-Markov CRF)
* ```PretrainedModel``` (use with pretrained unary like the output of a CNN)

It should be straightforward to run on your own dataset after looking through the exampe code. There are also a set of convenient wrappers I use in ```LCTM/datasets.py``` to handle evaluation splits between different datasets.

#### Custom Models

As in any structured prediction model there are three components: the model, inference, and learning. We support recent stochastic gradient descent methods from the deep learning literature like RMSProp and Adagrad. 

It is easy to implement your own models by extending the class ```CorePotential``` in the ```LCTM/energies``` files. Let's look at the pairwise transtion potential as an example:

```
class pairwise(CorePotential):
    def __init__(self, skip=1, name='pw'):
        self.skip = skip
        self.name = name

    def init_weights(self, model):
        return np.random.randn(model.n_nodes, model.n_nodes)
    
    def cost_fcn(self, model, Xi, Yi):
        return pw_cost(Yi, model.n_nodes, self.skip)

    def compute(self, model, Xi, score):
        return compute_pw(score, model.ws[self.name], self.skip)
```

You must initialize the weight matrix, define a cost function, and a score function. The cost is often a function of the data and labels and the score is a function of the data and weights. In this case we also include the skip-length parameters. Here, Xi and Yi correspond to individual sequences of size (TxF) and (Tx1) where T is the number of time-steps and F is the number of features.

Once you have implemented a potential you can add it to your model in ```LCTM/models.py```. Depending on the potential this may be all you need to do, however, if for example your potential requires higher order terms you may need to implement a new method of inference. For clarification see the Nowozin and Lampert monograph referenced below.

#### Inference

We include two inference methods for time-series data: a skip-chain generalization of the Viterbi algorithm and the segmental inference algorithm of [Lea et al](http://arxiv.org/abs/1602.02995). You can see example usage in the ICRA example file.

Note that to make this library more flexible we actually perform approximate inference for the skip-frame model by doing a forward pass, backward pass, and then max over the score at each timestep. This allowed us to trivially add new components like a multi-skip generalization of the skip-chain model. In our experiments the difference in performance is neglible (~0.1% difference in accuracy) compared to exact inference for linear- (or skip-) chain models. 


### Data and reference material

This work was developed for the following paper and has since been used on several other datasets as well: 

[Learning Convolutional Action Primitives for Fine-grained Action Recognition](http://colinlea.com/docs/pdf/2016_ICRA_CLea.pdf). Colin Lea, Rene Vidal, Greg Hager. ICRA 2016

In the ICRA paper we apply this model to two datasets: [JIGSAWS](http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) and [50 Salads](http://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/). JIGSAWS contains the kinematics (position, velocity, gripper state) of daVinci robot end effectors and is used for analyzing surgical training tasks like suturing. 50 Salads contains accelerometers attached to 10 kitchen tools and is used to model complex cooking tasks. While both datasets also include video, in this paper we only evaluated on the sensor data. For the video-based generalization see our [followup work](http://arxiv.org/abs/1602.02995).

Per the request of one of the dataset authors we can only release the features upon request. Email [Colin](mailto:colincsl@gmail.com) for the files.

For a background on structured prediction I suggest reading the monograph by Nowozin and Lampert: [Structured Learning and Prediction in Computer Vision](http://pub.ist.ac.at/~chl/papers/nowozin-fnt2011.pdf)

