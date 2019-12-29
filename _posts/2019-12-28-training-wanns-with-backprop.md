---
layout:     post
title:      Training Weight Agnostic Neural Networks with Backpropagation
date:       2019-12-28
summary:    I train WANNs with Adam and SGD to classify MNIST handwritten digits, and compare the results with those of a black-box optimizer used in <a href='https://papers.nips.cc/paper/8777-weight-agnostic-neural-networks'>the original paper</a>.
---


<!--
## Intro
-->

The recent paper [*Weight Agnostic Neural Networks*](https://papers.nips.cc/paper/8777-weight-agnostic-neural-networks) by Gaier and Ha proposes a neural architecture search algorithm that evolves neural networks for solving learning tasks. The search algroithm is inspired by [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), a popular neuroevolution algorithm. But instead of searching for weight values and network topologies simultaneously as done in NEAT, the WANN evolution procedure searches only for network topologies. Each topology in the search phase is evaluated by the average performance of networks with a range of tied weights (and therefore weight agnostic). More details can be found in the paper which is published [here](https://papers.nips.cc/paper/8777-weight-agnostic-neural-networks) and [here](https://arxiv.org/abs/1906.04358), along with a [website](https://weightagnostic.github.io/), a [blog](https://ai.googleblog.com/2019/08/exploring-weight-agnostic-neural.html) and [open-source code](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease).

WANNs are shown to perform reasonably well on several reinforcement learning tasks even though they're sparsely connected and use tied weights. 
The paper further demonstrates the effectiveness of WANNs in classifying MNIST handwritten digits by reformulating the classification task as a reinforcement learning problem. 
In particular, an ensemble of WANNs with tied weights performs as well as a standard neural network with thousands of parameters. 
And when trained with a black-box optimizer called PEPG,[^1] a single WANN is able to achieve a respectable test accuracy of 94.2%. 


I'm really excited about the paper: it not only speeds up neural architecture search by eliminating the inner optimization loop for weight training, but also brings to light the potential of sparse neural networks with minimal degrees of freedom in weights. I'm also curious that in the MNIST experiment, WANNs are trained with a black-box optimizer instead of the usual backpropagation even though the network architecture is fully differentiable. For this, the paper reports an interesting observation: training WANNs with backpropagation in the classification formulation does not fare as well as 
PEPG in the reinforcement learning formulation. Unfortunately, I wasn't able to find more details about this observation, so I decided to test it out myself and present some preliminary results here.

## Experiment setup
Let me first describe my experiment setup.

General setup:
* Dataset: MNIST256, a downsampled and preprocessed version of MNIST provided by the authors of the paper along with [their code](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease)
* WANN architecture: the same 1849-connection topology as used in the paper and provided by the authors (I'll call it *MNIST-WANN* in short) 
* I use [autograd](https://github.com/HIPS/autograd) to obtain gradients from MNIST-WANN[^2]


Optimizers:<br>
* Adam: learning rate=0.01, beta1=0.99 and beta2=0.999
* SGD: learning rate=2.0, momentum=0.9 

I've only attempted to tune learning rates with all other hyperparameters set by default. The implementation of both optimizers is provided by the authors.

Model training and evaluation:
* I train MNIST-WANN on the first 50000 samples in the training set, validate it on the remaining 10000 samples in the training set, and test it on the test set that contains 10000 samples
* For each optimizer, I compute its results as averages of 5 runs with different random seeds unless otherwise specified
* Loss function: cross-entropy loss
* Weight initialization: He uniform initialization
* Minibatch size: 128
* Epochs: 10

I'm now ready to present my experiment results.


## Accuracy

The following plot shows the accuracy of Adam and SGD as training progresses:

{:refdef: style="text-align: center;"}
![Training/validation accuracy](/assets/accuracy.png)
{: refdef}

In the above plot, the upper and lower boundaries of shadows represent the max and min accuracy at each epoch (over 5 runs), respectively. Compared to SGD, Adam enjoys more stability in training and validation accuracy. It appears that the trained MNIST-WANN is underfitting for both optimizers, however, and there seems to be room for improvement. 

I summarize the results of the 10th epoch in the table below.
As you can see, there is a gap between test accuracy of Adam/SGD and PEPG (94.2% as reported in the paper), though we should keep in mind that the MNIST-WANN trained by Adam/SGD is underfitting.


| Optimizer| Training    | Validation|Testing|
|:--------:|:-----------:|:---------:|:-----:|
|      Adam|      93.4%  |     93.9% | 93.5% |
|       SGD|      91.9%  |     92.8% | 92.2% |


## Landscape analysis

I’m also curious about what the solution found by each optimizer looks like. More precisely, I’m interested in the shape of the loss surface around each solution. One simple approach for visualizing a loss surface is to plot a 2D contour described by the following equation:

$$
f(\alpha_1, \alpha_2; \sigma) = \cal{L}(\hat{x} + \sigma \alpha_1 \delta_1 + \sigma \alpha_2 \delta_2), \quad \delta_1, \delta_2 \sim \mathcal{N}(\mathrm{0},\mathit{I}),
$$

where 
* $$\cal{L}$$ is the objective value (cross-entropy loss in this case) computed on the training set, 
* $$\hat{x}$$ is the point at which the contour plot is centered (the MNIST-WANN weight parameters optimized by PEPG, Adam or SGD),
* $$\alpha_1, \alpha_2$$ and $$\sigma$$ are variables that control our "view" of the landscape (smaller absolute values lead to a smaller neigborhood around $$\hat{x}$$ and a finer-grained landscape), and
* $$\delta_1, \delta_2$$ are vectors for perturbing $$\hat{x}$$ in constructing the landscape; they're sampled independently from the zero-mean normal distribution with identity covariance matrix. 


For Adam and SGD, I choose $$\hat{x}$$ to be the weights at the 10th epoch of training from one run, and for PEPG I use the weights provided by the authors. I sample the vectors $$\delta_1, \delta_2$$ only once and fix them for all experiments. Below I plot the function $$f(\alpha_1, \alpha_2; \sigma)$$ with $$\alpha_1, \alpha_2 \in \{-1, -0.975, \ldots, 0, \ldots, 0.975, 1\}$$ and $$\sigma \in \{0.1, 0.2, 0.3, 0.4\}$$ for each of the three optimizers, by varying optimizers across columns and $$\sigma$$ across rows:

{:refdef: style="text-align: center;"}
![Landscape](/assets/landscape.png)
{: refdef}

The landscapes around solutions of all three optimizers look smooth at $$\sigma=0.1, 0.2$$, with those for Adam and SGD taking shape of almost perfect ellipses. As $$\sigma$$ increases, the landscape for PEPG becomes more “rugged”, whereas those for Adam and SGD remain elliptical (though less perfect than at smaller $$\sigma$$'s). It is intriguing that, despite its irregular landscape shape, the solution of PEPG generalizes better than those of Adam and SGD. 


## Discussions
In this article, I've presented some preliminary results for training WANNs with Adam and SGD for classifying MNIST digits. Both optimizers do pretty well, though still falling short compared to PEPG used in the paper. I've also studied landscapes around solutions of PEPG/Adam/SGD, and shown that the PEPG solution has an irregular landscape around it while generalizing better.  

I think WANNs are very interesting and certainly deserve further exploration.
For now, the preliminary results shown above suggest that training of MNIST-WANN can be improved by running more epochs, tuning the momentum of SGD, and/or trying other optimizers such as RMSprop. It'd be also interesting to investigate why the PEPG solution produces the rugged landscape in contrast with Adam and SGD.


__Acknowledgement__: I'd like to thank the authors of the paper for generously open-sourcing their code.

<ins>*Update (12/29/2019)*</ins>:
I've tried to train MNIST-WANN for 20 more epochs (so 30 epochs in total). Below is the training progress over the 30 epochs:

{:refdef: style="text-align: center;"}
![Training/validation accuracy](/assets/accuracy.30.png)
{: refdef}

I also summarize the results for Adam at the 29th epoch (which achieves the highest validation accuracy) and those for SGD at the 30th epoch, averaged over 5 runs:

| Optimizer| Training    | Validation|Testing|
|:--------:|:-----------:|:---------:|:-----:|
|      Adam|      94.2%  |     94.4% | 94.0% |
|       SGD|      93.0%  |     93.6% | 93.2% |

<br>
In view of [the results of the 10th epoch](#accuracy), it appears that training MNIST-WANN for more epochs does help to ameliorate the underfitting issue and improves the test accuracy of Adam to 94%, close to the 94.2% of PEPG reported in the paper. The SGD-trained MNIST-WANN still underfits even after 30 epochs of training; on the other hand, the gap between training and validation accuracy narrows a bit, and the results become more stable after 20 epochs.

---

[^1]: PEPG used in the paper (also known as population-based REINFORCE therein) is a variant of the algorithm described in [this paper](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=A64D1AE8313A364B814998E9E245B40A?doi=10.1.1.180.7104&rep=rep1&type=pdf), which in turn is a black-box optimizer for reinforcement learning. The paper uses an open-source implementation from [estool](https://github.com/hardmaru/estool).

[^2]: It is possible to reimplement WANNs using standard neural network libraries such as PyTorch and Tensorflow. However, I find it most straightforward to use [autograd](https://github.com/HIPS/autograd) on top of the existing code: all I need to do is replace some "import numpy as np" statements by "import <span class="blue">autograd</span>.numpy as np" along with minor tweaks.
