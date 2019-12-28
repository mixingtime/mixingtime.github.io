---
layout:     post
title:      Update, Training Weight Agnostic Neural Networks with Backpropagation
date:       2019-12-26 
summary:    Generic test.
---



<ins>*Updates (12/26/19)*</ins>

After writing up this article, I couldn't wait to run more tests with WANNs. And here I am to give an update. 

I've tried to train WANN for 20 more epochs (so 30 epochs in total). Below is the training progress over the 30 epochs.

{:refdef: style="text-align: center;"}
![Training/validation accuracy](/assets/accuracy.30.png)
{: refdef}

I summarize the results for Adam at the 29th epoch (which achieves the best validation accuracy) and those for SGD at the 30th epoch, averaged over 5 runs:

| Optimizer| Training    | Validation|Testing|
|:--------:|:-----------:|:---------:|:-----:|
|      Adam|      94.2%  |     94.4% | 94.0% |
|       SGD|      93.0%  |     93.6% | 93.2% |

It appears that training WANN for more epochs with Adam does help to ameliorate the underfitting issue and improves the test accuracy to 94%, close to the 94.2% of PEPG reported by the WANN paper. The SGD-tuned WANN still underfits even after 30 epochs of training; on the other hand, the gap between training and validation accuracy narrows a bit, and the result becomes more stable after 20 epochs.



<!--
Some math
$$\alpha_1$$

Here we go
statement:\$$ 5 + 5 $$

Here other
statement:\$$ f(\alpha_1, \alpha_2) = \cal{L}(\hat{x} + \alpha_1 \delta_1 + \alpha_2 \delta_2),$$

Accuracy plot image here:
{:refdef: style="text-align: center;"}
![Training/validation accuracy](/assets/accuracy.png)
{: refdef}


Landscape image here:

![Landscape](/assets/landscape.png)
-->