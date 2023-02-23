## [ICLR2023] How I Learned to Stop Worrying and Love Retraining
*Authors: [Max Zimmer](https://maxzimmer.org/), [Christoph Spiegel](http://www.christophspiegel.berlin/), [Sebastian Pokutta](http://www.pokutta.com/)*

This repository contains the code to reproduce the experiments from the ICLR2023 paper ["How I Learned to Stop Worrying and Love Retraining"](https://arxiv.org/abs/2111.00843).
The code is based on [PyTorch 1.9](https://pytorch.org/) and the experiment-tracking platform [Weights & Biases](https://wandb.ai). 
The code to reproduce semantic segmentation as well as NLP experiments will be added soon.


### Structure and Usage
Experiments are started from the following file:
- [`main.py`](main.py): Starts experiments using the dictionary format of Weights & Biases.

The rest of the project is structured as follows:
- [`strategies`](strategies): Contains all used sparsification methods.
- [`runners`](runners): Contains classes to control the training and collection of metrics.
- [`metrics`](metrics): Contains all metrics as well as FLOP computation methods.
- [`models`](models): Contains all model architectures used.
- [`utilities`](models): Contains useful auxiliary functions and classes.


### Citation
In case you find the paper or the implementation useful for your own research, please consider citing:

```
@inproceedings{zimmer2023how,
title={How I Learned to Stop Worrying and Love Retraining},
author={Max Zimmer and Christoph Spiegel and Sebastian Pokutta},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=_nF5imFKQI}
}
```
