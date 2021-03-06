Doc2vec from scratch in PyTorch
===============================

[This notebook](https://nbviewer.jupyter.org/github/cbowdon/doc2vec-pytorch/blob/master/doc2vec.ipynb) explains how to implement `doc2vec` using PyTorch. It's aimed at relative beginners, but basic understanding of word embeddings (vectors) and PyTorch are assumed.

The implementation we end up with is hopefully correct but definitely not perfect. There's room for improvement in efficiency and features. Plus I have no intention of maintaining this, so please use a more established implementation for "serious" work. If you would like a PyTorch implementation, I recommend [this one](https://github.com/inejc/paragraph-vectors), from which this borrows extensively.

_Visualization of the notebook model's results for the BBC dataset:_

![](./img/bbc_pca_business_sport.png)

