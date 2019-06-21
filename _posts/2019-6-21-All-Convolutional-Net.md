

---
layout: post
title:  "The All Convolutional Net review"
date:   2019-06-21 15:53:36 +0530
categories: Python
---

# 经典论文回顾（一）：The All Convolutional Net



# 1.Motivation

The vast majority of modern convolutional neural networks (CNNs) used for object recognition are built using the same principles: They use alternating convolution and max-pooling layers followed by a small number of fully connected layers.

Since all of these extensions and different architectures come with their own parameters and training procedures the question arises which components of CNNs are actually necessary for achieving state of the art performance on current object recognition datasets. We take a first step towards answering this question by studying the most simple architecture we could conceive: a homogeneous network solely consisting of convolutional layers, with occasional dimensionality reduction by using a stride of 2. 

## 2.Problem Setup

First of all, we assume that in general there exist three possible explanations why pooling can help in CNNs: 1) the p-norm makes the representation in a CNN more invariant; 2) the spatial dimensionality reduction performed by pooling makes covering larger parts of the input in higher layers possible; 3) the feature-wise nature of the pooling operation (as opposed to a convolutional layer where features get mixed) could make optimization easier. Assuming that only the second part – the dimensionality reduction performed by pooling is crucial for achieving good performance with CNNs (a hypothesis that we later test in our experiments) one can now easily see that pooling can be removed from a network without abandoning the spatial dimensionality reduction by two means: 

\1. We can remove each pooling layer and increase the stride of the convolutional layer that preceded it accordingly. 

\2. We can replace the pooling layer by a normal convolution with stride larger than one (i.e.  for a pooling layer with k = 3 and r = 2 we replace it with a convolution layer with corresponding stride and kernel size and number of output channels equal to the number of input channels) 

## 3.Methond

model replaces all5@5convolutions by simple3@3convolutions. This serves two purposes: 1) it unifies the architecture to consist only of layers operating on 3@3 spatial neighborhoods of the previous layer feature map (with occasional subsampling); 2) if max-pooling is replaced by a convolutional layer, then 3@3 is the minimum filter size to allow overlapping convolution with stride 2. 

![1560946132484](.assets/allconv.png)
