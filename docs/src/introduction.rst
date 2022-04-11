Introduction
============

This document discusses the operation of RocketML's core C++ engine. The goal is to provide a high-level overview of the library's design 
principles, including a description of each one of its modules, as well as explain how to run applications directly from the command line.

The core engine implements several machine learning algorithms, which are realized as different *applications* in RocketML:

* Dense matrices

  * Linear regression (fit and predict)
  * Ridge regression (fit and predict)
  * Logistic regression (fit and predict)

* Sparse matrices (using LIBSVM data format):

  * Linear regression (fit and predict)
  * Logistic regression (fit and predict)
  * Truncated SVD (fit and transform)

* Image and video processing:

  * Truncated SVD (fit and transform)
  * Object detection
  * Image anomaly detection (KNN-based)

* Text mining:

  * Truncated SVD (fit and transform)
  * LDA topic modeling

* Raw data mining:

  * KNN-based anomaly detection
  * K-means clustering
  * Spectral clustering

* Deep neural networks:

  * Image classification
  * Image segmentation
  * PDE solvers

Each one of the applications above, as well as their configuration and usage modes, will be detailed in the next chapters.

