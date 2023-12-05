#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def learnML_parameters(X,Y):
    mu = np.zeros((10, 256))
    sig = np.zeros((10, 256))
    for i in range(10):
        Xyi = X[Y==i]
        mu[i] = Xyi.mean(0)
        sig[i] = Xyi.std(0)
    return mu, sig

def log_likelihood(img, mu, sig, defsig = 1e-5):
    where = np.where(sig == 0)[0]
    mask = np.ones_like(img, dtype=bool)
    if defsig >= 0: sig[where] = defsig
    else: mask[where] = False
    return -(np.log(2 * np.pi * (sig[mask] ** 2)) + ((img[mask] - mu[mask]) / sig[mask]) ** 2).sum() / 2

def classify_image(img, mu, sig, defeps=1e-5):
    return np.argmax([log_likelihood(img, mu[i], sig[i], defeps) for i in range(10)])

def classify_all_images(X, mu, sig, defeps=1e-5):
    return np.array([classify_image(x, mu, sig, defeps) for x in X])

def matrice_confusion(Y, Y_hat):
    mat = np.zeros((10, 10))
    for yi in range(10):
        u, c = np.unique(Y_hat[Y==yi], return_counts=True)
        mat[yi, u] = c
    return mat

def classificationRate(Y_train,Y_train_hat):
    m = matrice_confusion(Y_train, Y_train_hat)
    return np.where(Y_train == Y_train_hat, 1, 0).mean()

def classifTest(X_test,Y_test,mu,sig,defeps):
    Y_test_hat = classify_all_images(X_test, mu, sig, defeps)
    m = matrice_confusion(Y_test, Y_test_hat)
    plt.figure()
    plt.imshow(m)
    return np.where(Y_test != Y_test_hat)

