# Perotti-Valle Rayan 28614730
# Torres Andy 21304450

import numpy as np
import matplotlib.pyplot as plt

def labels_tobinary(Y, cl):
    res=np.zeros(len(Y))
    for i in range(len(Y)):
        if Y[i] ==cl:
            res[i]=1
            
    return res


def pred_lr(X, w, b):
    return 1./(1. + np.exp(-(X @ w+b)))


def classify_binary(Y_pred):
    return np.round(Y_pred)


def accuracy(Y_predb, Y_c):
    return np.where(Y_predb == Y_c, 1., 0.).mean()


def rl_gradient_ascent(X, Y_c, eta, niter_max):
    N,d = X.shape
    w = np.zeros(d)
    b = 0
    accs = np.zeros(niter_max)
    for i in range(niter_max):
        grad_w=X.T@(Y_c-pred_lr(X,w,b))
        tmp = np.sum(Y_c- pred_lr(X,w,b))
        w = w + eta*grad_w
        b = b + eta * tmp
        Y_predb=classify_binary(pred_lr(X,w,b))
        accs[i]=accuracy(Y_predb,Y_c)
    return w,b,accs,i


def visualization(w):
    plt.figure()
    plt.imshow(w.reshape(16,16), cmap='gray')
    plt.axis("off")
    plt.show()
    
    
def rl_gradient_ascent_one_against_all(X,Y, epsilon = 1e-3, niter_max=1000):
    N,d = X.shape
    classes = np.unique(Y)
    W = []
    B = []

    for c in classes:
        #Y_tmp = np.where(Y == c, 1., 0.)
        Y_tmp = labels_tobinary(Y, c)
        #print(Ytmp)
        w ,b, accs,it = rl_gradient_ascent(X, Y_tmp, epsilon, niter_max)
        W.append(w)
        B.append(b)

        """Y_pred = pred_lr(X, w, b)
        acc = accuracy(Y_pred, Y_tmp)
        """
        print(f"Classe : {c} \t acc train = {accs[-1]*100:.2f}%")

    return np.asarray(W).T, np.asarray(B).T


def classif_multi_class(Y_pred):
    classes_pred = np.argmax(Y_pred, axis=1)
    return classes_pred


def normalize(X):
    return X - 1


def pred_lr_multi_class(X, W, b):
    s = np.dot(X, W) + b
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)


def to_categorical(Y, K):
    return np.eye(K, dtype='uint8')[Y]


def rl_gradient_ascent_multi_class(X, Y, eta, numEp, verbose):
    K = 10
    W = np.zeros((X.shape[1], K))
    b = np.zeros(K)
    Y_c = to_categorical(Y, K)
    for i in range(numEp):
        W_old = W.copy()
        b_old = b.copy()
        Y_pred = pred_lr_multi_class(X, W_old, b_old)
        W = W + eta * X.T @ (Y_c - Y_pred)
        b = b + eta * np.sum(Y_c - Y_pred, axis=0)
        if verbose == 1 and (i == numEp - 1 or i % (numEp/10) == 0):
            print("epoch", i, "accuracy train =", accuracy(Y, classif_multi_class(Y_pred))*100, "%")

    return W, b


def rl_gradient_ascent_multi_class_batch(X, Y, tbatch=500, eta=0.2, numEp=200, verbose=1):
    K = 10
    W = np.zeros((X.shape[1], K))
    b = np.zeros(K)
    Y_c = to_categorical(Y, K)

    for epoch in range(numEp):
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), tbatch):
            batch_indices = indices[i:i + tbatch]
            X_batch = X[batch_indices]
            Y_batch = Y_c[batch_indices]
            W_old = W.copy()
            b_old = b.copy()
            Y_pred = pred_lr_multi_class(X_batch, W_old, b_old)
            W = W + eta * X_batch.T @ (Y_batch - Y_pred)
            b = b + eta * np.sum(Y_batch - Y_pred, axis=0)
        if verbose == 1 and (epoch == numEp - 1 or epoch % (numEp/10) == 0):
            Y_pred = pred_lr_multi_class(X, W, b)
            print("epoch", epoch, "accuracy train =", accuracy(Y, classif_multi_class(Y_pred))*100, "%")

    return W, b
    




            
    
    
    
            




