#Perotti-valle Rayan 28614730
#Torres Andy 21304450

import numpy as np
import matplotlib.pyplot as plt


def gen_data_lin(a, b, sig, N,Ntest):
    X_train = np.sort(np.random.rand(N))
    Y_train = a*X_train + b + np.random.randn(N)*sig
    X_test = np.sort(np.random.rand(Ntest))
    Y_test = a*X_test + b + np.random.randn(Ntest)*sig
    return X_train, Y_train, X_test, Y_test


def modele_lin_analytique(X_train, y_train):
    cov = np.cov(X_train,y_train,bias=True)
    ahat = cov[0,1]/cov[0,0]
    bhat = np.mean(y_train) - ahat * np.mean(X_train)
    return ahat,bhat


def calcul_prediction_lin(X_train,ahat,bhat):
    return ahat*X_train+bhat



def erreur_mc(y_test, yhat_test):
    return np.mean((y_test-yhat_test)**2)



def dessine_reg_lin(X_train, y_train, X_test, y_test,a,b):
    yhat_test = calcul_prediction_lin(X_test,a,b)
    plt.plot(X_test, y_test, 'r.',alpha=0.2,label="test")
    plt.plot(X_train, y_train, 'b.-',label="train")
    plt.plot(X_test,yhat_test, 'g',linewidth=3,label="prediction")
    plt.legend()
    plt.show()
    

    
def make_mat_lin_biais(X_train):
    N = len(X_train)
    return np.hstack((X_train.reshape(N,1), np.ones((N,1))))



def reglin_matriciel(Xe,y_train):
    A = Xe.T @ Xe
    B = Xe.T @ y_train
    return np.linalg.solve(A,B)


def calcul_prediction_matriciel(Xe,w):
    return Xe@w

def gen_data_poly2(a, b, c, sig, N, Ntest):
    X_train = np.sort(np.random.rand(N))
    Y_train = a*X_train**2 + b*X_train + c + np.random.randn(N)*sig
    X_test = np.sort(np.random.rand(Ntest))
    Y_test = a*X_test**2 + b*X_test + c + np.random.randn(Ntest)*sig
    return X_train, Y_train, X_test, Y_test


def make_mat_poly_biais(X):
    N=len(X)
    return np.hstack((X.reshape(N,1)**2,X.reshape(N,1),np.ones((N,1))))


def dessine_poly_matriciel(Xp_train,yp_train,Xp_test,yp_test,w):
    X_test=make_mat_poly_biais(Xp_test)
    yphat_test = calcul_prediction_matriciel(X_test,w)
    plt.plot(Xp_test, yp_test, 'r.',alpha=0.2,label="test")
    plt.plot(Xp_train, yp_train, 'b.-',label="train")
    plt.plot(Xp_test,yphat_test, 'g',linewidth=3,label="prediction")
    plt.legend()
    plt.show()
    

def descente_grad_mc(X, y, eps, nIterations):
    m=len(y)
    w = np.zeros(X.shape[1])
    allw = [w]
    for _ in range(nIterations):
        w = w - eps * 2 *(X.T @ ((X@w)-y))
        allw.append(w)
    allw = np.array(allw)
    return w, allw


def application_reelle(X_train,y_train,X_test,y_test):
    w = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
    print(f"{w=}")
    yhat   = X_train @ w
    yhat_t = X_test @ w
    print('Erreur moyenne au sens des moindres carrés (train):', erreur_mc(y_train, yhat))
    print('Erreur moyenne au sens des moindres carrés (test):', erreur_mc(y_test, yhat_t))
    return w, yhat, yhat_t


def normalisation(X_train, X_test):
    Xe=np.delete(X_train,len(X_train[0])-1,1)
    Xet=np.delete(X_test,len(X_test[0])-1,1)
    mu = Xe.mean(0)
    sig = Xe.std(0)
    sig[sig < 1e-8] = 1e-8
    Xn_train = (Xe - mu) / sig
    Xn_test = (Xet - mu) / sig
    Xn_train = np.hstack((Xn_train, np.ones((Xn_train.shape[0], 1))))
    Xn_test   = np.hstack((Xn_test, np.ones((X_test.shape[0], 1))))
    return Xn_train, Xn_test

    





    
    
    




