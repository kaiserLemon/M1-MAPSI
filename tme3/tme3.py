# Perotti-Valle Rayan 28614730
# Torres Andy 21304450

import numpy as np
import matplotlib.pyplot as plt


def learnML_parameters(X_train,Y_train): #on crée la fonction qui prend un tableau d'images d'une classe et renvoie un couple de tableaux
    mu = np.zeros((10,256))
    std = np.zeros((10,256))
    for i in range (10):
        mu[i] = X_train[Y_train==i].mean(0)#on calcule la moyenne
        std[i] = X_train[Y_train==i].std(0)#on calcule l'écart type
    return mu,std

def log_likelihood(img,mu,sig,defeps): #prend une image et un couple de paramètres et renvoie la log vraisemblance qu'aurait l'image
    if defeps > 0:
        sig = np.maximum(sig,defeps)
    
        return -1/2*np.sum(np.log(2*np.pi*(sig**2))+((img-mu)**2)/(sig**2))
    else:
        return -1/2*np.sum([np.log(2*np.pi*(sig[i]**2))+((img[i]-mu[i])**2)/(sig[i]**2) for i in range(len(img)) if (sig[i]>0)]) 
    
def classify_image(img,mu,sig,defeps): # prend une image et l'ensemble de paramètres déterminés et renvoie la classe la plus probable de l'image
    rep= np.array([log_likelihood(img,mu[i],sig[i],defeps) for i in range(10)])
    return np.argmax(rep)


def classify_all_images(X,mu,sig,defeps): #on effectue la fonction précédente pour un ensemble d'images 
    return np.array([classify_image(X[i],mu,sig,defeps) for i in range(len(X))])


def matrice_confusion(Y,Y_hat): #prend un vecteur d'etiquette et un de meme taille et renvoie la matrice de confusion
    rep= np.zeros((len(np.unique(Y)),len(np.unique(Y))))
    for i in range(len(Y)):
        rep[Y[i]][Y_hat[i]]+=1
    return rep

def classificationRate(Y_train,Y_train_hat):#on calcule le taux de bonne prédiction
    return np.where(Y_train == Y_train_hat, 1,0).mean()


def classifTest(X_test,Y_test,mu,sig,defeps):#on effectue les calculs avec cette fois ci les tableaux x_test et y_test
    Y_test_hat= classify_all_images(X_test,mu,sig,defeps)
    rep= matrice_confusion(Y_test,Y_test_hat)
    print(rep)
    print(f"Classification rate : {classificationRate(Y_test,Y_test_hat)=}")
    plt.figure(figsize=(3,3))
    plt.imshow(rep);
    return np.where(Y_test!=Y_test_hat)
    
                  
def binarisation(x):
    return np.where(x>0,1,0)


def learnBernoulli(X,Y):
    theta=[]
    for i in range(len(np.unique(Y))):
        theta.append(X[Y==i].mean(axis=0))
        
    return np.array(theta)


def logpobsBernoulli(X,theta,epsilon=1e-4):
    theta= np.where(theta == 0,epsilon, theta)
    theta= np.where(theta == 1,1-epsilon, theta)
    return np.array([np.sum(X*np.log(theta[i]) + (1-X)*np.log(1-theta[i])) for i in range(10)])


#Le résultats nous parait étrange, on observe des valeurs négatives, cela peut être du à la valeur de epsilon ou aux valeurs de theta.


def classifBernoulliTest(Xb_test,Y_test,theta): #on effectue les calculs avec les tableaux test 
    Y_test_hat=[np.argmax(logpobsBernoulli(Xb_test[i],theta)) for i in range (len(Xb_test))]
    
    rep=matrice_confusion(Y_test,Y_test_hat)
    print(rep)
    print(f"Classification rate : {classificationRate(Y_test,Y_test_hat)=}")
    
    plt.figure()
    plt.imshow(rep)
    
def learnGeom(X,Y,seuil=1e-4):
    return 1/learnBernoulli(X,Y)

def logpobsGeom(X,theta,seuil=1e-4):
    # On intialise une liste pour stocker les log-probabilités pour chaque classe
    log_probs = []

    for k in range(len(theta)):
        temp = theta[k] * (1 - 2 * seuil) + seuil
        log_prob_k = np.sum(np.log(temp) + (X - 1) * np.log(1 - temp))
        log_probs.append(log_prob_k)

    return log_probs
    

def classifyGeom(X,theta):
    return np.argmax(logpobsGeom(X,theta))
    



    

    
    
    
        
                   
    
    
    
