#Perotti-Valle Rayan 28614730
#Torres Andy 21304450

import numpy as np
import os # ajout de bibliothèque utile avant la méthode
import pickle as pkl

def normale_bidim(x,mu,sig):
    return 1./(2*np.pi*np.sqrt(np.linalg.det(sig)))*np.exp(-0.5*(x-mu)@np.linalg.inv(sig)@(x-mu).T)


def estimation_nuage_haut_gauche():
    return ([4.25,80.],[[0.2,1.3],[1.3,50.]])


def init(X):
    pi=[0.5,0.5]
    mu1=X.mean(0)+[1,1]
    mu2=X.mean(0)+[-1,-1]
    mu= np.vstack((mu1,mu2))
    
    sig=np.zeros((2,2,2))
    sig[0,:,:]=np.cov(X.T)
    sig[1,:,:]=np.cov(X.T)
    
    return pi,mu,sig

def Q_i(X, pi, mu, Sig):
    rep=np.zeros((5,2))
    
    p=np.array([[normale_bidim(x,mu[i],Sig[i])*pi[i] for x in X] for i in range (len(pi))])
    q=p/p.sum(0)
    return q


def update_param(X, q, pi, mu, Sig):
    new_pi=np.zeros(len(pi))
    new_mu=[]
    new_sig=[]
    
    for k in range(len(pi)):
        new_pi[k]=(q[k]).sum()
    new_pi=new_pi/new_pi.sum() 
        
    for i in q:
        new_mu.append((i.reshape(-1,1)*X).sum(0)/i.sum())
        
    new_mu=np.array(new_mu)
    
    for k in range(len(pi)):
        new_sig.append((q[k]*(X-new_mu[k]).T@(X-new_mu[k]))/q[k].sum())
    new_sig=np.array(new_sig)
        
    return new_pi,new_mu,new_sig


def EM(X,initFunc=init,nIterMax=100,saveParam=None):
    
    if saveParam is not None:                                         # détection de la sauvergarde
        if not os.path.exists(saveParam[:saveParam.rfind('/')]):     # création du sous-répertoire
            os.makedirs(saveParam[:saveParam.rfind('/')])
    
    pi,mu,sig=initFunc(X)
    for i in range (nIterMax):
        q=Q_i(X,pi,mu,sig)
        new_pi,new_mu,new_sig= update_param(X,q,pi,mu,sig)
        if (np.abs(new_mu-mu).sum()<1e-3).all():
            break
        else:
            pi=new_pi
            mu=new_mu
            sig=new_sig
            if saveParam !=None:
                pkl.dump({'pi':new_pi, 'mu':new_mu, 'Sig': new_sig},\
                         open(saveParam+str(i)+".pkl",'wb'))                 # sérialisation
    return i,new_pi,new_mu,new_sig


def init_4(X):
    pi = np.array([0.25, 0.25, 0.25, 0.25])

    m = np.mean(X, axis=0)
    mu2 = m.copy()
    mu2[0] -= 1
    mu2[1] += 1
    mu3 = m.copy()
    mu3[0] += 1
    mu3[1] -= 1
    mu = np.stack((m+1, mu3, mu2, m-1))
    
    Sig = np.cov(X.T)
    Sig = np.stack((Sig, Sig, Sig, Sig))
    return pi,mu,Sig


def bad_init_4(X):
    pi = np.array([0.25, 0.25, 0.25, 0.25])

    m = np.mean(X, axis=0)
    m2 = m.copy()
    m2[0] += 4
    m2[1] += 2
    m3 = m.copy()
    m3[0] += 3
    m3[1] += 4
    m4 = m.copy()
    m4[0] -= 5
    mu = np.stack((m2, m3, m, m4))
    
    Sig = np.cov(X.T)
    Sig = np.stack((Sig, Sig, Sig, Sig))
    return pi, mu, Sig
    
def logpobsBernoulli(X,theta):
    seuil=1e-5
    theta=np.maximum(np.minimum(1-seuil,theta),seuil)
    logp=(X*np.log(theta)+(1-X)*np.log(1-theta)).sum()
    
    return np.array(logp)


def init_B(X):
    pi=np.zeros((10,))+1/10
    
    theta=np.array([X[3*i:(3*(i+1)),:].mean(0) for i in range(10)])
    return pi, theta

def Q_i_B(X, pi, theta):
    rep= np.array([[logpobsBernoulli(X[i],theta[j])  for j in range(len(theta))] for i in range (len(X))])
    repm= np.max(rep)
    
    replog= repm+ np.log((np.exp(rep-repm)*pi).sum(1))
    res=[]
    for i in range(len(theta)):
        res.append(rep[:,i]+np.log(pi[i])-replog)
    res=np.array(res)
  
    return np.exp(res)


def update_param_B(X, q, pi,theta):
    new_pi=np.zeros(len(pi))
    new_theta=[]
   
    for k in range(len(pi)):
        new_pi[k]=(q[k]).sum()
    new_pi=new_pi/new_pi.sum() 
        
    for i in q:
        new_theta.append((i.reshape(-1,1)*X).sum(0)/i.sum())
        
    new_theta=np.array(new_theta)
   
    return new_pi,new_theta


def EM_B(X,initFunc=init_B,nIterMAx=100,saveParam=None):
    epsilon= 1e-3
    pi,theta=init_B(X)
    
    for i in range (nIterMAx):
        q=Q_i_B(X,pi,theta)
        new_pi,new_theta= update_param_B(X,q,pi,theta)
        if (np.abs(theta-new_theta).max()<1e-3):
            break
        else:
            pi=new_pi
            theta=new_theta
            
    return i,new_pi,new_theta
    

def calcul_purete(X, Y, pi, theta):
    Y_p=Q_i_B(X,pi,theta)
    Y_hat=np.argmax(Y_p,0)
    
    repv=[]
    repc=[]
    for i in range (10):
        Y_hat_c=Y[Y_hat==i]
        val,count=np.unique(Y_hat_c, return_counts=True)
        repv.append(np.max(count))
        repc.append(np.sum(count))
        
    repv=np.array(repv)
    repc=np.array(repc)
    purete=repv/repc
    poids=repc/X.shape[0]
    
    return purete,poids
    
    
    
                    
    
    
    
    
    
