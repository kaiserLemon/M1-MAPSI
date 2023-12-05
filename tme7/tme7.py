#Perotti-Valle Rayan 28614730
#Torres Andy 21304450

import numpy as np
import matplotlib.pyplot as plt

def learnHMM(allX,allS,N,K):
    A=np.zeros((N,N))
    B=np.zeros((N,K))
    for i in range(1,len(allS)):
        A[allS[i-1],allS[i]]+=1
    for i in range(N):
        A[i]=A[i]/np.sum(A[i])
    
    for i in range(1,len(allX)):
        B[allS[i-1],allX[i-1]]+=1
    for i in range(N):
        B[i]=B[i]/np.sum(B[i])

    return A,B

def viterbi(allx,Pi,A,B):
    """
    Parameters
    ----------
    allx : array (T,)
        Sequence d'observations.
    Pi: array, (K,)
        Distribution de probabilite initiale
    A : array (K, K)
        Matrice de transition
    B : array (K, M)
        Matrice d'emission matrix

    """

    ## initialisation
    psi = np.zeros((len(A), len(allx)),dtype=np.int8) # A = N
    psi[:,0]= -1
    delta = np.zeros((len(A), len(allx)))
    delta[:,0]=np.log(Pi)+np.log(B[:,allx[0]])
    
    etat_predits = np.zeros(len(allx),dtype=np.int8)
    T=allx.shape[0]
    K=A.shape[0]
    
    for t in range(1,T):
        delta_t1 =delta[:,t-1]
        logA=np.log(A.T)
        prod= delta_t1+logA
        
        psi[:,t]=np.argmax((delta_t1+logA),axis=1)
        max_row=[prod[i,psi[i,t]] for i in range(K)]
        logB= np.log(B[:,allx[t]])
        delta[:,t]=max_row+logB
        if(t%100000==0):
            print("t=",t,"delta[:,t]=",delta[:,t])
    etat_predits[T-1]=np.argmax(delta[:,T-1])
    for t in range(T-2,-1,-1):
        etat_predits[t]=psi[etat_predits[t+1],t+1]
    return etat_predits


def get_and_show_coding(etat_predits,annotation_test):
    etat_predits[etat_predits!=0]=1 
    annotation_test[annotation_test!=0]=1
    fig, ax = plt.subplots(figsize=(15,2))
    ax.plot(etat_predits[100000:200000], label="prediction", ls="--")
    ax.plot(annotation_test[100000:200000], label="annotation", lw=3, color="black", alpha=.4)
    plt.legend(loc="best")
    plt.show()
    return etat_predits,annotation_test


def create_confusion_matrix(true_sequence, predicted_sequence):
    m = np.zeros((len(np.unique(true_sequence)),len(np.unique(true_sequence))))
    for i,j in zip(predicted_sequence, true_sequence):
        if i==0:
            if j==0:
                m[i+1,j+1] += 1
            else:
                m[i+1,j-1]+=1
        else:
            if j==0:
                m[i-1,j+1] += 1
            else:
                m[i-1,j-1] += 1
    return m

def create_seq(N,Pi,A,B,states=[0,1,2,3],obs=['A','T','C','G']):
    state=[]
    ob=[]
    
    r = np.random.random()
    state.append(np.argwhere(np.cumsum(Pi) > r)[0][0])
    ob.append(obs[state[0]])
    for i in range(1,N):
        t = np.random.random()
        state.append( np.argwhere(np.cumsum(A[state[i-1]]) > r)[0][0])
        ob.append(obs[state[i]])
        
    for i in range(N):
        print(state[i],ob[i])

def get_annoatation2(annotation_train):
    res=np.copy(annotation_train)
    i=0
    q=4
    for i in range(len(annotation_train)):
        if annotation_train[i]!=0:
            if annotation_train[i-3]==1:
                if annotation_train[i+3]==1:
                    res[i]=4
                else:
                    res[i]=7
            if annotation_train[i-3]==2:
                if annotation_train[i+3]==2:
                    res[i]=5
                else:
                    res[i]=8

            if annotation_train[i-3]==3:
                if annotation_train[i+3]==3:
                    res[i]=6
                else:
                    res[i]=9
        
        
              
    return res
    
    
   

    
                        



    