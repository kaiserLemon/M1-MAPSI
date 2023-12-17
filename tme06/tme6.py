#Perotti-Valle Rayan 28614730
#Torres Andy 21304450

import numpy as np


def discretise(X,n):
    intervalle = 360/n
    res=[]
    for x in X:
        x = np.floor(x/intervalle)
        res.append(x)
    return res


def groupByLabel(Y):
    return {i:np.where(Y==i)[0] for i in np.unique(Y)}

    
def learnMarkovModel(X,d):
    A = np.zeros((d,d))
    Pi = np.zeros(d)
    x = discretise(X,d)
    
    for i in x:
        Pi[int(i[0])]+=1
        for j in range(len(i)-1):
            A[int(i[j]),int(i[j+1])]+=1
    
    
    A = A/np.maximum(A.sum(1).reshape(d,1),1)
    Pi = Pi/Pi.sum()
    return Pi,A


def learn_all_MarkovModels(X,Y,d):
    groups = groupByLabel(Y)
    lettre={}
    for y in groups:
        lettre[y]=learnMarkovModel([X[i] for i in groups[y]],d)
           
    return lettre

def stationary_distribution_freq(X,n):
    res=np.zeros(n)
    com=0
    for i in X:
        for j in i:
            res[int(j)]+=1
            com+=1
            
    for k in range(n):
        res[k]=res[k]/com
   
    return res

def stationary_distribution_sampling(pi,A,N):
    num_states = len(pi)
    current_distribution = pi.copy()
    for _ in range(N):
        next_state = np.random.choice(range(num_states), p=current_distribution)       
        current_distribution = np.dot(current_distribution, A)

    return current_distribution

def stationary_distribution_fixed_point(A, epsilon):
    num_states = A.shape[0]
    current_distribution = np.random.rand(num_states)  # Initialisation aléatoire

    while True:
        next_distribution = np.dot(current_distribution, A)
        error = np.square(np.subtract(next_distribution, current_distribution)).mean()

        if error < epsilon:
            return next_distribution  # Retourne le point fixe trouvé

        current_distribution = next_distribution

def stationary_distribution_fixed_point_VP(A):
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = np.where(np.isclose(eigenvalues, 1.0))[0]  
    eigenvector = eigenvectors[:, idx]

    stationary_distribution = eigenvector / np.sum(eigenvector)

    while True:
        next_distribution = np.dot(stationary_distribution, A)
        error = np.square(np.subtract(next_distribution, stationary_distribution)).mean()

        if error < epsilon:
            return next_distribution  # Retourne le point fixe trouvé

        stationary_distribution = next_distribution

def logL_Sequence(s, Pi, A):
    somme = np.log(Pi[int(s[0])])
    for i in range(1, len(s)):
        somme += np.log(A[int(s[i-1]), int(s[i])])
    return somme
     
def compute_all_ll(X,models):
    return np.array([[logL_Sequence(X[i], *models[k]) for i in range(len(X))] for k in models])

def accuracy(ll,Y):
    res=ll.argmax(0)
    d=np.zeros(Y.shape)
    for n,c in enumerate(np.unique(Y)):
        d[Y==c]=n
    return np.where(res!=d,0.,1.).mean()

def learn_all_MarkovModels_Laplace(X_train,ytrain,d):
    return null
    