#TORRES ANDY 21304450
#Perotti-Valle Rayan 28614730


import numpy as np
import math
import matplotlib.pyplot as plt
import itertools



#question I.1 Loi de Bernoulli

def bernoulli(p):
    #On prend en argument p compris entre 0 et 1 et on renvoie 0 ave cun proba de 1-p et 1 avec une proba de p
    #Ici on vient tirer un nombre aléatoire compris entre 0 et 1 et si il est compris entre 0 et p inclus on renvoie 1 et 0 sinon
    return np.random.random() <=p

#question I.2 Loi de Binomiale

def binomiale(n,p):
    #On prend en argument une proba p compris entre 0 et 1 et un entier n correspondant au nombre de répétition de l'événement
    somme=0
    for i in range (n): #On répète n fois l'épreuve de Bernoulli et on compte le nombre de succès
        m = bernoulli(p)
        if (m ==1):
            somme+=1
    return somme


#question I.3 Histogramme de la loi Binomiale

def galton(l,n,p): #On crée un tableau de l cases où chaque cases correspond à une itération de binomiale avec n et p
    tab=np.zeros(l)
    for i in range (l):
        tab[i]=binomiale(n,p)
        
    return tab
        
def histo_galton(l,n,p): #On trace l'histogramme du tableau créer avec la fonction précédente
    tab=galton(l,n,p)
    
    somme=[] #on crée un tableau contenant les valeurs présentent dans notre tableau afin d'obtenir le nombre de valeurs diffèrentes dans celui-ci 
    for i in (tab):
        if i not in somme: #si notre valeur n'est pas deja dans le tableau on l'ajoute
            somme.append(i)
        
    plt.hist(tab,len(somme)); #construit l'histogramme avec notre tableau de valeur et le nombre de colonne correspondant au nombre de valeurs diffèrentes présent dans celui-ci 
    
def normale(k,sigma): #on prend un paramètre k impair et un paramètre sigma et on renvoie le tableau des k valeurs yi
    
    if k%2==0: #on s'assure que k est bien impair pour que notre tableau soit symétrique
        raise ValueError
  
    x=np.linspace(-2*sigma,2*sigma,k) #on crée un tableau x des valeur de k entre -2sigma et 2sigma
    y=np.zeros(len(x))

    for i in range (len(x)): #on parcourt x pour calculer les valeurs de y pour chaque k valeurs
        y[i]=(1/(math.sqrt(2*math.pi)*sigma))*math.exp((-1/2)*(x[i]/sigma)**2)

    return y

def proba_affine(k,slope): #on prend en paramètre k impair pour représenter comme dans la fonction précédente le tableau des k valeurs yi dépendant cette fois-ci de sa pente slope
    if k%2==0:
        raise ValueError
    
    y=np.zeros(k)
    somme=0.0
    if slope==0: #si la pente vaut 0 yi vaut 1/ki
        for i in range (len(y)):
            y[i]=1/k  
    else:
        for i in range (k): #sinon on calcul yi avec la formule donnée 
            y[i]=(1/k)+(i-((k-1)/2))*slope           
            
    for j in y:
        somme+=j
    if round(somme,0)!=1.0: #on vérifie que la pente soit ni trop grande ni trop petite
        raise ValueError
            
    return y

def Pxy(A,B): #on prend deux tableaux en entrée et on calcul la distribution jointe 
    rep=np.empty([len(A),len(B)]) #on crée un tableau à deux dimensions 
    for i in range (len(A)):
        for j in range (len(B)): #on parcourt les deux tableaux et on multplie les valeurs entre elles
            rep[i,j]=A[i]*B[j]  
    return rep 


def calcYZ(P_XYZT): #on prend le tableau P_XYZT en paramètre et on calcul le tableau P_YZ représentant P(Y,Z)
    P_YZ = np.empty([2,2])
    for i in range (2):
        for j in range (2): #on parcourt les valeurs du tableau P_XYZT afin de calculer la distribution avec la fomrule donnée 
            somme = P_XYZT[0][i][j][0] + P_XYZT[0][i][j][1] + P_XYZT[1][i][j][0] + P_XYZT[1][i][j][1]
            P_YZ[i,j]=somme
    
    return P_YZ
        

def calcXTcondYZ(P_XYZT): #Permet de calculer le tableau représentant la distribution P(X,Y|Y,Z)
    base = list(itertools.product([0, 1], repeat=4))
    P_XTcondYZ = np.zeros((2,2,2,2))#on crée un tableau à 4 dimensions 
    P_YZ = calcYZ(P_XYZT) #on utilise la fonction précédente pour trouver le tableau P_YZ dépendant de P_XYZT
    for x, y, z, t in base: #on parcourt toutes les valeurs du tableau afin d'appliquer la formule donnée
        P_XTcondYZ[x][y][z][t] = P_XYZT[x][y][z][t] / P_YZ[y][z]
    return P_XTcondYZ

def calcX_etTcondYZ(P_XYZT): #calcul la paire de tableaux représentant les distributions P(X|Y,Z) et P(T|Y,Z) 
    P_XTcondYZ = calcXTcondYZ(P_XYZT) #on utilise la fonction précedente pour recupérer le tableau
    return P_XTcondYZ.sum(3),P_XTcondYZ.sum(0)
    
def testXTindepCondYZ(P_XYZT,epsilon=1e-10):#permet de vérifier si X et T sont indépendantes conditionnellement à (Y,Z)
    isIndep = True
    base = list(itertools.product([0, 1], repeat=4))
    XcondYZ,TcondYZ = calcX_etTcondYZ(P_XYZT) #on récupère les deux tableaux 
    for x,y,z,t in base: #on parcourt les valeurs afin de vérifier la condition P(X,T|Y,Z)=P(X|Y,Z)xP(T|Y,Z)
        tmp = XcondYZ[x][y][z] * TcondYZ[y][z][t] - P_XYZT[x][y][z][t]
        if tmp > epsilon or tmp < -epsilon:
            isIndep = False
    return isIndep

def testXindepYZ(P_XYZT,epsilon=1e-10): #on vérifie si X et (Y,Z) sont indépendantes dans la distribution P_XYZT
    P_XYZ = P_XYZT.sum(3)
    P_X = P_XYZ.sum((1,2))
    P_YZ = P_XYZ.sum(0)
    isIndep = True
    base = list(itertools.product([0, 1], repeat=3))
    for x,y,z in base:
        tmp = P_X[x] * P_YZ[y][z] - P_XYZ[x][y][z]
        if tmp > epsilon or tmp < -epsilon:
            isIndep = False
    return isIndep

def conditional_indep(P,X,Y,Z,epsilon): # renvoie vrai si dans le Potential on peut lire l'indépendance conditionnelle
    pXYZ = P.margSumIn([X, Y, *Z])
    if Z:
        pXY_Z = pXYZ/pXYZ.margSumIn(Z)
    else:
        pXY_Z = pXYZ
    tmp = pXY_Z.margSumIn([X, *Z]) * pXY_Z.margSumIn([Y, *Z]) - pXY_Z
    return tmp.abs().max() < epsilon

def compact_conditional_proba(P,X): #prend en paramètre une probabilité jointe et une variable aléatoire et renvoie la probabilité conditionnelle P(Xin|K)
    epsilon = 1/(10**10)
    S = P.var_names.copy() #on reprend l'algo donnée dans l'énoncé
    S.remove(X)
    K = S.copy()
    for Y in S:
        K.remove(Y)
        if not conditional_indep(P,X,Y,K,epsilon): K.append(Y)
    pXK = P.margSumIn([X, *K])
    pX_K = pXK / pXK.margSumIn(K) if K else pXK
    return pX_K.putFirst(X)

def create_bayesian_network(P,x): #calcul la liste des P(Xi|Ki) selon une probabilité jointe donnée
    Ps = [] #on applique l'algo donnée dans l'énoncé
    S = P.var_names.copy()
    for i in range(len(S)-1,0,-1) :
        Q = compact_conditional_proba(P, S[i])
        #print(Q.var_names)
        Ps.append(Q)
        P = P.margSumOut(S[i])
    return Ps


def calcNbParams(Pjointe): #prend en paramètre la loi jointe et calcul le nombre de paramètre de la loi jointe et le nombre de paramètre dans le réseau bayésien 
    bayes = create_bayesian_network(Pjointe,0.0001) #on crée le réseau bayésien avec la fonction précédente
    return Pjointe.toarray().size,sum([p.toarray().size for p in bayes])
    
    
    
         
               
                       
                        
                     
                        
                        
   
    
    
    
