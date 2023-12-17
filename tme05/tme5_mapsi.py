# Torres Andy 21304450
# Perotti-Valle Rayan 28614730

#import pydotplus as pydot
import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy.stats
#import pyAgrum as gum
#import pyAgrum.lib.ipython as gnb
#import matplotlib.image as mpimg


def sufficient_statistics (data, dico, x, y, z ): #On ecrit une fonction qui prend en parametres les memes arguments que create_contingency_table
    res= utils.create_contingency_table (data, dico, x, y, z)#on utilise la fonction create_contingency_table pour recuperer le tableau
    nz=res.shape[0] #on recupere la valeur de z
    nx,ny=res[0][1].shape
    s=[]

    if len(data[0])<5*nx*ny*nz:
        return (-1,1)
    for N_z,N_XYz in res: #on parcourt les éléments du tableau pour faire la formule de l'enonce
        N_xz=N_XYz.sum(1) #on fait la somme des N_X_Y_z sur y pour obtenir N_x_z
        N_yz=N_XYz.sum(0) #on fait la somme des N_X_Y_z sur x pour obtenir N_y_z
        if N_z==0: 
            N_z=1
            nz=nz-1 #on retire un car on est tombé sur un cas avec 0
        frac=np.outer(N_xz,N_yz)/N_z
        den=np.where(frac==0,1,frac)
        s.append((N_XYz-frac)**2/den)
    s=np.array(s)
    d=(nx-1)*(ny-1)*nz
    return (s.sum(),d)


def indep_score(data, dico, x, y, z):
    res,dof=sufficient_statistics(data, dico, x, y, z)#on appelle la fonction précédente

    return scipy.stats.chi2.sf(res,dof)#on renvoie la p-value pour un degré de liberté avec le couple (x,dof) obtenu via la fonction précédente

def best_candidate ( data, dico, x, z, alpha ):#permet de la variable Y
    p_values = [] 
    if x==0:                                                                
        return []
    for y in range(x):
        p_values.append(indep_score(data, dico, x, y, z))
    p_values = np.array(p_values)
    p_values=np.where(p_values>alpha,1,p_values)

    return [] if np.min(p_values)==1 else [np.argmin(p_values)]
    
def create_parents(data, dico, x,alpha): #on crée la liste des parents pour un variable aléatoire x
    p=[]
    parent=best_candidate(data, dico, x,p,alpha)
    while len(parent)!=0:
        p+=parent
        parent=best_candidate(data, dico, x,p,alpha)
    return p

def learn_BN_structure (data, dico, a): #on crée un tableau contenant le liste des parents pour chaque noeuds
    return [create_parents(data, dico, x, a) for x in range(len(dico))] #on parcourt chaque noeud et on appelle la fonction create_parents