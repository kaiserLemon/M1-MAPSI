# Perotti-valle Rayan 28614730
# Torres Andy 21304450

import numpy as np
from tqdm import tqdm


def exp(rate):
    u = np.random.rand(*rate.shape)
    rate=np.where(rate>0,rate,1e-200)
    return -np.log(1.0-u)/rate


def get_kr_for(graph,fr,to):
    _,gk,gr=graph
    k=np.array([[gk.get((i, int(v)),0) for v in to] for i in fr])
    r=np.array([[gr.get((i, int(v)),0) for v in to] for i in fr])
    return k,r


def simulation(graph, sources, maxT):
    names, k, r = graph
    nbNodes = len(names)
    infectious = np.zeros(nbNodes) + maxT 
    infectious[sources] = 0             
    times = infectious.copy()
    while True: 
        contaminant = infectious.argmin()        
        Tref = infectious[contaminant]        
        infectious[contaminant] = maxT
        if Tref >= maxT:
            break
        cibles = np.nonzero(times > Tref)[0]
        params = get_kr_for(graph, [contaminant], cibles) # récupération des paramètres vers les cibles
        contamination = np.random.random(len(params[0][0])) < params[0][0]
        t = Tref + exp(params[1][0] * contamination)
        times[cibles] = np.minimum(t, times[cibles])
        infectious[cibles] = np.minimum(t, times[cibles])

    return times


def getProbaMC(graph,sources, maxT, nbsimu=100000):
    names,gk,gr=graph 
    nbNodes=len(names)
    rInf= np.zeros(nbNodes)
    for _ in range(nbsimu):
        times = simulation(graph, sources, maxT)
        vs = np.where(times < maxT)[0]
        rInf[vs] += 1
    return rInf / nbsimu


def getPredsSuccs(graph):
    names,gk,gr=graph
    nbNodes=len(names)
    preds={}
    succs={}
    for (a,b),v in gk.items():
        s=succs.get(a,[])
        s.append((b,v,gr[(a,b)]))
        succs[a]=s
        p=preds.get(b,[])
        p.append((a,v,gr[(a,b)]))
        preds[b]=p
    
    return (preds,succs)


def compute_ab(v, times, preds, maxT, eps=1e-20):
    preds=preds.get(v,[])
    t=times[v]
    if t==0:
        return (1,0)
    a=eps
    b=0
    if len(preds)>0:
        c,k,r=map(np.array,zip(*preds))
        t_j = times[c]
        j = np.nonzero(times[c] < t )[0]
        alpha = k*r*np.exp(-r*(t - t_j))
        beta =  alpha/r + 1 - k
        b = np.log(beta[j]).sum()
        if t < maxT:
            a = max(eps, (alpha[j]/beta[j]).sum())
        else: 
            a = 1.0
    return (a,b)


def compute_ll(times,preds, maxT):
    eps = 10e-20
    sa = []
    sb = []
    ll = 0
    for i in range(len(times)):
        a, b = compute_ab(i, times, preds, maxT, eps)
        sa.append(a)
        sb.append(b)
        ll += b + np.log(a)
    return ll, sa, sb



def addVatT(v, times, newt, preds, succs, sa, sb, maxT):
    eps = 10e-20
    t = times[v]
    if t >= 0:
        raise Error("v must have been removed before")
    succs = succs.get(v, [])
    times[v] = newt
    t = newt
    sa[v], sb[v] = compute_ab(v, times, preds, maxT)
    if len(succs) > 0:
        c, k, r = map(np.array, zip(*succs))
        tp = times[c]
        which = (tp > t)
        tp = tp[which]
        dt = tp-t
        k = k[which]
        r = r[which]
        c = c[which]
        rt = -r*dt
        b1 = k*np.exp(rt)
        b = b1+1.0-k
        a = r*b1
        a = a/b
        b = np.log(b)
        sa[c] = sa[c] + np.where(tp < maxT, a, 0.0)
        sa[c] = np.where(sa[c] > eps, sa[c], eps)
        sb[c] = sb[c] + b
        sb[c] = np.where(sb[c] > 0, 0, sb[c])
        
        
def logsumexp(x, axis=-1):
    x_star = np.max(x)
    x = x - x_star
    x = x_star + np.log(np.exp(x).sum(-1))
    return x


def gb(graph, infections, maxT, burnin=1000,nbEpochs=10000,k=100,k2=50, ref=None):
    preds, succs = getPredsSuccs(graph)
    nbNodes = len(graph[0])
    times = np.array([maxT]*nbNodes,dtype=float)
    _ , sa, sb = computell(times, preds, maxT, eps)

    l = list(range(nbNodes))
    for infecte in infections:
        times[infecte[0]] = infecte[1]
        l.remove(infecte[0]) 
    print('Burnin')
    for _ in tqdm(range(burnin)):
        v = np.random.choice(l)
        sampleV(v, times, preds, succs, sa, sb, k, k2, maxT)

    print('Collecting times')
    print()
    n = 0
    rInf = np.zeros_like(times)
    pbar = tqdm(range(nbEpochs - burnin))
    for _ in pbar:
        v = np.random.choice(l)
        sampleV(v, times, preds, succs, sa, sb, k, k2, maxT)
        rInf += (times < maxT)
        n += 1
        if n%1000 == 0:
            mse = ((rInf - ref)**2).sum() / n
            pbar.set_postfix(mse=mse)
    rate = rInf/n
    return rate



































    
    
    
    
    
    
    
    
    
    
    
    
    