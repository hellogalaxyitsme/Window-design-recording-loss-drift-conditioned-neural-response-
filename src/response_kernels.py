"""Fixed likelihood-ratio alternatives and bounded-score evidence."""
import numpy as np
from scipy.special import logsumexp
from conditional_response import BET_FRACTIONS

ALTERNATIVES=np.array([.2,1/3,.5,2/3,.8,1.25,1.5,2,3,5])

def wealth_matrix(scores, repeats, trials, epsilon=0., direction='increase'):
    v=scores['observed'].reshape(repeats,trials); m=scores['mean'].reshape(repeats,trials)
    active=scores['informative'].reshape(repeats,trials)
    if direction=='decrease': v=1-v; m=1-m
    upper=np.minimum(1,m+epsilon); active=active&(upper>1e-14)&(upper<1-1e-14)
    lam=np.divide(BET_FRACTIONS[None,None,:],upper[:,:,None],
        out=np.zeros((repeats,trials,len(BET_FRACTIONS))),where=upper[:,:,None]>0)
    factor=1+lam*(v-upper)[:,:,None]; factor[~active]=1
    return logsumexp(np.cumsum(np.log(factor),axis=1),axis=2)-np.log(len(BET_FRACTIONS))

