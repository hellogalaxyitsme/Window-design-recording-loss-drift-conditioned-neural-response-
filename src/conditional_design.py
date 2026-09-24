"""Exact one-dimensional conditional fibers for polynomial count trajectories.

The integer-kernel calculation is classical algebraic statistics. The design
criterion studied here is the minimum event count retained by conditioning.
"""
from fractions import Fraction
from functools import reduce, lru_cache
from math import gcd, lcm
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson, binom

DESIGNS = {
    'constant': dict(nodes=(0,1), centers_ms=(-350.,150.), degree=0),
    'linear': dict(nodes=(0,1,2), centers_ms=(-850.,-350.,150.), degree=1),
    'quadratic_equal': dict(nodes=(0,1,2,3), centers_ms=(-850.,-1550/3,-550/3,150.), degree=2),
    'quadratic_sparse': dict(nodes=(0,1,3,4), centers_ms=(-850.,-600.,-100.,150.), degree=2),
}


@lru_cache(maxsize=256)
def primitive_kernel(nodes):
    if any(isinstance(x, (bool, np.bool_)) or not np.isfinite(x) or x != int(x) for x in nodes):
        raise ValueError('Integer time nodes required; rescale rational coordinates first')
    nodes=tuple(int(x) for x in nodes)
    if len(nodes)<2 or len(set(nodes))!=len(nodes): raise ValueError('Distinct nodes required')
    fractions=[]
    for j,t in enumerate(nodes):
        denominator=1
        for k,u in enumerate(nodes):
            if j!=k: denominator*=t-u
        fractions.append(Fraction(1,denominator))
    denominator=reduce(lcm,[f.denominator for f in fractions],1)
    raw=[int(f*denominator) for f in fractions]
    common=reduce(gcd,[abs(x) for x in raw]); integers=[x//common for x in raw]
    if max(abs(x) for x in integers)>np.iinfo(np.int64).max:
        raise ValueError('Primitive kernel exceeds supported integer range')
    w=np.array(integers,dtype=np.int64)
    if w[-1]<0: w=-w
    for degree in range(len(nodes)-1):
        assert sum(int(a)*int(t)**degree for a,t in zip(w,nodes))==0
    w.setflags(write=False)
    return w


def minimum_events(nodes):
    w=primitive_kernel(tuple(nodes)); return int(w[w>0].sum())


def fiber(counts, nodes):
    counts=np.asarray(counts)
    if counts.shape!=(len(nodes),) or np.any(~np.isfinite(counts)) or np.any(counts<0) or np.any(counts!=np.floor(counts)) or np.any(counts>=2**63):
        raise ValueError('Invalid count vector')
    counts=counts.astype(np.int64); w=primitive_kernel(tuple(nodes))
    lower=max(-(int(x)//int(a)) for x,a in zip(counts,w) if a>0)
    upper=min(int(x)//int(-a) for x,a in zip(counts,w) if a<0)
    k=np.arange(lower,upper+1,dtype=np.int64)
    configurations=counts[None,:]+k[:,None]*w[None,:]
    assert np.all(configurations>=0) and lower<=0<=upper
    return configurations


def fiber_distribution(counts,nodes,log_response=0.):
    if not np.isfinite(log_response): raise ValueError('Finite log response required')
    configurations=fiber(counts,nodes)
    log_weight=-gammaln(configurations+1).sum(axis=1)+log_response*configurations[:,-1]
    return configurations,np.exp(log_weight-logsumexp(log_weight))


@lru_cache(maxsize=250000)
def fiber_moments(counts,nodes,response):
    configurations,p=fiber_distribution(counts,nodes,np.log(response))
    z=configurations[:,-1]; low=float(z[0]); width=float(z[-1]-z[0])
    mean=float(p@z)
    variance=float(p@(z-mean)**2)
    return low,width,(mean-low)/width if width else 0.,mean,variance


def design_scores(counts,nodes,response=1.):
    counts=np.asarray(counts)
    if counts.ndim!=2 or counts.shape[1]!=len(nodes): raise ValueError('Invalid count shape')
    if not np.isfinite(response) or response<=0: raise ValueError('Invalid response')
    if np.any(~np.isfinite(counts)) or np.any(counts<0) or np.any(counts!=np.floor(counts)): raise ValueError('Invalid counts')
    unique,inverse=np.unique(counts,axis=0,return_inverse=True)
    v=np.zeros(len(unique)); means=np.zeros(len(unique)); raw=np.zeros(len(unique)); var=np.zeros(len(unique)); active=np.zeros(len(unique),bool)
    for i,row in enumerate(unique):
        low,width,mean,center,variance=fiber_moments(tuple(int(x) for x in row),tuple(nodes),float(response))
        raw[i]=center; var[i]=variance
        if width:
            v[i]=(row[-1]-low)/width; means[i]=mean; active[i]=True
    return dict(observed=v[inverse],mean=means[inverse],raw_mean=raw[inverse],variance=var[inverse],informative=active[inverse])


def design_log_likelihood_ratios(counts,nodes,alternatives,null_response=1.):
    """Conditional likelihood ratios for fixed alternatives, one row per trial."""
    counts=np.asarray(counts); alternatives=np.asarray(alternatives,float)
    if counts.ndim!=2 or counts.shape[1]!=len(nodes) or alternatives.ndim!=1 or not len(alternatives):
        raise ValueError('Invalid count or alternatives shape')
    if np.any(~np.isfinite(counts)) or np.any(counts>=2**63) or np.any(counts<0) or np.any(counts!=np.floor(counts)) or np.any(~np.isfinite(alternatives)) or np.any(alternatives<=0) or not np.isfinite(null_response) or null_response<=0:
        raise ValueError('Invalid counts or response multiplier')
    unique,inverse=np.unique(counts,axis=0,return_inverse=True)
    out=np.zeros((len(unique),len(alternatives)))
    for i,row in enumerate(unique):
        configurations=fiber(row,nodes)
        if len(configurations)==1: continue
        z=configurations[:,-1]; base=-gammaln(configurations+1).sum(axis=1)
        log_null_z=logsumexp(base+z*np.log(null_response))
        log_alt_z=logsumexp(base[:,None]+z[:,None]*np.log(alternatives)[None,:],axis=0)
        out[i]=row[-1]*np.log(alternatives/null_response)-log_alt_z+log_null_z
    return out[inverse]


def informative_mask(counts,nodes):
    """A fiber has another point iff at least one adjacent integer move is feasible."""
    x=np.asarray(counts); w=primitive_kernel(tuple(nodes))
    if x.ndim<1 or x.shape[-1]!=len(nodes) or np.any(~np.isfinite(x)) or np.any(x<0) or np.any(x!=np.floor(x)):
        raise ValueError('Invalid count vector')
    return np.all(x>=np.maximum(-w,0),axis=-1)|np.all(x>=np.maximum(w,0),axis=-1)


def poisson_informative_probability(means,nodes):
    """Exact informative-set probability under independent Poisson window counts."""
    means=np.asarray(means,float); w=primitive_kernel(tuple(nodes))
    if means.ndim<1 or means.shape[-1]!=len(nodes) or np.any(~np.isfinite(means)) or np.any(means<0): raise ValueError('Invalid means')
    plus=np.maximum(w,0); minus=np.maximum(-w,0)
    first=np.prod(poisson.sf(plus-1,means),axis=-1)
    second=np.prod(poisson.sf(minus-1,means),axis=-1)
    both=np.prod(poisson.sf(np.abs(w)-1,means),axis=-1)
    return np.clip(first+second-both,0,1)


def thinned_informative_probability(reference_counts,nodes,retention):
    """Exact expected informative indicator after independent event thinning.

    This conditions on an arbitrary fixed reference count vector and requires
    neither independent reference counts nor a Poisson biological process.
    """
    counts=np.asarray(reference_counts); w=primitive_kernel(tuple(nodes)); q=np.asarray(retention)
    if counts.ndim<1 or counts.shape[-1]!=len(nodes) or np.any(~np.isfinite(counts)) or np.any(counts<0) or np.any(counts!=np.floor(counts)):
        raise ValueError('Invalid reference counts')
    if np.any(~np.isfinite(q)) or np.any((q<0)|(q>1)): raise ValueError('Invalid retention')
    plus=np.maximum(w,0); minus=np.maximum(-w,0)
    first=np.prod(binom.sf(plus-1,counts,q),axis=-1)
    second=np.prod(binom.sf(minus-1,counts,q),axis=-1)
    both=np.prod(binom.sf(np.abs(w)-1,counts,q),axis=-1)
    return np.clip(first+second-both,0,1)


def leading_information_constant(means,nodes):
    """Coefficient of q^m in expected conditional Fisher information for log response.

    At the minimum event total the only non-singleton fiber contains w+ and w-.
    The two configurations have probabilities q^m times A and B to leading order.
    Expected conditional variance is A*B/(A+B) times the squared response-count jump.
    """
    means=np.asarray(means,float); w=primitive_kernel(tuple(nodes))
    if means.shape!=w.shape or np.any(~np.isfinite(means)) or np.any(means<=0): raise ValueError('Positive means required')
    plus=np.maximum(w,0); minus=np.maximum(-w,0)
    log_a=float(plus@np.log(means)-gammaln(plus+1).sum())
    log_b=float(minus@np.log(means)-gammaln(minus+1).sum())
    probability=float(np.exp(log_a)+np.exp(log_b))
    information=float(np.exp(log_a+log_b-logsumexp([log_a,log_b]))*w[-1]**2)
    return dict(minimum_events=int(plus.sum()),probability_coefficient=probability,information_coefficient=information)
