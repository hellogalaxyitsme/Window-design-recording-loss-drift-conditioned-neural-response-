"""Exact monotone-system reliability for informative conditional count sets."""
import itertools
import math
import numpy as np
from conditional_design import primitive_kernel,informative_mask


def state_space(nodes):
    cap=np.abs(primitive_kernel(tuple(nodes)))
    if math.prod(int(c)+1 for c in cap)>100000:
        raise ValueError('Grouped-loss state space exceeds 100000 states; use a smaller kernel')
    states=np.array(list(itertools.product(*[range(int(c)+1) for c in cap])),dtype=np.int64)
    strides=np.array([np.prod(cap[j+1:]+1,dtype=np.int64) for j in range(len(cap))])
    np.testing.assert_array_equal(states@strides,np.arange(len(states)))
    return cap,states,strides,informative_mask(states,nodes)


def group_informative_probability(counts,nodes,retention):
    """Counts shape (trials, groups, windows); independent group retention."""
    x=np.asarray(counts)
    if x.ndim!=3 or x.shape[-1]!=len(nodes) or np.any(~np.isfinite(x)) or np.any(x<0) or np.any(x!=np.floor(x)):
        raise ValueError('Invalid grouped counts')
    if not 0<=retention<=1:raise ValueError('Invalid retention')
    cap,states,strides,informative=state_space(nodes);batch=len(x);q=float(retention)
    if q==0:return np.zeros(batch)
    if q==1:return informative_mask(x.sum(axis=1),nodes).astype(float)
    probability=np.zeros((batch,len(states)));probability[:,0]=1.
    rows=np.arange(batch)[:,None]
    for g in range(x.shape[1]):
        destination=np.minimum(states[None,:,:]+x[:,g,None,:],cap)@strides
        updated=(1-q)*probability
        np.add.at(updated,(rows,destination),q*probability)
        probability=updated
    np.testing.assert_allclose(probability.sum(axis=1),1.,atol=1e-12)
    return probability[:,informative].sum(axis=1)


def minimum_retained_groups(counts,nodes):
    """Minimum group witness size; -1 means the reference trial is uninformative."""
    x=np.asarray(counts)
    if x.ndim!=3 or x.shape[-1]!=len(nodes) or np.any(~np.isfinite(x)) or np.any(x<0) or np.any(x!=np.floor(x)):
        raise ValueError('Invalid grouped counts')
    cap,states,strides,informative=state_space(nodes);batch,groups,_=x.shape
    cost=np.full((batch,len(states)),groups+1,dtype=np.int32);cost[:,0]=0
    rows=np.arange(batch)[:,None]
    for g in range(groups):
        destination=np.minimum(states[None,:,:]+x[:,g,None,:],cap)@strides
        updated=cost.copy();np.minimum.at(updated,(rows,destination),cost+1);cost=updated
    best=cost[:,informative].min(axis=1);best[best>groups]=-1
    return best
