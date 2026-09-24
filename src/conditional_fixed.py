"""Classical fixed-sample conditional convolution test for independent trials.

Unlike the sequential process, conditioning on the entire sequence here assumes
independent trials (or a separately justified joint conditional product law).
"""
from functools import lru_cache
import numpy as np
from conditional_design import fiber_distribution,primitive_kernel


@lru_cache(maxsize=250000)
def support_pmf(counts,nodes):
    configurations,p=fiber_distribution(counts,nodes)
    low=int(configurations[0,-1]);step=int(primitive_kernel(nodes)[-1])
    p.setflags(write=False)
    return low,step,p


def conditional_equal_tail_test(counts,nodes):
    """Finite-support convolution, equal-tail two-sided conditional p-value.

    Uses floating-point polynomial convolution. Intended for conventional alpha
    levels, not claims about astronomically small p-values.
    """
    x=np.asarray(counts);nodes=tuple(nodes)
    if x.ndim!=2 or x.shape[1]!=len(nodes) or np.any(x<0) or np.any(x!=np.floor(x)):
        raise ValueError('Invalid counts')
    pmf=np.array([1.]);observed=0;informative=0
    for row in x:
        low,step,p=support_pmf(tuple(map(int,row)),nodes)
        observed+=(int(row[-1])-low)//step
        if len(p)>1:
            pmf=np.convolve(pmf,p);informative+=1
    total=float(pmf.sum())
    if not np.isclose(total,1.,rtol=1e-9,atol=1e-12):raise ArithmeticError('Conditional convolution lost normalization')
    pmf/=total
    pvalue=min(1.,2*min(float(pmf[:observed+1].sum()),float(pmf[observed:].sum())))
    return dict(pvalue=pvalue,informative_trials=informative,support_size=len(pmf))
