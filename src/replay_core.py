"""Exact replay and matched-count reference utilities for the dated extension."""
import numpy as np
from scipy.special import logsumexp
from conditional_fixed import support_pmf
from conditional_design import DESIGNS, informative_mask

NAMES = ('quadratic_equal', 'quadratic_sparse')


def wilson(k, n):
    p = k / n; z = 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(center-half), float(center+half)


def replay_distribution(x, nodes):
    """Null law of offset/step-scaled total; also sum Var(X_last | S)."""
    pmf = np.ones(1); information = 0.; informative = 0
    for row in x:
        _, step, p = support_pmf(tuple(map(int, row)), tuple(nodes))
        if len(p) > 1:
            k = np.arange(len(p))*step
            information += float(p @ (k-p@k)**2)
            pmf = np.convolve(pmf, p); informative += 1
    if not np.isclose(pmf.sum(), 1, atol=1e-10, rtol=1e-8):
        raise ArithmeticError('Replay convolution normalization failure')
    pmf /= pmf.sum()
    pvalues = np.minimum(1., 2*np.minimum(np.cumsum(pmf), np.cumsum(pmf[::-1])[::-1]))
    return pmf, pvalues < .05, informative, information


def tilted_pmf(pmf, response, step=1):
    if response <= 0: raise ValueError('Positive response required')
    logp = np.full(len(pmf), -np.inf)
    positive = pmf > 0
    logp[positive] = np.log(pmf[positive])
    logp += np.log(response)*np.arange(len(pmf))*step
    return np.exp(logp-logsumexp(logp))


def replay_alternative(x, nodes, response):
    # Tilt each trial before convolution to avoid amplifying underflow in a
    # long null convolution's extreme tails. Equivalent exponential family.
    result=np.ones(1)
    for row in x:
        _,step,p=support_pmf(tuple(map(int,row)),tuple(nodes))
        if len(p)>1: result=np.convolve(result,tilted_pmf(p,response,step))
    if not np.isclose(result.sum(),1.,atol=1e-10,rtol=1e-8):
        raise ArithmeticError('Alternative convolution normalization failure')
    return result/result.sum()


def count_times(times, anchors):
    times = np.sort(np.asarray(times))
    return {name: np.stack([
        np.searchsorted(times, anchors+(center+50)/1000, side='left')-
        np.searchsorted(times, anchors+(center-50)/1000, side='left')
        for center in DESIGNS[name]['centers_ms']], axis=1) for name in NAMES}


def membership(times, anchors):
    """Joint window IDs, -1 outside; windows within each design are disjoint."""
    times = np.asarray(times); codes = np.full((len(times), 2), -1, dtype=int)
    for d, name in enumerate(NAMES):
        for j, center in enumerate(DESIGNS[name]['centers_ms']):
            left = anchors+(center-50)/1000; right = anchors+(center+50)/1000
            trial = np.searchsorted(left, times, side='right')-1
            valid = (trial >= 0) & (times < right[np.maximum(trial, 0)])
            if np.any(codes[valid, d] >= 0): raise ValueError('Overlapping windows within design')
            codes[valid, d] = trial[valid]*4+j
    return codes


def count_memberships(codes, trials):
    return np.stack([np.bincount(codes[codes[:, d]>=0, d], minlength=trials*4).reshape(trials,4)
                     for d in range(2)])


def matched_count_draws(codes, trials, k, repeats, rng):
    categories, sizes = np.unique(codes, axis=0, return_counts=True)
    if not 0 <= k <= len(codes): raise ValueError('Invalid retained total')
    sampled = rng.multivariate_hypergeometric(sizes, k, size=repeats, method='marginals')
    if not np.all(sampled.sum(axis=1) == k): raise AssertionError('Retention count changed')
    counts = np.zeros((2, repeats, trials*4), dtype=np.int64)
    for c, code in enumerate(categories):
        for d in range(2):
            if code[d] >= 0: counts[d, :, code[d]] += sampled[:, c]
    values = np.stack([informative_mask(counts[d].reshape(repeats,trials,4), DESIGNS[name]['nodes']).sum(axis=1)
                       for d, name in enumerate(NAMES)], axis=1)
    return np.c_[values, values[:,1]-values[:,0]]


def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); result=np.empty(len(p))
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result
