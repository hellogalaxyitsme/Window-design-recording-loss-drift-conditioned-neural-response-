"""Conditional count contrasts and bounded sequential evidence.

Classical conditional Poisson inference removes nuisance intercept/slope.
Bounded betting follows the established test-supermartingale construction.
These functions do not turn detected events into verified biological spikes.
"""
from functools import lru_cache
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import brentq

BET_FRACTIONS = np.array([.02, .05, .1, .2, .4, .7, .95])


@lru_cache(maxsize=100000)
def triple_support(n, difference):
    """Support for Z | N=X+Y+Z, difference=Z-X, with factorial log weights."""
    if n < 0 or abs(difference) > n or int(n) != n or int(difference) != difference:
        raise ValueError('Invalid integer sufficient statistics')
    low, high = max(0, difference), (n + difference) // 2
    z = np.arange(low, high + 1, dtype=float)
    x, y = z - difference, n - 2*z + difference
    log_base = -gammaln(x+1) - gammaln(y+1) - gammaln(z+1)
    z.setflags(write=False); log_base.setflags(write=False)
    return z, log_base


def triple_distribution(n, difference, log_curvature=0.):
    z, log_base = triple_support(int(n), int(difference))
    w = log_base + z*log_curvature
    p = np.exp(w - logsumexp(w))
    return z, p


@lru_cache(maxsize=200000)
def triple_moments(n, difference, curvature):
    support, p = triple_distribution(n, difference, np.log(curvature))
    center = float(p @ support)
    spread = float(p @ (support-center)**2)
    width = float(support[-1]-support[0])
    normalized = (center-support[0])/width if width else 0.
    return float(support[0]), width, normalized, center, spread


def validate_counts(counts, columns):
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != columns:
        raise ValueError(f'Expected an n by {columns} array')
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError('Counts must be finite nonnegative integers')
    return counts.astype(np.int64)


def triple_scores(counts, curvature=1.):
    """Observed bounded allocation and its boundary-null mean, trial by trial.

    An informative trial has at least two possible allocations given N,D.
    Nuisance conditioning may remove all information even when N > 0.
    """
    counts = validate_counts(counts, 3)
    if not np.isfinite(curvature) or curvature <= 0:
        raise ValueError('Curvature must be finite and positive')
    observed = np.zeros(len(counts)); mean = np.zeros(len(counts))
    variance = np.zeros(len(counts)); raw_mean = np.zeros(len(counts))
    informative = np.zeros(len(counts), dtype=bool)
    for i, (x, y, zobs) in enumerate(counts):
        low, width, expected, center, spread = triple_moments(int(x+y+zobs), int(zobs-x), float(curvature))
        raw_mean[i] = center
        if width > 0:
            observed[i] = (zobs-low) / width
            mean[i] = expected
            variance[i] = spread
            informative[i] = True
    return dict(observed=observed, mean=mean, variance=variance, raw_mean=raw_mean,
                informative=informative)


def pair_scores(counts, ratio=1.):
    counts = validate_counts(counts, 2)
    if not np.isfinite(ratio) or ratio <= 0: raise ValueError('Invalid ratio')
    total = counts.sum(axis=1); keep = total > 0
    value = np.divide(counts[:, 1], total, out=np.zeros(len(total)), where=keep)
    return dict(observed=value, mean=np.full(len(total), ratio/(1+ratio)), informative=keep)


def bounded_evidence(observed, null_mean, informative=None, epsilon=0., direction='increase'):
    """Mixture of fixed-fraction betting wealths, valid under conditional mean bounds.

    For increases assume E[V_t|past,current conditioning statistics] <= m_t.
    Conditional TV distance <= epsilon implies a mean bound m_t+epsilon
    because V_t is in [0,1]. The bound is an ASSUMPTION, not an estimated
    artifact rate. Positive mixtures and nonnegative factors yield a
    test supermartingale, including zero-information trials as factors one.
    """
    v = np.asarray(observed, float); m = np.asarray(null_mean, float)
    if v.shape != m.shape or v.ndim != 1: raise ValueError('Array shape mismatch')
    if not (0 <= epsilon <= 1): raise ValueError('Invalid TV radius')
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(m)) or np.any((v<0)|(v>1)|(m<0)|(m>1)):
        raise ValueError('Scores must be finite and bounded in [0,1]')
    if direction == 'decrease': v, m = 1-v, 1-m
    elif direction != 'increase': raise ValueError('Unknown direction')
    active = np.ones(len(v), bool) if informative is None else np.asarray(informative, bool)
    upper = np.minimum(1., m+epsilon)
    # m=0 represents a degenerate reference; abstain rather than divide by zero.
    active = active & (upper>1e-14) & (upper<1-1e-14)
    lambdas = np.divide(BET_FRACTIONS[None, :], upper[:, None],
                        out=np.zeros((len(v), len(BET_FRACTIONS))), where=upper[:, None]>0)
    factors = 1 + lambdas*(v-upper)[:, None]
    factors[~active, :] = 1.
    if np.any(factors <= 0): raise ArithmeticError('Bet is not strictly positive')
    log_wealth = np.cumsum(np.log(factors), axis=0)
    log_e = logsumexp(log_wealth, axis=1)-np.log(len(BET_FRACTIONS))
    return log_e


def conditional_loglik(counts, log_curvature):
    counts = validate_counts(counts, 3)
    value = 0.
    for x, y, zobs in counts:
        support, base = triple_support(int(x+y+zobs), int(zobs-x))
        value += float(zobs*log_curvature - logsumexp(base+support*log_curvature))
    return value  # data-only factorial constant omitted


def conditional_estimate(counts):
    """Conditional MLE and Wald SE (descriptive; not robust to misspecification)."""
    counts = validate_counts(counts, 3)
    def score(log_k):
        return sum(float(z - p@s) for x,y,z in counts
                   for s,p in [triple_distribution(int(x+y+z), int(z-x), log_k)])
    left, right = score(-20), score(20)
    if abs(left) < 1e-10 and abs(right) < 1e-10:
        return {'curvature': np.nan, 'log_curvature': np.nan, 'se_log': np.nan, 'boundary': 'uninformative'}
    if left < 0 or right > 0:
        return {'curvature': 0. if left < 0 else np.inf,
                'log_curvature': -np.inf if left < 0 else np.inf, 'se_log': np.nan, 'boundary': 'boundary'}
    root = brentq(score, -20, 20)
    info = sum(float(p@(s-p@s)**2) for x,y,z in counts
               for s,p in [triple_distribution(int(x+y+z), int(z-x), root)])
    return {'curvature': float(np.exp(root)), 'log_curvature': float(root),
            'se_log': float(1/np.sqrt(info)) if info>0 else np.nan, 'boundary': 'interior'}


def curvature_identification_interval(observed_curvature, gamma=1., artifact_fraction=0.):
    """Sharp scalar bounds under common per-window artifact-fraction bounds.

    gamma bounds the product of baseline and detector curvature in [1/gamma,gamma].
    Each of three windows has artifact intensity fraction in [0,artifact_fraction].
    This population relation does not itself supply sampling uncertainty.
    """
    if gamma < 1 or not 0 <= artifact_fraction < 1 or observed_curvature < 0:
        raise ValueError('Invalid sensitivity parameter')
    retained = (1-artifact_fraction)**2
    return observed_curvature*retained/gamma, observed_curvature*gamma/retained
