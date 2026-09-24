import itertools
import unittest
import numpy as np
from scipy.stats import binom
from conditional_design import fiber_distribution
from conditional_fixed import conditional_equal_tail_test


class ConditionalFixedTest(unittest.TestCase):
    def test_pair_matches_pooled_exact_binomial(self):
        x=np.array([[2,3],[0,1],[4,2],[1,5]]);total=x.sum();observed=x[:,-1].sum()
        expected=min(1,2*min(binom.cdf(observed,total,.5),binom.sf(observed-1,total,.5)))
        self.assertAlmostEqual(conditional_equal_tail_test(x,(0,1))['pvalue'],expected,places=13)

    def test_quadratic_full_joint_enumeration(self):
        nodes=(0,1,3,4);x=np.array([[1,2,2,1],[2,3,2,1],[0,2,0,1]])
        supports=[fiber_distribution(row,nodes) for row in x];observed=x[:,-1].sum();left=0.;right=0.
        for choice in itertools.product(*[range(len(p)) for _,p in supports]):
            value=sum(supports[j][0][k,-1] for j,k in enumerate(choice))
            probability=np.prod([supports[j][1][k] for j,k in enumerate(choice)])
            if value<=observed:left+=probability
            if value>=observed:right+=probability
        self.assertAlmostEqual(conditional_equal_tail_test(x,nodes)['pvalue'],min(1,2*min(left,right)),places=13)

    def test_uninformative_is_one(self):
        result=conditional_equal_tail_test(np.array([[1,0,0,0],[0,0,1,0]]),(0,1,3,4))
        self.assertEqual(result['pvalue'],1);self.assertEqual(result['informative_trials'],0)


if __name__=='__main__':unittest.main()
