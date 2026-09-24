"""Numerical checks against enumerated generative probabilities and guarantees."""
import unittest
import numpy as np
from scipy.special import gammaln
from conditional_response import (triple_distribution, triple_scores, pair_scores,
    bounded_evidence, conditional_estimate, curvature_identification_interval)

class ConditionalTests(unittest.TestCase):
    def test_nuisance_cancellation(self):
        for n,d in [(0,0),(1,-1),(2,0),(9,3),(30,-6)]:
            for k in [.25,1,4]:
                z,p = triple_distribution(n,d,np.log(k)); x=z-d; y=n-2*z+d
                for a,b in [(.1,-1),(3,0),(100,1.2)]:
                    mu=np.array([a,a*np.exp(b),a*np.exp(2*b)*k])
                    counts=np.stack([x,y,z],axis=1)
                    log=(counts*np.log(mu)-mu-gammaln(counts+1)).sum(axis=1)
                    expected=np.exp(log-log.max()); expected/=expected.sum()
                    np.testing.assert_allclose(p,expected,rtol=1e-10,atol=1e-12)

    def test_no_information_floor(self):
        scores=triple_scores([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        self.assertFalse(scores['informative'].any())
        e=bounded_evidence(scores['observed'],scores['mean'],scores['informative'])
        np.testing.assert_allclose(e,0,atol=1e-15)
        self.assertTrue(triple_scores([[0,2,0],[1,0,1]])['informative'].all())

    def test_composite_null_e_expectation(self):
        for n,d in [(2,0),(9,1),(50,-10)]:
            z,_=triple_distribution(n,d)
            counts=np.stack([z-d,n-2*z+d,z],axis=1)
            scores=triple_scores(counts,curvature=2.)
            for k in [.2,1,2]:
                _,p=triple_distribution(n,d,np.log(k))
                # Each support point is a separate ONE-observation experiment.
                e=np.array([np.exp(bounded_evidence([v],[m],epsilon=.05)[0])
                            for v,m in zip(scores['observed'],scores['mean'])])
                self.assertLessEqual(p@e,1+1e-12)
                # Worst Huber contamination at the largest score is inside TV .05.
                self.assertLessEqual(.95*(p@e)+.05*e.max(),1+1e-12)

    def test_information_not_repeated_per_spike(self):
        s=pair_scores([[0,10000]])
        e=bounded_evidence(s['observed'],s['mean'])
        self.assertLess(np.exp(e[-1]),2.)

    def test_sequential_mixture_matches_enumeration(self):
        import itertools
        for direction in ['increase','decrease']:
            expectation=0.
            for seq in itertools.product([0.,1.],repeat=5):
                e=bounded_evidence(seq,np.full(5,.3),direction=direction)
                probability=np.prod([.3 if x else .7 for x in seq])
                expectation+=probability*np.exp(e[-1])
            self.assertAlmostEqual(expectation,1.,places=12)

    def test_bounds_attained(self):
        for gamma in [1,2,5]:
            for eta in [0,.1,.5]:
                low,high=curvature_identification_interval(3,gamma,eta)
                self.assertAlmostEqual(low,3*(1-eta)*(1-eta)/gamma)
                self.assertAlmostEqual(high,3/((1-eta)*(1-eta))*gamma)

    def test_estimation_and_invalid_input(self):
        self.assertEqual(conditional_estimate([[0,0,0]])['boundary'],'uninformative')
        with self.assertRaises(ValueError): triple_scores([[1,-1,2]])
        with self.assertRaises(ValueError): pair_scores([[1,.5]])
        with self.assertRaises(ValueError): bounded_evidence([2],[.5])

if __name__=='__main__': unittest.main()
