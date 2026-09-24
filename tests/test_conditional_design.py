import itertools
import math
import unittest
import numpy as np
from scipy.special import gammaln
from conditional_design import primitive_kernel,minimum_events,fiber,fiber_distribution,design_scores,informative_mask,leading_information_constant,thinned_informative_probability,poisson_informative_probability,design_log_likelihood_ratios
from conditional_response import triple_scores

def compositions(total,parts):
    if parts==1:
        yield (total,); return
    for first in range(total+1):
        for remainder in compositions(total-first,parts-1): yield (first,)+remainder

class DesignTests(unittest.TestCase):
    def test_invalid_inputs_do_not_silently_change_geometry(self):
        for nodes in [(0,.5,2),(0,1,np.nan),(0,1,np.inf),(0,1,1)]:
            with self.assertRaises(ValueError): primitive_kernel(nodes)
        for counts in [[1,np.nan,2],[1,np.inf,2],[1,-1,2],[1,.5,2],[1,2,2**63]]:
            with self.assertRaises(ValueError): fiber(counts,(0,1,2))
        for alternatives in [[np.nan],[np.inf],[0],[],[[1,2]]]:
            with self.assertRaises(ValueError): design_log_likelihood_ratios([[1,2,3]],(0,1,2),alternatives)
        with self.assertRaises(ValueError): thinned_informative_probability([1,2,3],(0,1,2),np.nan)
        with self.assertRaises(ValueError): poisson_informative_probability([1,np.nan,3],(0,1,2))

    def test_equal_spacing_event_floor(self):
        for degree in range(6):
            nodes=tuple(range(degree+2))
            self.assertEqual(minimum_events(nodes),2**degree)
            expected=np.array([(-1)**(degree+1-j)*math.comb(degree+1,j) for j in nodes])
            np.testing.assert_array_equal(primitive_kernel(nodes),expected)

    def test_exhaustive_minimum_and_quadratic_optimum(self):
        for nodes in [(0,1),(0,1,2),(0,1,2,3),(0,1,3,4)]:
            floor=minimum_events(nodes)
            for n in range(floor+1):
                configs=np.array(list(compositions(n,len(nodes))))
                mask=informative_mask(configs,nodes)
                self.assertEqual(bool(mask.any()),n>=floor)
                for x,yes in zip(configs,mask): self.assertEqual(len(fiber(x,nodes))>1,bool(yes))
        # Any two events on distinct nodes are determined by sum and sum of squares.
        for size in range(3,10):
            nodes=np.arange(size)
            signatures={}
            for n in [0,1,2]:
                for x in compositions(n,size):
                    signature=(n,int(nodes@x),int((nodes**2)@x))
                    self.assertNotIn(signature,signatures)
                    signatures[signature]=x
        self.assertEqual(minimum_events((0,1,3,4)),3)

    def test_polynomial_nuisance_cancellation(self):
        for nodes,x in [((0,1,2,3),(4,6,7,5)),((0,1,3,4),(3,5,8,4))]:
            for log_r in [-1,0,1]:
                configurations,p=fiber_distribution(x,nodes,log_r)
                t=np.asarray(nodes)
                for coef in [(-1,.1,.05),(2,-.2,-.1)]:
                    mu=np.exp(coef[0]+coef[1]*t+coef[2]*t*t); mu[-1]*=np.exp(log_r)
                    log=(configurations*np.log(mu)-mu-gammaln(configurations+1)).sum(axis=1)
                    target=np.exp(log-log.max()); target/=target.sum()
                    np.testing.assert_allclose(p,target,rtol=1e-10,atol=1e-12)

    def test_unique_quadratic_three_event_placement(self):
        for last in range(3,33):
            for first,second in itertools.combinations(range(1,last),2):
                expected=(4*first==last and 4*second==3*last)
                self.assertEqual(minimum_events((0,first,second,last))==3,expected)

    def test_matches_specialized_triple(self):
        counts=np.array(list(compositions(9,3)))
        for response in [.5,1,2]:
            a=design_scores(counts,(0,1,2),response); b=triple_scores(counts,response)
            for key in a: np.testing.assert_allclose(a[key],b[key],atol=1e-12)

    def test_sparse_information_constants(self):
        a=leading_information_constant([1,1,1],(0,1,2))
        self.assertAlmostEqual(a['information_coefficient'],1/3)
        b=leading_information_constant([1,1,1,1],(0,1,2,3))
        c=leading_information_constant([1,1,1,1],(0,1,3,4))
        self.assertEqual(b['minimum_events'],4); self.assertEqual(c['minimum_events'],3)
        self.assertAlmostEqual(b['information_coefficient'],1/12)
        self.assertAlmostEqual(c['information_coefficient'],1/4)

    def test_exact_thinning_against_full_enumeration(self):
        from scipy.stats import binom
        for nodes,ref in [((0,1,2),(2,3,2)),((0,1,2,3),(1,3,3,1)),((0,1,3,4),(1,2,2,1))]:
            for q in [.1,.5,1.]:
                total=0.
                for x in itertools.product(*(range(v+1) for v in ref)):
                    if informative_mask(np.array(x),nodes): total+=float(np.prod(binom.pmf(x,ref,q)))
                self.assertAlmostEqual(float(thinned_informative_probability(ref,nodes,q)),total,places=12)

    def test_exact_probability_sparse_limit(self):
        for nodes in [(0,1),(0,1,2),(0,1,2,3),(0,1,3,4)]:
            coeff=leading_information_constant(np.ones(len(nodes)),nodes)
            q=1e-5
            exact=poisson_informative_probability(np.full(len(nodes),q),nodes)
            leading=coeff['probability_coefficient']*q**coeff['minimum_events']
            self.assertAlmostEqual(float(exact/leading),1,places=4)

    def test_likelihood_ratio_reference_expectation(self):
        for nodes,x in [((0,1,2),(5,3,4)),((0,1,2,3),(3,4,5,6)),((0,1,3,4),(5,4,3,2))]:
            configurations,p=fiber_distribution(x,nodes)
            lr=design_log_likelihood_ratios(configurations,nodes,[.25,.5,1,2,4])
            np.testing.assert_allclose(p@np.exp(lr),np.ones(5),atol=1e-12)

if __name__=='__main__': unittest.main()
