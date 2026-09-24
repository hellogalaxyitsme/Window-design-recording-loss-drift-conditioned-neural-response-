import itertools
import unittest
import numpy as np
from conditional_design import fiber_distribution,DESIGNS,informative_mask
from conditional_fixed import conditional_equal_tail_test
from replay_core import (replay_distribution,replay_alternative,tilted_pmf,membership,
    count_memberships,matched_count_draws,count_times,holm,NAMES)
from robustness import trajectory


class ExtensionTests(unittest.TestCase):
    def test_replay_against_joint_enumeration(self):
        x=np.array([[1,2,2,1],[2,3,2,2]])
        nodes=(0,1,3,4); null,reject,n,info=replay_distribution(x,nodes)
        laws=[fiber_distribution(row,nodes,np.log(1.5)) for row in x]
        expected=np.zeros(len(null)); low=sum(a[:,-1].min() for a,p in laws)
        for i,j in itertools.product(range(len(laws[0][1])),range(len(laws[1][1]))):
            total=int(laws[0][0][i,-1]+laws[1][0][j,-1]-low)
            expected[total]+=laws[0][1][i]*laws[1][1][j]
        np.testing.assert_allclose(replay_alternative(x,nodes,1.5),expected,atol=1e-14)
        np.testing.assert_allclose(tilted_pmf(null,1.5),expected,atol=1e-14)
        self.assertGreater(info,0);self.assertEqual(n,2)
        # Null rejection sum is controlled; direct exact p-values agree per allocation.
        self.assertLessEqual(null@reject,.05+1e-12)
        for a,p in laws:
            self.assertAlmostEqual(p.sum(),1)

    def test_zero_support(self):
        x=np.zeros((10,4),int);p,reject,n,info=replay_distribution(x,(0,1,3,4))
        self.assertEqual(n,0);self.assertEqual(info,0);self.assertFalse(reject.any())
        np.testing.assert_array_equal(replay_alternative(x,(0,1,3,4),2),[1])

    def test_shared_membership_and_endpoints(self):
        anchors=np.array([2.,4.]); times=np.array([0.,1.1,1.15,1.2,1.4,1.9,2.1,2.15,2.2,4.15])
        codes=membership(times,anchors); counts=count_memberships(codes,2); direct=count_times(times,anchors)
        for d,name in enumerate(NAMES): np.testing.assert_array_equal(counts[d],direct[name])
        sim=matched_count_draws(codes,2,len(times),5,np.random.default_rng(1))
        expected=[informative_mask(counts[d],DESIGNS[name]['nodes']).sum() for d,name in enumerate(NAMES)]
        np.testing.assert_array_equal(sim,np.tile([*expected,expected[1]-expected[0]],(5,1)))

    def test_hypergeometric_against_all_subsets(self):
        codes=np.array([[0,0],[1,1],[1,1],[2,2],[2,2],[3,3],[-1,-1]])
        vals=[]
        for subset in itertools.combinations(range(7),5):
            c=count_memberships(codes[list(subset)],1)
            v=[int(informative_mask(c[d],DESIGNS[name]['nodes']).sum()) for d,name in enumerate(NAMES)]
            vals.append([*v,v[1]-v[0]])
        sims=matched_count_draws(codes,1,5,50000,np.random.default_rng(7))
        np.testing.assert_allclose(sims.mean(axis=0),np.mean(vals,axis=0),atol=.01)
        np.testing.assert_array_equal(matched_count_draws(codes,1,0,3,np.random.default_rng(1)),np.zeros((3,3)))

    def test_integrated_means_and_holm(self):
        t=np.array([-1.,-.5,.5,1.]);z=np.array([-1.,1.])[:,None]
        np.testing.assert_allclose(trajectory(t,z,0,True,32),trajectory(t,z,0,True,64),rtol=1e-13)
        self.assertGreater(np.max(abs(trajectory(t,z,0,True)-trajectory(t,z,0,False))),0)
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])


if __name__=='__main__': unittest.main()
