import itertools
import unittest
import numpy as np
from conditional_design import informative_mask,thinned_informative_probability
from group_dropout import group_informative_probability,minimum_retained_groups,state_space


class GroupDropoutTest(unittest.TestCase):
    def test_invalid_input_and_excessive_state_space(self):
        with self.assertRaises(ValueError): state_space(tuple(range(9)))
        for value in [np.nan,np.inf,-1,.5]:
            with self.assertRaises(ValueError): group_informative_probability(np.full((1,1,4),value),(0,1,3,4),.2)

    def test_exhaustive_masks(self):
        rng=np.random.default_rng(555)
        for nodes in [(0,1),(0,1,2),(0,1,2,3),(0,1,3,4)]:
            x=rng.poisson(.7,size=(7,5,len(nodes)));q=.23;exact=np.zeros(7);minimum=np.full(7,6)
            for mask in itertools.product([0,1],repeat=5):
                n=sum(mask);active=informative_mask(np.einsum('bgj,g->bj',x,mask),nodes)
                exact+=active*q**n*(1-q)**(5-n);minimum[active]=np.minimum(minimum[active],n)
            minimum[minimum==6]=-1
            np.testing.assert_allclose(group_informative_probability(x,nodes,q),exact,atol=1e-13)
            np.testing.assert_array_equal(minimum_retained_groups(x,nodes),minimum)

    def test_single_event_groups_reduce_to_thinning(self):
        nodes=(0,1,3,4);x=np.array([2,3,2,1]);groups=np.repeat(np.eye(4,dtype=int),x,axis=0)
        for q in [0,.1,.5,1]:
            np.testing.assert_allclose(group_informative_probability(groups[None,:,:],nodes,q),
                thinned_informative_probability(x,nodes,q),atol=1e-13)
        self.assertEqual(minimum_retained_groups(groups[None,:,:],nodes)[0],3)

    def test_one_group_reduces_to_trial_dropout(self):
        nodes=(0,1,2,3);x=np.array([[[1,3,3,1]],[[0,0,0,0]]])
        np.testing.assert_allclose(group_informative_probability(x,nodes,.3),[.3,0])
        np.testing.assert_array_equal(minimum_retained_groups(x,nodes),[1,-1])


if __name__=='__main__':unittest.main()
