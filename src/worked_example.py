"""Deterministic minimal-reference example printed in the CMPB article."""
import json
import numpy as np
from conditional_design import minimum_events,thinned_informative_probability
from replay_core import replay_distribution,replay_alternative

def main():
    rows=[]
    for name,nodes,x,m in [('Equal spacing',(0,1,2,3),[0,3,0,1],4),
                           ('Three-event',(0,1,3,4),[0,2,0,1],3)]:
        counts=np.array([x]);q=.2
        p=float(thinned_informative_probability(counts,nodes,q)[0])
        assert minimum_events(nodes)==m and np.isclose(p,q**m,rtol=1e-12)
        null,reject,n,information=replay_distribution(counts,nodes)
        rows.append(dict(design=name,reference=x,event_floor=m,q=q,
            retained_support_probability=p,informative_trials=n,
            conditional_information=information,
            conditional_null_rejection=float(null@reject),
            conditional_response15_rejection=float(replay_alternative(counts,nodes,1.5)@reject)))
    print(json.dumps(dict(results=rows,interpretation='Different minimal references, not equal-rate populations. One informative trial need not support rejection at alpha=.05.'),indent=2))

if __name__=='__main__':main()
