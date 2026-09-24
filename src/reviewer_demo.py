"""Small runnable example: no datasets, credentials, downloads or GPU needed."""
import json
from pathlib import Path
import numpy as np
from conditional_design import minimum_events,thinned_informative_probability
from replay_core import replay_distribution,replay_alternative


def main():
    x=np.array([[1,2,2,1],[0,0,0,0],[2,3,3,2]])
    report={}
    for name,nodes in [('equal',(0,1,2,3)),('three_event',(0,1,3,4))]:
        null,reject,n,information=replay_distribution(x,nodes)
        report[name]=dict(minimum_events=minimum_events(nodes),observed_informative_trials=n,
            expected_informative_at_q02=float(thinned_informative_probability(x,nodes,.2).sum()),
            conditional_information=information,conditional_null_rejection=float(null@reject),
            conditional_response15_rejection=float(replay_alternative(x,nodes,1.5)@reject))
    assert report['equal']['minimum_events']==4 and report['three_event']['minimum_events']==3
    assert all(r['conditional_null_rejection']<=.05+1e-12 for r in report.values())
    out=Path('results/reviewer_demo');out.mkdir(parents=True,exist_ok=True)
    (out/'demo.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__': main()
