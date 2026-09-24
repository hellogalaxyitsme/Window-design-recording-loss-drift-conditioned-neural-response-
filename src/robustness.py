"""Graded departures from the conditional count model."""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from conditional_design import DESIGNS, design_log_likelihood_ratios
from conditional_fixed import conditional_equal_tail_test
from response_kernels import ALTERNATIVES
from simulate_pooling import pooled_contrast
from replay_core import NAMES, wilson

OUT=Path('results/peer_extension/e14')
SETTINGS=[('shared_exact',False,0.,0.,False),('varying_exact',True,0.,0.,False)]
SETTINGS += [(f'gain_cv{cv}',False,cv,0.,False) for cv in [.1,.25,.5]]
SETTINGS += [(f'cubic_{c}',False,0.,c,False) for c in [-.1,-.05,.05,.1]]
SETTINGS += [('integrated_shared',False,0.,0.,True),('integrated_varying',True,0.,0.,True),
             ('varying_gain_cv0.25',True,.25,0.,False)]


def trajectory(t, z, cubic, integrated, points=32):
    def value(s): return np.exp(.1*s+.3*s*s+z*(.3*s+.5*s*s)+cubic*s**3)
    if not integrated: return value(t)
    nodes, weights=np.polynomial.legendre.leggauss(points)
    result=np.zeros(np.broadcast_shapes(np.shape(t),np.shape(z)))
    for n,w in zip(nodes,weights): result += w/2*value(t+.1*n)
    return result


def experiment(job):
    index, setting, q, response=job
    name,varying,cv,cubic,integrated=setting; repeats=2000; trials=1000
    rng=np.random.default_rng(202609081+index)
    gain=rng.lognormal(-.125,.5,size=(repeats,trials,1))
    z=rng.choice([-1.,1.],size=(repeats,trials,1)) if varying else 0.
    rows=[]
    for design in NAMES:
        spec=DESIGNS[design]; t=(np.asarray(spec['centers_ms'])+350)/500
        mu=3*q*gain*trajectory(t,z,cubic,integrated)
        if integrated:
            check=trajectory(t,np.array([-1.,1.])[:,None] if varying else 0.,cubic,True,64)
            np.testing.assert_allclose(trajectory(t,np.array([-1.,1.])[:,None] if varying else 0.,cubic,True),check,rtol=1e-12)
        if cv: mu*=rng.gamma(1/cv**2,cv**2,size=mu.shape)
        mu[:,:,-1]*=response; x=rng.poisson(mu)
        _,p,_,valid=pooled_contrast(x,spec['nodes'])
        lr=design_log_likelihood_ratios(x.reshape(-1,4),spec['nodes'],ALTERNATIVES).reshape(repeats,trials,-1)
        evidence=logsumexp(np.cumsum(lr,axis=1),axis=2)-np.log(len(ALTERNATIVES))
        exact=np.array([conditional_equal_tail_test(row,spec['nodes'])['pvalue'] for row in x])
        methods={'conditional_exact_fixed':exact<.05,'conditional_lr_fixed':evidence[:,-1]>=np.log(20),
                 'conditional_lr_anytime':evidence.max(axis=1)>=np.log(20),'pooled_glm_cluster':p<.05}
        for method,reject in methods.items():
            k=int(reject.sum()); low,high=wilson(k,repeats)
            rows.append(dict(setting=name,retention=q,response=response,design=design,method=method,
                repetitions=repeats,trials=trials,rejections=k,rejection_rate=k/repeats,mc_lower95=low,mc_upper95=high,
                undefined_pooled=int((~valid).sum()),seed=202609081+index))
    print('COMPLETE',name,q,response,flush=True); return rows


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(i,s,q,r) for i,(s,q,r) in enumerate((s,q,r) for s in SETTINGS for q in [.1,.2,.5] for r in [1.,1.5])]
    (OUT/'plan.json').write_text(json.dumps(dict(jobs=jobs,protocol='fixed graded-departure settings encoded in this program'),indent=2))
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for result in pool.map(experiment,jobs):
            rows.extend(result); pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
    assert len(rows)==576
    (OUT/'completion.json').write_text(json.dumps(dict(rows=len(rows),jobs=len(jobs))))
