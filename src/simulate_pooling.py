"""Classical pooled GLM versus per-trial nuisance conditioning."""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from conditional_design import DESIGNS,primitive_kernel,design_log_likelihood_ratios,informative_mask
from response_kernels import ALTERNATIVES
from conditional_fixed import conditional_equal_tail_test


def pooled_contrast(x,nodes):
    """Saturated pooled log-mean contrast with Poisson and cluster variances."""
    x=np.asarray(x,float); repeats,trials,width=x.shape; mean=x.mean(axis=1)
    w=primitive_kernel(tuple(nodes)).astype(float);w/=w[-1]
    valid=np.all(mean>0,axis=1); safe=np.where(mean>0,mean,1.)
    estimate=np.log(safe)@w;gradient=w[None,:]/safe
    # Empirical influence per trial yields the trial-cluster sandwich variance.
    influence=np.einsum('rtj,rj->rt',x-mean[:,None,:],gradient)
    robust_variance=np.var(influence,axis=1,ddof=1)/trials
    poisson_variance=np.sum(w[None,:]**2/safe,axis=1)/trials
    robust_p=2*norm.sf(np.divide(abs(estimate),np.sqrt(robust_variance),out=np.zeros(repeats),where=robust_variance>0))
    poisson_p=2*norm.sf(abs(estimate)/np.sqrt(poisson_variance))
    robust_p[~valid]=np.nan;poisson_p[~valid]=np.nan;estimate[~valid]=np.nan
    return estimate,robust_p,poisson_p,valid


def experiment(job):
    index,mode,q,response=job;repeats=2000;trials=1000;rng=np.random.default_rng(202609075+index)
    # Same latent trial gains and nuisance states for the two designs.
    gain=rng.lognormal(-.125,.5,size=(repeats,trials,1))
    z=rng.choice([-1.,1.],size=(repeats,trials,1));rows=[]
    for name in ['quadratic_equal','quadratic_sparse']:
        spec=DESIGNS[name];t=(np.asarray(spec['centers_ms'])+350)/500
        logbase=.1*t+.3*t*t
        mu=3*q*gain*np.exp(logbase)[None,None,:]
        if mode=='varying_quadratic':mu*=np.exp(z*(.3*t+.5*t*t)[None,None,:])
        if mode=='window_bursts':mu*=rng.gamma(.25,4,size=mu.shape)
        mu[:,:,-1]*=response;x=rng.poisson(mu)
        estimate,rp,pp,valid=pooled_contrast(x,spec['nodes'])
        lr=design_log_likelihood_ratios(x.reshape(-1,4),spec['nodes'],ALTERNATIVES).reshape(repeats,trials,-1)
        evidence=logsumexp(np.cumsum(lr,axis=1),axis=2)-np.log(len(ALTERNATIVES))
        fixed=np.array([conditional_equal_tail_test(row,spec['nodes'])['pvalue'] for row in x])
        methods=dict(pooled_glm_cluster=rp<.05,pooled_glm_poisson=pp<.05,
            conditional_lr_fixed=evidence[:,-1]>=np.log(20),conditional_lr_anytime=evidence.max(axis=1)>=np.log(20),
            conditional_exact_fixed=fixed<.05)
        for method,reject in methods.items():
            rate=float(reject.mean()); rows.append(dict(mode=mode,retention=q,true_response=response,design=name,
                method=method,repeats=repeats,trials=trials,rejections=int(reject.sum()),rejection_rate=rate,
                monte_carlo_se=float(np.sqrt(rate*(1-rate)/repeats)),undefined_pooled_repeats=int((~valid).sum()),
                mean_pooled_log_response=float(np.nanmean(estimate)),true_log_response=float(np.log(response)),
                mean_informative_trials=float(informative_mask(x,spec['nodes']).sum(axis=1).mean())))
    print('COMPLETE',mode,q,response,flush=True);return rows


if __name__=='__main__':
    out=Path('results/paper_v1/pooling_simulation_v2');out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for mode in ['shared_quadratic','varying_quadratic','window_bursts']:
        for q in [.1,.2,.5,1]:
            for response in [1,1.5]:jobs.append((len(jobs),mode,q,response))
    (out/'plan.json').write_text(json.dumps(dict(seed=202609075,repeats=2000,trials=1000,jobs=jobs),indent=2))
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for result in pool.map(experiment,jobs):
            rows.extend(result);pd.DataFrame(rows).to_csv(out/'calibration_power.csv',index=False)
    (out/'completion.json').write_text(json.dumps(dict(rows=len(rows))))
