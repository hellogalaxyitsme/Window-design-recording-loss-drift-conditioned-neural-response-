"""Model-conditional response evidence and pre-stimulation diagnostics.

Nominal control flags are diagnostics, not measured biological false positives.
All units remain in denominators. Subject summaries give sessions equal weight.
"""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binom,ttest_rel
from conditional_design import DESIGNS,design_scores,design_log_likelihood_ratios
from response_kernels import ALTERNATIVES
from response_kernels import wealth_matrix

ROOT=Path('results/paper_v1')
OUT=ROOT/'design_response'


def evaluate(x,nodes,identity,unit_ids=None):
    x=np.asarray(x)
    if x.ndim==2: x=x[None,:,:]
    units,trials,width=x.shape; flat=x.reshape(-1,width)
    scores=design_scores(flat,nodes)
    bounded=np.logaddexp(wealth_matrix(scores,units,trials,direction='increase'),
                         wealth_matrix(scores,units,trials,direction='decrease'))-np.log(2)
    lr=design_log_likelihood_ratios(flat,nodes,ALTERNATIVES).reshape(units,trials,-1)
    likelihood=logsumexp(np.cumsum(lr,axis=1),axis=2)-np.log(len(ALTERNATIVES))
    pre=x[:,:,-2]; post=x[:,:,-1]; total=(pre+post).sum(axis=1); sy=post.sum(axis=1)
    bp=np.minimum(1,2*np.minimum(binom.cdf(sy,total,.5),binom.sf(sy-1,total,.5)))
    tp=ttest_rel(post,pre,axis=1).pvalue
    ids=list(range(units)) if unit_ids is None else list(unit_ids)
    rows=[]
    informative=scores['informative'].reshape(units,trials)
    variance=scores['variance'].reshape(units,trials)
    expected=scores['raw_mean'].reshape(units,trials)
    for i in range(units):
        row=dict(identity,unit_id=ids[i],trials=trials,informative_trials=int(informative[i].sum()),
                 conditional_null_information=float(variance[i].sum()),
                 conditional_score=float((post[i]-expected[i]).sum()),
                 recorded_events=int(x[i].sum()),post_events=int(sy[i]),
                 bounded_loge=float(bounded[i,-1]),bounded_max_loge=float(max(0,bounded[i].max())),
                 conditional_lr_loge=float(likelihood[i,-1]),conditional_lr_max_loge=float(max(0,likelihood[i].max())),
                 pooled_pair_binomial_p=float(bp[i]),paired_t_p=float(tp[i]))
        for method in ['bounded','conditional_lr']:
            row[method+'_nominal_fixed_flag']=row[method+'_loge']>=np.log(20)
            row[method+'_nominal_anytime_flag']=row[method+'_max_loge']>=np.log(20)
            # Bonferroni over all selected units within this session/design/shift.
            # This is NOT a correction for choosing designs/shifts after inspection.
            row[method+'_session_bonferroni_flag']=row[method+'_loge']>=np.log(20*units)
        rows.append(row)
    return rows


def finalspark():
    rows=[]
    for dataset in ['fs369','fs437']:
        folder=ROOT/'finalspark_design_counts'/dataset
        trials=pd.read_parquet(folder/'trials.parquet')
        dates=pd.to_datetime(trials.anchor_ns,unit='ns',utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        for target in ['all','cap300','min30_cap300','broad_shift0']:
            data=np.load(folder/(target+'_counts.npz'))
            for name,spec in DESIGNS.items():
                for shift in ([0] if target=='broad_shift0' else [0,-300,-600,-900]):
                    x=data[f'{name}_shift{shift}']
                    for date in ['all',*np.unique(dates)]:
                        selected=x if date=='all' else x[dates==date]
                        rows.extend(evaluate(selected,spec['nodes'],dict(dataset=dataset,target=target,
                            design=name,shift_ms=shift,date=date,level='export_population')))
            print('COMPLETE',dataset,target,'response diagnostics',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'finalspark_response.csv',index=False)


def external(meta):
    folder=ROOT/'external_counts'/meta['session']; data=np.load(folder/'counts.npz')
    unit=pd.read_parquet(folder/'units.parquet'); rows=[]
    for name,spec in DESIGNS.items():
        for shift in [0,-300,-600,-900]:
            identity=dict(dataset='dandi_000774',subject=meta['subject'],session=meta['session'],
                split=meta['split'],design=name,shift_ms=shift,target='source_good_units')
            rows.extend(evaluate(data[f'{name}_shift{shift}'],spec['nodes'],identity,unit.unit_id))
    table=pd.DataFrame(rows); table.to_parquet(OUT/(meta['session']+'_response.parquet'),index=False)
    print('COMPLETE',meta['session'],'response diagnostics',flush=True)
    return table


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--public-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.public_only: finalspark()
    manifest=json.loads((ROOT/'external_counts/manifest.json').read_text())
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool: frames=list(pool.map(external,manifest))
    table=pd.concat(frames,ignore_index=True);table.to_parquet(OUT/'external_unit_response.parquet',index=False)
    values=['informative_trials','conditional_null_information','bounded_nominal_fixed_flag',
            'bounded_nominal_anytime_flag','bounded_session_bonferroni_flag',
            'conditional_lr_nominal_fixed_flag','conditional_lr_nominal_anytime_flag',
            'conditional_lr_session_bonferroni_flag']
    session=table.groupby(['subject','session','split','design','shift_ms'])[values].mean().reset_index()
    session.to_csv(OUT/'external_session_response.csv',index=False)
    subject=session.groupby(['subject','split','design','shift_ms'])[values].mean().reset_index()
    subject.to_csv(OUT/'external_subject_response.csv',index=False)
    (OUT/'interpretation.json').write_text(json.dumps(dict(
        primary_comparison='Information retained by frozen designs; response evidence is assumption-sensitive supporting analysis.',
        evalue_null='Independent Poisson window counts conditional on arbitrary trial-specific polynomial log-mean nuisance and history.',
        threshold='Nominal E>=20; within-session unit-family Bonferroni also shown. No across-design or across-control confirmatory claim.',
        controls='Pre-stimulation shifts diagnose temporal structure/model mismatch. They are not validated null-response labels.',
        retrospective='Anytime maxima are retrospective diagnostics of a fixed rule; this was not a prospective sequential trial.',
        alternatives=ALTERNATIVES.tolist()),indent=2))
