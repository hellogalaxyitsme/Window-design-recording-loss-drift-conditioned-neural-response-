"""Exact channel-dropout expectations at matched expected event retention."""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_design import DESIGNS,informative_mask,thinned_informative_probability
from group_dropout import group_informative_probability,minimum_retained_groups
from analyze_design_retention import QS

BASE=Path('results/paper_v1');OUT=BASE/'group_dropout'


def analyze(job):
    file,identity=job;data=np.load(file);rows=[];floors=[]
    for name,spec in DESIGNS.items():
        x=data[name]
        if x.ndim==3:x=x[None,:,:,:]
        wells,trials,groups,width=x.shape
        flat=x.reshape(-1,groups,width);total=flat.sum(axis=1)
        original=informative_mask(total,spec['nodes']).reshape(wells,trials)
        minimum=minimum_retained_groups(flat,spec['nodes']).reshape(wells,trials)
        for i in range(wells):
            well=identity.get('wells',['population'])[i]
            for r in np.unique(minimum[i]):
                floors.append(dict(identity,well=well,design=name,minimum_groups=int(r),trials=int((minimum[i]==r).sum())))
        for q in QS:
            expected=group_informative_probability(flat,spec['nodes'],q).reshape(wells,trials).sum(axis=1)
            independent=thinned_informative_probability(total,spec['nodes'],q).reshape(wells,trials).sum(axis=1)
            for i in range(wells):
                rows.append(dict(identity,well=identity.get('wells',['population'])[i],design=name,retention=q,
                    trials=trials,groups=groups,original_informative=int(original[i].sum()),
                    expected_channel_dropout=float(expected[i]),expected_event_thinning=float(independent[i]),
                    expected_trial_dropout=float(q*original[i].sum()),
                    single_channel_sufficient_trials=int((minimum[i]==1).sum())))
        print('COMPLETE',Path(file).stem,name,'channel dropout',flush=True)
    return rows,floors


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((BASE/'group_counts/manifest.json').read_text());jobs=[]
    for meta in manifest['finalspark']:
        jobs.append((str(BASE/'group_counts'/(meta['dataset']+'_counts.npz')),dict(dataset=meta['dataset'],target=meta['target'])))
    for meta in manifest['organoid']:
        jobs.append((str(BASE/'group_counts'/(meta['recording']+'_counts.npz')),
            dict(dataset='kcl_organoid',recording=meta['recording'],target=meta['target'],wells=meta['wells'])))
    rows=[];floors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for a,b in pool.map(analyze,jobs):rows.extend(a);floors.extend(b)
    pd.DataFrame(rows).drop(columns='wells',errors='ignore').to_csv(OUT/'retention.csv',index=False)
    pd.DataFrame(floors).drop(columns='wells',errors='ignore').to_csv(OUT/'minimum_group_counts.csv',index=False)
    (OUT/'interpretation.json').write_text(json.dumps(dict(
        mechanism='Independent electrode retention; each retained electrode contributes all its reference events.',
        comparison='All three mechanisms retain expected fraction q of the reference events, with different within-trial dependence.',
        trials='Expected informative counts sum marginal probabilities. A channel mask can persist across trials; no trial independence is asserted.',
        inference='This exact support calculation does not validate the Poisson response model or identify historical dropout.'),indent=2))
