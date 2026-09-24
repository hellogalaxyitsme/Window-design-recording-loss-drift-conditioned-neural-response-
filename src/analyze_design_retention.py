"""Exact thinning-retention curves on recorded count vectors; no Poisson assumption."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from conditional_design import DESIGNS,informative_mask,thinned_informative_probability,minimum_events

QS=[.01,.02,.05,.1,.2,.4,.6,.8,1.]
ROOT=Path('results/paper_v1')

def curves(counts,nodes,identity,unit_ids=None):
    x=np.asarray(counts)
    if x.ndim==2: x=x[None,:,:]
    units,trials,width=x.shape
    unique,inverse=np.unique(x.reshape(-1,width),axis=0,return_inverse=True)
    original=informative_mask(x,nodes).sum(axis=1)
    unit_ids=list(range(units)) if unit_ids is None else list(unit_ids)
    rows=[]
    for q in QS:
        probability=thinned_informative_probability(unique,nodes,q)
        expected=probability[inverse].reshape(units,trials).sum(axis=1)
        for i in range(units):
            rows.append(dict(identity,unit_id=unit_ids[i],trials=trials,retention=q,
                original_informative=int(original[i]),expected_informative=float(expected[i]),
                expected_fraction=float(expected[i]/trials),
                whole_trial_dropout_expected=float(q*original[i]),minimum_events=minimum_events(nodes),
                reference_total_events=int(x[i].sum())))
    return rows

def main(include_private=True):
    out=ROOT/'design_retention'; out.mkdir(parents=True,exist_ok=True); fs=[]; ext=[]; populations=[]
    for dataset in (['fs369','fs437'] if include_private else []):
        folder=ROOT/'finalspark_design_counts'/dataset
        for target in ['all','cap300','min30_cap300']:
            data=np.load(folder/f'{target}_counts.npz')
            for name,spec in DESIGNS.items():
                for shift in [0,-300,-600,-900]:
                    identity=dict(dataset=dataset,target=target,design=name,shift_ms=shift,level='export_population')
                    fs.extend(curves(data[f'{name}_shift{shift}'],spec['nodes'],identity))
    if include_private: pd.DataFrame(fs).to_csv(out/'finalspark_retention.csv',index=False)
    manifest=json.loads((ROOT/'external_counts/manifest.json').read_text())
    for meta in manifest:
        folder=ROOT/'external_counts'/meta['session']; data=np.load(folder/'counts.npz')
        unit=pd.read_parquet(folder/'units.parquet')
        for name,spec in DESIGNS.items():
            for shift in [0,-300,-600,-900]:
                identity=dict(dataset='dandi_000774',subject=meta['subject'],session=meta['session'],split=meta['split'],
                              target='source_good_units',design=name,shift_ms=shift)
                x=data[f'{name}_shift{shift}']
                ext.extend(curves(x,spec['nodes'],dict(identity,level='unit'),unit.unit_id))
                populations.extend(curves(x.sum(axis=0),spec['nodes'],dict(identity,level='session_population')))
        print('COMPLETE',meta['session'],'retention curves',flush=True)
        pd.DataFrame(ext).to_parquet(out/'external_unit_retention.parquet',index=False)
        pd.DataFrame(populations).to_csv(out/'external_population_retention.csv',index=False)
    table=pd.DataFrame(ext)
    # First average units within session, then sessions within subject, so one subject with
    # two sessions or more isolated units does not silently become more biological replicates.
    session=table.groupby(['subject','session','split','design','shift_ms','retention'],as_index=False).agg(
        mean_expected_fraction=('expected_fraction','mean'),mean_informative_trials=('expected_informative','mean'),units=('unit_id','nunique'))
    session.to_csv(out/'external_session_retention.csv',index=False)
    subject=session.groupby(['subject','split','design','shift_ms','retention'],as_index=False).agg(
        mean_expected_fraction=('mean_expected_fraction','mean'),mean_informative_trials=('mean_informative_trials','mean'),sessions=('session','nunique'))
    subject.to_csv(out/'external_subject_retention.csv',index=False)
    (out/'configuration.json').write_text(json.dumps(dict(retention_grid=QS,
        estimand='Expected number of non-singleton conditional count sets after independent retention of recorded events.',
        caveat='Fixed-recording reference; no biological ground truth or claim that actual detector thresholding is independent thinning.'),indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--public-only',action='store_true')
    main(include_private=not parser.parse_args().public_only)
