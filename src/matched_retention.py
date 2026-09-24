"""Conditional-on-K exchangeable retention; same random subset for both designs."""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_design import DESIGNS,informative_mask
from replay_core import NAMES,membership,count_memberships,matched_count_draws,holm

ROOT=Path('results/paper_v1/organoid_counts'); OUT=Path('results/peer_extension/e16')


def recording(job):
    index,folder=job; events=pd.read_parquet(folder/'detected_events.parquet')
    metadata=pd.read_csv(folder/'wells.csv'); anchors=np.load(folder/'anchors_s.npy'); saved=np.load(folder/'counts.npz')
    rows=[]
    for i,meta in enumerate(metadata.to_dict('records')):
        e=events[events.well==meta['well']]; codes=membership(e.time_s.to_numpy(),anchors)
        original=count_memberships(codes,len(anchors))
        for d,name in enumerate(NAMES): np.testing.assert_array_equal(original[d],saved[name+'_threshold4'][i])
        for threshold in [5,6,7,8]:
            keep=e.threshold_multiple.to_numpy()>=threshold; actual=count_memberships(codes[keep],len(anchors))
            for d,name in enumerate(NAMES): np.testing.assert_array_equal(actual[d],saved[name+f'_threshold{threshold}'][i])
            seed=202609084+index*1000+i*10+threshold; k=int(keep.sum())
            sims=matched_count_draws(codes,len(anchors),k,2000,np.random.default_rng(seed))
            observed=[int(informative_mask(actual[d],DESIGNS[name]['nodes']).sum()) for d,name in enumerate(NAMES)]
            observed.append(observed[1]-observed[0])
            for j,stat in enumerate([*NAMES,'sparse_minus_equal']):
                v=sims[:,j]; obs=observed[j]
                p=min(1.,2*min((1+int((v<=obs).sum()))/2001,(1+int((v>=obs).sum()))/2001))
                rows.append(dict(recording=folder.name,plate=folder.name.split('plus')[0],well=meta['well'],
                    age_DPD=meta['age_DPD'],cell_line=meta['cell_line'],batch=meta['batch'],threshold=threshold,
                    statistic=stat,M=len(e),K=k,trials=len(anchors),observed=obs,reference_mean=float(v.mean()),
                    reference_lower95=float(np.quantile(v,.025,method='inverted_cdf')),
                    reference_upper95=float(np.quantile(v,.975,method='inverted_cdf')),
                    reference_sd=float(v.std(ddof=1)),difference=float(obs-v.mean()),pvalue_mc=p,repetitions=2000,seed=seed))
        print(folder.name,meta['well'],'complete',flush=True)
    frame=pd.DataFrame(rows)
    frame['p_holm_well_family']=frame.groupby('well').pvalue_mc.transform(lambda p:holm(p))
    frame.to_csv(OUT/(folder.name+'.csv'),index=False); return frame


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    folders=sorted(p.parent for p in ROOT.glob('*/detected_events.parquet'))
    if len(folders)!=8: raise ValueError('Expected all eight frozen raw recordings')
    (OUT/'plan.json').write_text(json.dumps(dict(recordings=[p.name for p in folders],repetitions=2000,
        family='12 statistics within each well-recording; additional global Holm',
        law='Uniform K-subset of all M events, implemented through joint membership categories'),indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool: frames=list(pool.map(recording,enumerate(folders)))
    table=pd.concat(frames,ignore_index=True); table['p_holm_global']=holm(table.pvalue_mc)
    assert len(table)==2304
    table.to_csv(OUT/'matched_count_results.csv',index=False)
    table['well_family_flag']=table.p_holm_well_family<.05;table['global_flag']=table.p_holm_global<.05
    summary=table.groupby(['recording','plate','age_DPD','threshold','statistic'],as_index=False).agg(
        wells=('well','size'),observed_mean=('observed','mean'),reference_mean=('reference_mean','mean'),
        difference_mean=('difference','mean'),difference_min=('difference','min'),difference_max=('difference','max'),
        well_family_flags=('well_family_flag','sum'),global_flags=('global_flag','sum'))
    summary.to_csv(OUT/'recording_summary.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(rows=len(table),count_checks='All original and threshold counts reconcile',
        interpretation='Conditional exchangeability diagnostic, not identification of detector mechanism'),indent=2))
