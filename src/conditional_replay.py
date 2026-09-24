"""Observed-support conditional replay, not biological response validation."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from conditional_design import DESIGNS
from replay_core import NAMES,count_times,replay_distribution,replay_alternative,wilson

ROOT=Path('results/paper_v1'); OUT=Path('results/peer_extension/e15')


def summarize_reference(counts,identity,rng):
    rows=[]
    for name in NAMES:
        x=counts[name]; nodes=DESIGNS[name]['nodes']
        null,reject,informative,information=replay_distribution(x,nodes)
        for response in [1.,1.25,1.5,2.]:
            p=null if response==1 else replay_alternative(x,nodes,response)
            if len(p)!=len(null): raise AssertionError('Conditional support changed under alternative')
            power=float(p@reject); sample=rng.choice(len(p),size=2000,p=p)
            k=int(reject[sample].sum()); lo,hi=wilson(k,2000)
            rows.append(dict(identity,design=name,trials=len(x),informative_trials=informative,
                conditional_information=information,zero_support=informative==0,response=response,
                exact_conditional_rejection=power,mc_rejections=k,mc_repetitions=2000,
                mc_rejection_rate=k/2000,mc_lower95=lo,mc_upper95=hi))
    return rows


def external(job):
    index,meta=job; folder=ROOT/'external_counts'/meta['session']
    units=pd.read_parquet(folder/'units.parquet'); anchors=pd.read_parquet(folder/'trials.parquet').anchor_s.to_numpy()
    saved=np.load(folder/'counts.npz'); rows=[]
    seed=202609082+index*10000; replayseed=202609083+index*10000
    with h5py.File(meta['source'],'r') as f:
        ends=f['units/spike_times_index'][()]; starts=np.r_[0,ends[:-1]]
        for i,unit in enumerate(units.itertuples()):
            times=np.sort(f['units/spike_times'][int(starts[unit.source_row]):int(ends[unit.source_row])])
            for q in [1.,.2]:
                retained=times if q==1 else times[np.random.default_rng(seed+i).random(len(times))<q]
                counts=count_times(retained,anchors)
                if q==1:
                    for name in NAMES: np.testing.assert_array_equal(counts[name],saved[name+'_shift0'][i])
                identity=dict(dataset='mouse',subject=meta['subject'],session=meta['session'],split=meta['split'],
                    unit_id=int(unit.unit_id),retention=q,thinning_seed=seed+i,replay_seed=replayseed+i)
                rows.extend(summarize_reference(counts,identity,np.random.default_rng(replayseed+i)))
            if i%50==0: print(meta['session'],'unit',i,'of',len(units),flush=True)
    pd.DataFrame(rows).to_csv(OUT/(meta['session']+'.csv'),index=False)
    print('COMPLETE',meta['session'],flush=True); return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--public-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'external_counts/manifest.json').read_text())
    (OUT/'plan.json').write_text(json.dumps(dict(sessions=[m['session'] for m in manifest],
        retentions=[1,.2],responses=[1,1.25,1.5,2],repetitions=2000,
        reference='One fixed event mask per unit shared between designs; no biological calibration claim'),indent=2))
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for part in pool.map(external,enumerate(manifest)): rows.extend(part)
    for i,dataset in enumerate([] if args.public_only else ['fs369','fs437']):
        source=ROOT/'finalspark_design_counts'/dataset/'cap300_counts.npz'; saved=np.load(source)
        counts={name:saved[name+'_shift0'] for name in NAMES}
        rows.extend(summarize_reference(counts,dict(dataset=dataset,subject=dataset,session=dataset,
            split='secondary',unit_id=0,retention=1.,thinning_seed=-1,replay_seed=202609083+200000+i),
            np.random.default_rng(202609083+200000+i)))
    table=pd.DataFrame(rows); table.to_csv(OUT/'unit_replay.csv',index=False)
    unique=table[table.response==1].copy(); summaries=[]
    keys=['dataset','subject','session','split','retention','design']
    for key,g in unique.groupby(keys,sort=False):
        row=dict(zip(keys,key)); row.update(units=len(g),zero_support_fraction=float(g.zero_support.mean()))
        for col in ['informative_trials','conditional_information']:
            row[col+'_mean']=float(g[col].mean())
            for label,q in [('q0',0),('q25',.25),('median',.5),('q75',.75),('q100',1)]: row[col+'_'+label]=float(g[col].quantile(q))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'session_support_distribution.csv',index=False)
    session=table.groupby(keys+['response'],as_index=False).agg(units=('unit_id','size'),
        mean_power=('exact_conditional_rejection','mean'),zero_support_fraction=('zero_support','mean'),
        mean_information=('conditional_information','mean'),mean_informative=('informative_trials','mean'))
    session.to_csv(OUT/'session_replay.csv',index=False)
    subject=session.groupby(['dataset','subject','split','retention','design','response'],as_index=False).agg(
        sessions=('session','size'),mean_power=('mean_power','mean'),zero_support_fraction=('zero_support_fraction','mean'),
        mean_information=('mean_information','mean'),mean_informative=('mean_informative','mean'))
    subject.to_csv(OUT/'subject_replay.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(rows=len(table),reference_checks='All unthinned mouse counts exactly match prior extraction',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__': main()
