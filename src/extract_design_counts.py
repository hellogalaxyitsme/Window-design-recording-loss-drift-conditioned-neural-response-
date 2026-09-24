"""Stream FinalSpark event tables for fixed design windows and matched controls."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from conditional_design import DESIGNS
from event_helpers import ns,interval_counts

ROOT=None
OUT=Path('results/paper_v1/finalspark_design_counts')
CANDIDATES=Path('results/pilot_v1')
WAVEFORMS=Path('results/validation_v1')
SHIFTS=(0,-300,-600,-900)

def extract(path):
    start=time.monotonic(); dataset=path.name.split('_')[0]; out=OUT/dataset; out.mkdir(parents=True,exist_ok=True)
    trials=pd.read_parquet(CANDIDATES/f'{dataset}_trials.parquet').sort_values('anchor_ns')
    # Original candidates have >=2 s preceding-trigger separation; earliest window starts -1.8 s.
    assert (trials.previous_gap_s>=2).all() and (trials.next_gap_s>=.5).all()
    trials.to_parquet(out/'trials.parquet',index=False); anchors=trials.anchor_ns.to_numpy()
    windows={}
    for name,spec in DESIGNS.items():
        for shift in SHIFTS:
            centers=np.asarray(spec['centers_ms'])+shift
            windows[f'{name}_shift{shift}']=np.c_[centers-50,centers+50]
    results={target:{key:np.zeros((len(trials),len(w)),dtype=np.int64) for key,w in windows.items()}
             for target in ['all','cap300','min30_cap300']}
    total=0
    with pd.HDFStore(path,'r') as store:
        for chunk in store.select(f'{dataset}_wholelife_events',columns=['time_of_event','max_voltage_uv'],chunksize=1000000):
            t=ns(chunk.time_of_event); v=np.abs(chunk.max_voltage_uv.to_numpy(float))
            if np.any(~np.isfinite(v)): raise ValueError('Nonfinite event amplitude')
            order=np.argsort(t,kind='stable'); t=t[order]; v=v[order]
            selected={'all':t,'cap300':t[v<=300],'min30_cap300':t[(v>=30)&(v<=300)]}
            for target,times in selected.items():
                for key,win in windows.items():
                    for j,(left,right) in enumerate(win):
                        results[target][key][:,j]+=interval_counts(times,
                            anchors+int(round(left*1e6)),anchors+int(round(right*1e6)))
            total+=len(chunk)
            if total%10000000==0: print(f'{dataset}: {total:,} events',flush=True)
    for target,counts in results.items(): np.savez_compressed(out/f'{target}_counts.npz',**counts)
    # Independently cross-check all shift-zero counts against the prior matched event table.
    events=pd.read_parquet(WAVEFORMS/f'{dataset}_waveform_events.parquet')
    ids=trials.trial_id.to_numpy(); waveform={}; comparisons=[]
    for key,win in windows.items():
        if not key.endswith('shift0'): continue
        target=np.zeros((len(ids),len(win)),np.int64)
        for j,(left,right) in enumerate(win):
            # The export has integer-nanosecond timestamps; use identical rounded edges.
            left=round(left*1e6)/1e6; right=round(right*1e6)/1e6
            e=events[(events.relative_ms>=left)&(events.relative_ms<right)]
            for label,mask in [('all',np.ones(len(e),bool)),('cap300',e.max_voltage_uv.abs()<=300),
                               ('min30_cap300',(e.max_voltage_uv.abs()>=30)&(e.max_voltage_uv.abs()<=300))]:
                counts=e[mask].groupby('trial_id').size().reindex(ids,fill_value=0).to_numpy()
                np.testing.assert_array_equal(counts,results[label][key][:,j])
            target[:,j]=e[e.broad_pass].groupby('trial_id').size().reindex(ids,fill_value=0).to_numpy()
        waveform[key]=target
        comparisons.append(dict(design=key,cap300_events=int(results['cap300'][key].sum()),broad_events=int(target.sum())))
    np.savez_compressed(out/'broad_shift0_counts.npz',**waveform)
    report=dict(dataset=dataset,source=str(path),events_scanned=total,trials=len(trials),
        windows_ms={key:win.tolist() for key,win in windows.items()},shift0_crosscheck='passed',
        target_waveform_comparisons=comparisons,elapsed_s=time.monotonic()-start,
        note='Earlier controls are amplitude-filtered event counts; they have no claim of full waveform or uptime verification.',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'provenance.json').write_text(json.dumps(report,indent=2))
    print(dataset,'completed',total,'events',round(time.monotonic()-start,1),'s',flush=True)
