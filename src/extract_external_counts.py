"""Read sorted-unit events from NWB and count frozen windows, without raw edits."""
from pathlib import Path
import hashlib
import json
import time
import h5py
import numpy as np
import pandas as pd
from conditional_design import DESIGNS

ROOT=None
OUT=Path('results/paper_v1/external_counts')
SHIFTS=(0,-300,-600,-900)

def decode(x):
    return x.decode() if isinstance(x,bytes) else str(x)

def extract(file):
    start=time.monotonic()
    with h5py.File(file,'r') as f:
        subject=decode(f['general/subject/subject_id'][()]); session=decode(f['general/session_id'][()])
        out=OUT/session; out.mkdir(parents=True,exist_ok=True)
        t=f['intervals/trials/start_time'][()]; order=np.argsort(t,kind='stable'); t=t[order]
        previous=np.r_[np.nan,np.diff(t)]; following=np.r_[np.diff(t),np.nan]
        # Earliest control starts -1.8 s; latest response ends .2 s.
        eligible=(previous>=1.85)&(following>=.25)&(f['intervals/trials/pulse_number'][()][order]==1)
        trial=pd.DataFrame(dict(trial_id=f['intervals/trials/id'][()][order],anchor_s=t,
            previous_gap_s=previous,next_gap_s=following,eligible=eligible))
        for key in ['amplitude','pulse_duration','polarity','contacts','run','shape']:
            if key in f['intervals/trials']:
                values=f['intervals/trials'][key][()][order]
                trial[key]=[decode(x) for x in values] if values.dtype.kind in 'OSU' else values
        trial.to_parquet(out/'all_trials.parquet',index=False)
        trial=trial[eligible].copy(); t=trial.anchor_s.to_numpy(); trial.to_parquet(out/'trials.parquet',index=False)
        groups=np.array([decode(x) for x in f['units/group'][()]])
        chosen=np.flatnonzero(groups=='good')
        if not len(chosen): raise ValueError('No curated good units; do not silently change quality rule')
        unit=pd.DataFrame(dict(source_row=chosen,unit_id=f['units/id'][()][chosen]))
        for key in ['unit_id','group','KSlabel','brain_reg','probe','KScontamination','KSamplitude','firing_rate']:
            if key in f['units']:
                values=f['units'][key][()][chosen]
                unit['source_'+key]=[decode(x) for x in values] if values.dtype.kind in 'OSU' else values
        unit.to_parquet(out/'units.parquet',index=False)
        index=f['units/spike_times_index'][()]; beginnings=np.r_[0,index[:-1]]
        window_specs={}
        for name,spec in DESIGNS.items():
            for shift in SHIFTS:
                centers=np.asarray(spec['centers_ms'])+shift
                window_specs[f'{name}_shift{shift}']=np.c_[centers-50,centers+50]/1000
        window_specs['original_four']=np.array([[-.85,-.65],[-.55,-.35],[-.25,-.05],[.05,.25]])
        counts={key:np.zeros((len(chosen),len(t),len(windows)),np.int32) for key,windows in window_specs.items()}
        reordered=0; duplicates=0
        for u,row in enumerate(chosen):
            spikes=f['units/spike_times'][int(beginnings[row]):int(index[row])]
            if np.any(~np.isfinite(spikes)): raise ValueError('Nonfinite spike timestamp')
            if np.any(np.diff(spikes)<0): spikes=np.sort(spikes); reordered+=1
            duplicates+=int(np.sum(np.diff(spikes)==0))
            for key,windows in window_specs.items():
                for w,(left,right) in enumerate(windows):
                    counts[key][u,:,w]=np.searchsorted(spikes,t+right,side='left')-np.searchsorted(spikes,t+left,side='left')
        np.savez_compressed(out/'counts.npz',**counts)
        provenance=dict(source=str(file),subject=subject,session=session,
            split='development' if subject in ['jlh31','jlh32'] else 'evaluation',
            all_trials=len(order),eligible_trials=len(t),source_units=len(index),selected_units=len(chosen),
            quality_rule="source units/group equals 'good'; no activity-based unit or trial filter",
            sorted_units=reordered,duplicate_timestamps_retained=duplicates,
            windows_s={key:windows.tolist() for key,windows in window_specs.items()},
            note='Intervals are supported by trial spacing; sorted-unit event availability is not a hardware uptime log.',
            elapsed_s=time.monotonic()-start,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        print({key:provenance[key] for key in ['subject','session','split','eligible_trials','selected_units','elapsed_s']},flush=True)
        return provenance

