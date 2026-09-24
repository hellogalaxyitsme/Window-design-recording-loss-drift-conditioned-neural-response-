"""Continuous organoid event extraction and fixed-window detector sensitivity."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
from functools import partial
import numpy as np
import pandas as pd
from conditional_design import DESIGNS
from organoid_raw_core import open_voltage,detect_negative_peaks

ROOT=None
OUT=None
THRESHOLDS=[4,5,6,7,8]

def extract(path,recompute=False):
    start=time.monotonic(); out=OUT/path.stem; out.mkdir(parents=True,exist_ok=True)
    code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('organoid_raw_core.py')]}
    configuration=dict(designs=DESIGNS,thresholds=THRESHOLDS,window_width_ms=100,
                       anchor_start_s=2,anchor_step_s=2,anchor_end_margin_s=.3)
    configuration=json.loads(json.dumps(configuration))
    if (out/'completion.json').exists() and not recompute:
        old=json.loads((out/'completion.json').read_text())
        if old['code_hashes']!=code_hashes or old.get('configuration')!=configuration:
            raise ValueError(f'Stale output for {path.name}; use --recompute to rebuild derived counts')
        print('ALREADY COMPLETE',path.name,flush=True); return old
    raw,mapping,meta=open_voltage(path,verify=True); fs=meta['sampling_frequency']; scale=meta['voltage_scale']*1e6
    channels=pd.DataFrame(mapping); wells=sorted(channels.well.unique()); duration=meta['duration_s']
    anchors=np.arange(2.,duration-.3,2.); metadata=pd.read_csv(ROOT/'metadata.csv')
    well_meta=[]
    for well in wells:
        selected=metadata[metadata.recording_filename_well==path.stem+'_'+well]
        if len(selected)!=1: raise ValueError(f'Expected one metadata row for {path.stem}_{well}')
        record=selected.iloc[0].to_dict(); record['well']=well; well_meta.append(record)
    pd.DataFrame(well_meta).to_csv(out/'wells.csv',index=False); channels.to_csv(out/'channels.csv',index=False)
    np.save(out/'anchors_s.npy',anchors)
    counts={f'{name}_threshold{threshold}':np.zeros((len(wells),len(anchors),len(spec['nodes'])),dtype=np.int32)
            for name,spec in DESIGNS.items() for threshold in THRESHOLDS}
    qc=[]; event_parts=[]
    for wi,well in enumerate(wells):
        group=channels[channels.well==well].sort_values('electrode')
        block=np.asarray(raw[:,group.column.to_numpy()])
        for ci,row in enumerate(group.itertuples()):
            adc=block[:,ci]; v=adc.astype(np.float64)*scale
            peak,height,quality=detect_negative_peaks(v,fs)
            quality.update(well=well,electrode=int(row.electrode),column=int(row.column),
                minimum_uv=float(v.min()),maximum_uv=float(v.max()),
                adc_rail_samples=int(np.sum((adc==np.iinfo(np.int16).min)|(adc==np.iinfo(np.int16).max))))
            for threshold in THRESHOLDS:
                keep=height>=threshold*quality['noise_uv'] if quality['valid'] else np.zeros(len(peak),bool)
                times=peak[keep]/fs; quality[f'events_threshold{threshold}']=int(keep.sum())
                for name,spec in DESIGNS.items():
                    for j,center in enumerate(spec['centers_ms']):
                        left=anchors+(center-50)/1000; right=anchors+(center+50)/1000
                        counts[f'{name}_threshold{threshold}'][wi,:,j]+=np.searchsorted(times,right,side='left')-np.searchsorted(times,left,side='left')
            if len(peak):
                event_parts.append(pd.DataFrame(dict(well=well,electrode=int(row.electrode),sample=peak,
                    time_s=peak/fs,negative_peak_uv=height,noise_uv=quality['noise_uv'],threshold_multiple=height/quality['noise_uv'])))
            qc.append(quality)
        del block
        print(path.name,well,'processed',flush=True)
    table=pd.DataFrame(qc); table.to_csv(out/'channel_quality.csv',index=False)
    if not table.valid.all():
        # Preserve diagnostic outputs but do not silently count unusable traces as silence.
        raise ValueError(f'{path.name}: unusable channel noise scale; inspect quality table before aggregation')
    np.savez_compressed(out/'counts.npz',**counts)
    if event_parts: pd.concat(event_parts,ignore_index=True).to_parquet(out/'detected_events.parquet',index=False)
    result=dict(source=meta,code_hashes=code_hashes,configuration=configuration,thresholds=THRESHOLDS,wells=len(wells),anchors=len(anchors),
        channels=len(channels),detected_events_5=int(table.events_threshold5.sum()),
        detected_events_4=int(table.events_threshold4.sum()),elapsed_s=time.monotonic()-start,
        peak_definition='Negative local peaks; minimum separation ceil(1ms*fs); full-trace MAD noise scale; nested amplitude thresholds',
        stimuli='Fixed pseudo-anchors in continuous recordings; no electrical-stimulation ground truth')
    (out/'completion.json').write_text(json.dumps(result,indent=2,default=str))
    print('COMPLETE',path.name,result['detected_events_5'],'threshold-5 events',round(result['elapsed_s'],1),'s',flush=True)
    return result

