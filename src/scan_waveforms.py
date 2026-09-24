"""Read-only raw timestamp scan and candidate-window extraction."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from event_helpers import ns as times_ns

def source_signature(path):
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}


def select_windows(times,anchors,left,right):
    if len(anchors)>1 and np.any(np.diff(anchors)<right-left): raise ValueError('Windows overlap')
    ids=np.searchsorted(anchors+left,times,side='right')-1
    valid=ids>=0
    valid &= times < anchors[np.maximum(ids,0)]+right
    return valid,ids[valid]


def scan(package,pilot,out,cfg):
    ds=package.name.split('_')[0]; raw=package.with_name(f'{ds}_raw.hdf5')
    complete=out/f'{ds}_scan_complete.json'
    if complete.exists():
        saved=json.loads(complete.read_text())
        if saved['raw_source']!=source_signature(raw) or saved['config']!=cfg: raise ValueError('Completed scan signature mismatch')
        print(f'{ds}: reuse completed scan',flush=True); return
    folder=out/f'{ds}_raw_shards'; folder.mkdir(exist_ok=True)
    trials=pd.read_parquet(pilot/f'{ds}_trials.parquet').sort_values('anchor_ns').reset_index(drop=True)
    anchors=trials.anchor_ns.to_numpy(); trial_ids=trials.trial_id.to_numpy()
    left=int(cfg['window_left_ms']*1e6); right=int(cfg['window_right_ms']*1e6)
    blocks=[]; shards=[]; selected=0
    with pd.HDFStore(raw,'r') as store:
        key=f'{ds}_wholelife_raw'; st=store.get_storer(key); total=int(st.nrows)
        np.testing.assert_array_equal(st.table.read(0,3,field='time').astype('int64'),times_ns(store.select(key,start=0,stop=3).time))
        for start in range(0,total,cfg['raw_block_rows']):
            stop=min(start+cfg['raw_block_rows'],total)
            t=st.table.read(start=start,stop=stop,field='time').astype('int64')
            if np.any(t==np.iinfo(np.int64).min): raise ValueError('Null raw timestamp')
            blocks.append({'row_start':start,'row_stop':stop,'min_ns':int(t.min()),'max_ns':int(t.max())})
            mask,ids=select_windows(t,anchors,left,right)
            if mask.any():
                frame=store.select(key,start=start,stop=stop).loc[mask].copy()
                frame['trial_id']=trial_ids[ids]; frame['raw_row']=np.arange(start,stop)[mask]
                name=f'block_{start:012d}.parquet'; frame.to_parquet(folder/name,index=False)
                shards.append(name); selected+=len(frame)
            if start%(cfg['raw_block_rows']*500)==0:
                print(f'{ds}: raw timestamps {stop:,}/{total:,}; retained {selected:,} rows',flush=True)
    pd.DataFrame(blocks).to_parquet(out/f'{ds}_raw_blocks.parquet',index=False)
    event_parts=[]
    with pd.HDFStore(package,'r') as store:
        for frame in store.select(f'{ds}_wholelife_events',chunksize=500_000):
            t=times_ns(frame.time_of_event); mask,ids=select_windows(t,anchors,left,right)
            if mask.any():
                frame=frame.loc[mask].copy(); frame['trial_id']=trial_ids[ids]; event_parts.append(frame)
    events=pd.concat(event_parts,ignore_index=True)
    events.to_parquet(out/f'{ds}_window_events.parquet',index=False)
    complete.write_text(json.dumps({'dataset':ds,'raw_source':source_signature(raw),'package_source':source_signature(package),
        'raw_rows_indexed':total,'selected_raw_rows':selected,'selected_events':len(events),'shards':shards,'config':cfg,
        'created_utc':str(pd.Timestamp.now(tz='UTC')),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    print(f'{ds}: scan complete; selected raw rows {selected:,}, events {len(events):,}',flush=True)

