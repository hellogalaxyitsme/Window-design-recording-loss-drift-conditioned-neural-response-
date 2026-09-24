"""Numerical waveform support and event-matching rules."""
import numpy as np
import pandas as pd
from event_helpers import ns as times_ns

def label_runs(raw,cfg,anchors):
    raw=raw.sort_values(['trial_id','electrode','time','raw_row']).reset_index(drop=True)
    raw['time_ns']=times_ns(raw.time)
    # Isolate past feature processing from all samples near/after the trigger.
    raw['pre_segment']=raw.time_ns<raw.trial_id.map(anchors)-45_000_000
    dt=raw.time_ns.diff().to_numpy()
    change=(raw.trial_id.diff()!=0)|(raw.electrode.diff()!=0)|(raw.pre_segment!=raw.pre_segment.shift())|(dt<=0)|(dt>cfg['run_gap_us']*1000)
    raw['run_id']=np.cumsum(change)-1
    runs=raw.groupby('run_id',sort=True).agg(trial_id=('trial_id','first'),electrode=('electrode','first'),
        first_ns=('time_ns','min'),last_ns=('time_ns','max'),samples=('time_ns','size'),
        minimum_uv=('voltage_uv','min'),maximum_uv=('voltage_uv','max'),pre_segment=('pre_segment','first'))
    runs['max_abs_uv']=np.maximum(runs.minimum_uv.abs(),runs.maximum_uv.abs())
    runs['peak_to_peak_uv']=runs.maximum_uv-runs.minimum_uv
    runs['duration_ms']=(runs.last_ns-runs.first_ns)/1e6
    extremes=(raw.voltage_uv.to_numpy()==runs.loc[raw.run_id,'minimum_uv'].to_numpy())|(
        raw.voltage_uv.to_numpy()==runs.loc[raw.run_id,'maximum_uv'].to_numpy())
    runs['extrema_fraction']=pd.Series(extremes).groupby(raw.run_id).mean()
    return raw,runs


def match_events(events,raw,runs,trials,cfg):
    events=events.copy(); events['event_ns']=times_ns(events.time_of_event)
    events['event_id']=np.arange(len(events)); rows=[]
    raw_groups={k:g for k,g in raw.groupby(['trial_id','electrode'],sort=False)}
    anchor=trials.set_index('trial_id').anchor_ns
    for key,e in events.groupby(['trial_id','electrode'],sort=False):
        g=raw_groups.get(key)
        if g is None:
            for r in e.itertuples(): rows.append({'event_id':r.event_id,'matched':False})
            continue
        t=g.time_ns.to_numpy(); target=e.event_ns.to_numpy()
        j=np.searchsorted(t,target); a=np.clip(j-1,0,len(t)-1); b=np.clip(j,0,len(t)-1)
        ix=np.where(np.abs(t[a]-target)<np.abs(t[b]-target),a,b)
        closest=g.iloc[ix]; selected=runs.loc[closest.run_id].reset_index()
        for k,r in enumerate(e.itertuples()):
            s=selected.iloc[k]; time=int(anchor.loc[key[0]])
            truncated=(s.first_ns<=time+cfg['window_left_ms']*1e6+cfg['run_gap_us']*1000) or (
                s.last_ns>=time+(-45 if s.pre_segment else cfg['window_right_ms'])*1e6-cfg['run_gap_us']*1000)
            delta=int(t[ix[k]]-r.event_ns)
            rows.append({'event_id':r.event_id,'matched':abs(delta)<=cfg['raw_match_tolerance_us']*1000,
                'delta_us':delta/1000,'raw_voltage_uv':float(closest.voltage_uv.iloc[k]),'run_id':int(s.run_id),
                'run_max_abs_uv':float(s.max_abs_uv),'run_peak_to_peak_uv':float(s.peak_to_peak_uv),
                'run_duration_ms':float(s.duration_ms),'run_samples':int(s.samples),
                'run_extrema_fraction':float(s.extrema_fraction),'query_truncated':bool(truncated)})
    events=events.merge(pd.DataFrame(rows),on='event_id',validate='one_to_one')
    events['relative_ms']=(events.event_ns-events.trial_id.map(anchor))/1e6
    events['waveform_available']=events.matched & ~events.query_truncated.fillna(True)
    events['broad_pass']=events.waveform_available&(events.max_voltage_uv.abs()<=300)&(events.run_max_abs_uv<=cfg['run_max_uv'])
    events['conservative_pass']=events.broad_pass&(events.max_voltage_uv.abs()>=cfg['conservative_min_peak_uv'])
    return events

