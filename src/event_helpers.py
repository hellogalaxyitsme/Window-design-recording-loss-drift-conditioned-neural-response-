"""Timestamp conversion, half-open counts and exported protocol selection."""
import hashlib
import numpy as np
import pandas as pd

def ns(values):
    return pd.to_datetime(values, utc=True).astype('int64').to_numpy()


def interval_counts(sorted_times, left, right, weights=None):
    """Half-open [left,right); works for multiple queries and chunked accumulation."""
    lo = np.searchsorted(sorted_times, left, side='left')
    hi = np.searchsorted(sorted_times, right, side='left')
    if weights is None:
        return hi-lo
    cumulative = np.r_[0., np.cumsum(weights, dtype=float)]
    return cumulative[hi]-cumulative[lo]


def cluster_stimulations(stim, cfg):
    stim = stim.copy()
    stim['_t'] = ns(stim.time_of_stim)
    stim = stim.sort_values(['_t','electrode']).reset_index(drop=True)
    stim['_cluster'] = np.r_[0, np.cumsum(np.diff(stim._t) > cfg['trigger_merge_ms']*1e6)]
    params = [c for c in stim if c not in ['time_of_stim','_t','_cluster']]
    stim['_ok'] = ((stim.nb_stim_pulse == 2) & (stim.stim_shape == 0)
          & (stim.post_trigger_delay == 0) & (stim.a1 > 0) & (stim.a2 > 0)
          & stim.pulse_train_period.isin([1000,2000,10000]))
    grouped = stim.groupby('_cluster', sort=True)
    df = grouped.agg(anchor_ns=('_t','min'),end_ns=('_t','max'),
                    s_n_channels=('electrode','nunique'),s_log_rows=('electrode','size'),
                    protocol_ok=('_ok','all'))
    for c in ['a1','a2','d1','d2','nb_stim_pulse','pulse_train_period','stim_polarity',
              'post_stim_amp_settle','post_stim_charge_recov_off']:
        for op in ['mean','min','max']:
            df[f's_{c}_{op}'] = getattr(grouped[c],op)()
    membership = stim.groupby(['_cluster','electrode']).size().unstack(fill_value=0).gt(0).astype(int)
    membership.columns = [f's_channel_{int(c)}' for c in membership.columns]
    df = df.join(membership).rename_axis('trial_id').reset_index()
    df['previous_gap_s'] = np.r_[np.nan, (df.anchor_ns.to_numpy()[1:]-df.end_ns.to_numpy()[:-1])/1e9]
    df['next_gap_s'] = np.r_[(df.anchor_ns.to_numpy()[1:]-df.end_ns.to_numpy()[:-1])/1e9, np.nan]
    # Unknown boundary gaps are excluded, not treated as infinite isolation.
    df['isolated'] = (df.previous_gap_s >= cfg['previous_trigger_gap_s']) & (df.next_gap_s >= cfg['next_trigger_gap_s'])
    df['pattern'] = ''
    candidate_ids = df.loc[df.protocol_ok & df.isolated,'trial_id']
    hashes = {}
    for idx,g in stim[stim._cluster.isin(candidate_ids)].groupby('_cluster'):
        canonical = g[params].sort_values(params).to_json(orient='values', double_precision=12)
        hashes[idx] = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    df['pattern'] = df.trial_id.map(hashes).fillna('')
    return df

