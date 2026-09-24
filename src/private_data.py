"""Reconstruct the study's private numerical inputs from authorized local exports."""
import argparse
import json
from pathlib import Path
import pandas as pd
from event_helpers import cluster_stimulations
from scan_waveforms import scan
from waveform_support import label_runs, match_events
import extract_design_counts as design

CONFIG = Path(__file__).resolve().parents[1] / 'config/private_extraction.json'

def prepare_candidates(package, output, cfg):
    dataset = package.name.split('_')[0]
    with pd.HDFStore(package, 'r') as store:
        stim = store.select(f'{dataset}_wholelife_stimulations')
    all_trials = cluster_stimulations(stim, cfg)
    trials = all_trials[all_trials.protocol_ok & all_trials.isolated].copy().reset_index(drop=True)
    if trials.empty:
        raise ValueError(f'{dataset}: no isolated protocol candidates; cannot reproduce this recording')
    output.mkdir(parents=True, exist_ok=True)
    trials.to_parquet(output / f'{dataset}_trials.parquet', index=False)
    return trials

def extract(package, work_root):
    package = Path(package).resolve()
    dataset = package.name.split('_')[0]
    cfg = json.loads(CONFIG.read_text())
    raw_path = package.with_name(f'{dataset}_raw.hdf5')
    if not raw_path.is_file():
        raise FileNotFoundError(f'Waveform support requires the paired raw export: {raw_path}')
    candidates = work_root / 'results/pilot_v1'
    waveform = work_root / 'results/validation_v1'
    waveform.mkdir(parents=True, exist_ok=True)
    trials = prepare_candidates(package, candidates, cfg['trigger_selection'])
    scan(package, candidates, waveform, cfg['waveform'])
    completion = json.loads((waveform / f'{dataset}_scan_complete.json').read_text())
    if not completion['shards']:
        raise ValueError(f'{dataset}: no raw samples in candidate windows; waveform comparison is unavailable')
    raw = pd.concat([pd.read_parquet(waveform / f'{dataset}_raw_shards' / name) for name in completion['shards']], ignore_index=True)
    if raw.raw_row.duplicated().any():
        raise ValueError('Duplicate selected raw source row')
    raw, runs = label_runs(raw, cfg['waveform'], trials.set_index('trial_id').anchor_ns)
    events = match_events(pd.read_parquet(waveform / f'{dataset}_window_events.parquet'), raw, runs, trials, cfg['waveform'])
    events.to_parquet(waveform / f'{dataset}_waveform_events.parquet', index=False)
    design.OUT = work_root / 'results/paper_v1/finalspark_design_counts'
    design.CANDIDATES = candidates
    design.WAVEFORMS = waveform
    design.extract(package)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    args = parser.parse_args()
    packages = []
    for dataset in ('fs369', 'fs437'):
        matches = list(args.data_root.rglob(f'{dataset}_package.hdf5'))
        if len(matches) != 1:
            parser.error(f'Expected exactly one {dataset}_package.hdf5 under --data-root; found {len(matches)}')
        packages.extend(matches)
    for package in packages:
        extract(package, args.work_root.resolve())

if __name__ == '__main__':
    main()
