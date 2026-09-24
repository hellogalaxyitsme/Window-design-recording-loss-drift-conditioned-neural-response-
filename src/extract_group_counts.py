"""Extract electrode-specific frozen counts and verify their population sums."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_design import DESIGNS
from event_helpers import ns,interval_counts

BASE=Path('results/paper_v1')
OUT=BASE/'group_counts'


def finalspark(path):
    dataset=path.name.split('_')[0];prior=BASE/'finalspark_design_counts'/dataset
    anchors=pd.read_parquet(prior/'trials.parquet').anchor_ns.to_numpy()
    electrodes=np.arange(64,96) if dataset=='fs369' else np.arange(32)
    counts={name:np.zeros((len(anchors),32,len(spec['nodes'])),np.int32) for name,spec in DESIGNS.items()}
    total=0
    with pd.HDFStore(path,'r') as store:
        for chunk in store.select(dataset+'_wholelife_events',columns=['time_of_event','max_voltage_uv','electrode'],chunksize=1000000):
            t=ns(chunk.time_of_event);v=np.abs(chunk.max_voltage_uv.to_numpy());ch=chunk.electrode.to_numpy()
            assert np.all(np.isfinite(v)) and np.all(np.isin(ch,electrodes))
            for j,electrode in enumerate(electrodes):
                times=np.sort(t[(ch==electrode)&(v<=300)])
                for name,spec in DESIGNS.items():
                    for k,center in enumerate(spec['centers_ms']):
                        counts[name][:,j,k]+=interval_counts(times,
                            anchors+int(round((center-50)*1e6)),anchors+int(round((center+50)*1e6)))
            total+=len(chunk)
    reference=np.load(prior/'cap300_counts.npz')
    for name,x in counts.items():np.testing.assert_array_equal(x.sum(axis=1),reference[name+'_shift0'])
    np.savez_compressed(OUT/(dataset+'_counts.npz'),**counts)
    return dict(dataset=dataset,target='cap300',electrodes=electrodes.tolist(),trials=len(anchors),
        events_scanned=total,population_crosscheck='passed')


def organoid():
    manifest=json.loads((BASE/'organoid_counts/manifest.json').read_text());rows=[]
    for record in manifest:
        stem=Path(record['source']['path']).stem;prior=BASE/'organoid_counts'/stem
        events=pd.read_parquet(prior/'detected_events.parquet');events=events[events.threshold_multiple>=5]
        anchors=np.load(prior/'anchors_s.npy');metadata=pd.read_csv(prior/'wells.csv')
        electrodes=[10*c+r for c in range(1,5) for r in range(1,5)]
        counts={name:np.zeros((len(metadata),len(anchors),16,len(spec['nodes'])),np.int32) for name,spec in DESIGNS.items()}
        for wi,well in enumerate(metadata.well):
            for j,electrode in enumerate(electrodes):
                times=np.sort(events[(events.well==well)&(events.electrode==electrode)].time_s.to_numpy())
                for name,spec in DESIGNS.items():
                    for k,center in enumerate(spec['centers_ms']):
                        counts[name][wi,:,j,k]=np.searchsorted(times,anchors+(center+50)/1000)-np.searchsorted(times,anchors+(center-50)/1000)
        reference=np.load(prior/'counts.npz')
        for name,x in counts.items():np.testing.assert_array_equal(x.sum(axis=2),reference[name+'_threshold5'])
        np.savez_compressed(OUT/(stem+'_counts.npz'),**counts)
        rows.append(dict(recording=stem,target='threshold5',electrodes=electrodes,wells=list(metadata.well),
            trials=len(anchors),population_crosscheck='passed'))
        print('COMPLETE',stem,'group-count crosscheck',flush=True)
    return rows


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--public-only',action='store_true');parser.add_argument('--private-root',type=Path);args=parser.parse_args()
    if not args.public_only and args.private_root is None: parser.error('--private-root is required unless --public-only is used')
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in ([] if args.public_only else sorted(args.private_root.rglob('fs*_package.hdf5'))):
        rows.append(finalspark(path));print('COMPLETE',path.name,'group-count crosscheck',flush=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(finalspark=rows,organoid=organoid()),indent=2))
