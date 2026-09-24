"""Download and extract the fixed public study recordings."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def command(script,*args):
    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name(script+'.py')),*args],check=True,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))


SOURCE_ROOT=Path(__file__).resolve().parents[1]
def fetch(root):
    import download_external as dl
    dl.ROOT=root; root.mkdir(parents=True,exist_ok=True);jobs=[]
    meta=SOURCE_ROOT/'config/public'
    assets=json.loads((meta/'dandi_000774_assets.json').read_text())
    assert assets['count']==len(assets['results'])
    for a in assets['results']:
        jobs.append(dict(relative_path='dandi_000774/'+a['path'],size=a['size'],
            url=f'https://api.dandiarchive.org/api/assets/{a["asset_id"]}/download/',license='CC-BY-4.0'))
    files={}
    for record in ['kcl_raw_2','kcl_raw_3','kcl_raw_4']:
        for f in json.loads((meta/(record+'.json')).read_text())['files']:files[f['name']]=f
    raw=[]
    for plate in ['AP2_47','AP3_47','AP4_45','AP6_46']:
        for day in [15,37]:
            f=files[f'{plate}plusDIV{day}.raw']
            raw.append(dict(relative_path='kcl_organoid/raw/'+f['name'],size=f['size'],url=f['download_url'],md5=f['computed_md5'],license='CC0'))
    jobs+=raw;(root/'kcl_raw_download_plan.json').write_text(json.dumps(raw,indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(dl.download,jobs))
    (root/'reviewer_download_manifest.json').write_text(json.dumps(results,indent=2))
    if any('error' in r for r in results):raise RuntimeError('Download failed; inspect manifest')


def extract(root,work_root):
    import extract_external_counts as mouse
    command('patch_pyaxion_python310')
    import extract_organoid_counts as org
    mouse.ROOT=root/'dandi_000774'; mouse.OUT=work_root/'results/paper_v1/external_counts'
    org.ROOT=root/'kcl_organoid'; org.OUT=work_root/'results/paper_v1/organoid_counts'
    org.ROOT.mkdir(parents=True,exist_ok=True)
    metadata=SOURCE_ROOT/'config/public/kcl_well_metadata.csv'
    source=json.loads((SOURCE_ROOT/'config/public/kcl_summary.json').read_text())
    expected=next(f['computed_md5'] for f in source['files'] if f['name']=='metadata.csv')
    if hashlib.md5(metadata.read_bytes()).hexdigest()!=expected:
        raise ValueError('Bundled public well metadata does not match the source MD5')
    shutil.copyfile(metadata,org.ROOT/'metadata.csv')
    mouse.OUT.mkdir(parents=True,exist_ok=True);org.OUT.mkdir(parents=True,exist_ok=True)
    files=sorted(mouse.ROOT.rglob('*.nwb'))
    if len(files)!=10:raise ValueError('Need the ten frozen mouse assets; run download stage')
    manifest=[mouse.extract(f) for f in files]
    (mouse.OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    raw=[org.ROOT/'raw'/f'{plate}plusDIV{day}.raw' for plate in ['AP2_47','AP3_47','AP4_45','AP6_46'] for day in [15,37]]
    manifest=[org.extract(f,recompute=True) for f in raw]
    (org.OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))



def main():
    parser=argparse.ArgumentParser(description="Download or extract the public study inputs.")
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--work-root",type=Path,required=True)
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--extract",action="store_true")
    args=parser.parse_args()
    if not(args.download or args.extract): parser.error("specify --download and/or --extract")
    if args.download: fetch(args.data_root)
    if args.extract: extract(args.data_root,args.work_root)

if __name__ == "__main__": main()
