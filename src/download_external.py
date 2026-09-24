"""Download explicitly selected public reference data with a provenance manifest."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = None

def download(job):
    target = ROOT / job['relative_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if not target.exists() or target.stat().st_size != job['size']:
        temp = target.with_suffix(target.suffix + '.partial')
        for attempt in range(8):
            try:
                offset=temp.stat().st_size if temp.exists() else 0
                if offset>job['size']: raise ValueError('Partial download is larger than the source manifest')
                if offset<job['size']:
                    headers={'User-Agent': 'Neural-recording-research/1.0'}
                    if offset: headers['Range']=f'bytes={offset}-'
                    req = urllib.request.Request(job['url'], headers=headers)
                    with urllib.request.urlopen(req, timeout=90) as response:
                        append=offset>0 and response.status==206
                        if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise ValueError('Unexpected HTTP range response')
                        with temp.open('ab' if append else 'wb') as out:
                            while block := response.read(4 * 1024 * 1024): out.write(block)
                if temp.stat().st_size != job['size']:
                    raise ValueError(f'File size mismatch: {temp.stat().st_size} != {job["size"]}')
                temp.replace(target)
                break
            except Exception as exc:
                if attempt == 7:
                    return dict(job, error=repr(exc))
    hashes = {name: hashlib.new(name) for name in ['sha256', 'md5']}
    with target.open('rb') as handle:
        while block := handle.read(4 * 1024 * 1024):
            for h in hashes.values(): h.update(block)
    actual = {name: h.hexdigest() for name, h in hashes.items()}
    if job.get('md5') and actual['md5'] != job['md5']:
        raise ValueError(f'MD5 mismatch: {target}')
    result = dict(job, local_path=str(target), hashes=actual, elapsed_s=time.monotonic()-start)
    print(f'COMPLETE {target.name}: {job["size"]:,} bytes in {result["elapsed_s"]:.1f}s', flush=True)
    return result

