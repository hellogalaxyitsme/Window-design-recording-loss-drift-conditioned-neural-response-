"""Apply a documented Python-3.10 syntax compatibility fix to pyaxion 1.0.2.

The package advertises Python >=3.9 but its spike module contains Python-3.11
starred subscripts, preventing even its raw-voltage reader from importing.
For numpy indexing, a[*idx] and a[tuple(idx)] have the same intended meaning.
No voltage parsing or signal-processing logic is changed.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

dist=importlib.metadata.distribution('pyaxion')
if dist.version!='1.0.2': raise SystemExit('Compatibility patch is restricted to pyaxion 1.0.2')
path=Path(dist.locate_file('pyaxion/axis_reader/dataset/spike_dataset.py'))
before=path.read_bytes(); original=before.decode('utf8')
old='waveforms[*idx]'; new='waveforms[tuple(idx)]'
out=Path('results/paper_v1/organoid_probe'); out.mkdir(parents=True,exist_ok=True)
if original.count(old)==3:
    (out/'pyaxion_1.0.2_spike_dataset_original.py').write_bytes(before)
    changed=original.replace(old,new); compile(changed,str(path),'exec')
    path.write_text(changed,encoding='utf8')
    record=dict(package='pyaxion',version=dist.version,path=str(path),replacements=3,
        old=old,new=new,before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reason=__doc__)
    (out/'pyaxion_compatibility.json').write_text(json.dumps(record,indent=2))
elif original.count(old)==0 and original.count(new)==3:
    print('Compatibility patch already present')
else:
    raise SystemExit('Unexpected upstream source; refusing an unverified replacement')
from pyaxion.axis_reader import AxisFile
print('Axion reader import passed')
