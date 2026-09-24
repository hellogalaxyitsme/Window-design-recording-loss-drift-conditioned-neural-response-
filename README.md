# Window design and recording loss in drift-conditioned neural response inference

Numerical analysis code for Window design and recording loss in drift-conditioned neural response inference. No recordings or datasets are bundled in the codebase.

## Install and check

The original numerical environment used Python 3.10. Install the pinned dependencies in an isolated environment; the commands below use Linux shell syntax.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_analysis.py test
.venv/bin/python run_analysis.py demo
```

On Windows use `.venv\Scripts\python.exe`. The entry point sets its own import path. The demo needs no recordings; the tests include small synthetic recording fixtures. Tests and demonstrations do not reproduce the full empirical study.

## Public recordings

Use external directories for source data and working outputs. The fixed public inputs need approximately 25 GB plus space for extracted events and results.

```bash
.venv/bin/python run_analysis.py public --data-root /storage/neural-data --work-root /storage/neural-work --download --extract
.venv/bin/python run_analysis.py simulate --work-root /storage/neural-work
.venv/bin/python run_analysis.py curves --work-root /storage/neural-work
.venv/bin/python run_analysis.py analyze --work-root /storage/neural-work
.venv/bin/python run_analysis.py repeated-reference --work-root /storage/neural-work --phase pilot
.venv/bin/python run_analysis.py repeated-reference --work-root /storage/neural-work --phase final
```

The public inputs are ten NWB sessions from [DANDI 000774](https://dandiarchive.org/dandiset/000774/0.260520.1753), version 0.260520.1753 (CC-BY-4.0), and eight continuous organoid recordings from the [King's College London collection](https://doi.org/10.18742/30394168.v1) (CC0). Public asset identifiers, download URLs, attribution metadata and the required well mapping are in `config/public/`. Downloaded files are checked against recorded sizes and available source MD5 digests; local SHA-256 hashes are also recorded. Preserve source attribution and access terms.

The Axion reader's documented Python 3.10 indexing-syntax fix is applied to `pyaxion==1.0.2` in the active environment before loading raw organoid files. Use the isolated environment above. Raw recordings are opened read-only.

`analyze` runs retention curves, response controls, electrode-group loss, original single-mask conditional replay and matched-count retention, in that order. Select individual stages with `--steps retention response groups replay matched`. Large simulations and raw-data extraction are CPU and storage intensive; running them replaces their corresponding outputs in the chosen work directory.

The repeated-reference commands use ten pilot masks (indices 0–9, response 1.5) and twenty separate final masks (10–29, responses 1, 1.25, 1.5 and 2), at retention 0.2. This reproduces the reported fixed study settings; it does not reselect the number of masks. Retained masks are shared across designs and response multipliers. Fixed seeds and settings are recorded in `config/final_study.json` and the numerical modules.

## Authorized FinalSpark inputs

Obtain access to the original exports from FinalSpark. Place the extracted `fs369_package.hdf5`, `fs369_raw.hdf5`, `fs437_package.hdf5` and `fs437_raw.hdf5` in an external directory, with each raw file beside its package file. Both paired raw files are needed for the waveform-support sensitivity analysis. No private exports or granular derived tables are distributed here.

```bash
.venv/bin/python run_analysis.py private --data-root /storage/private-exports --work-root /storage/neural-work
.venv/bin/python run_analysis.py analyze --work-root /storage/neural-work --include-private --private-root /storage/private-exports
```

Run public extraction first for the combined analysis. The private command reconstructs all protocol candidates, scans the selected waveform windows, matches exported events and computes the fixed design/control counts. It preserves inactive candidates. The extraction rules are in `config/private_extraction.json`; undocumented acquisition timing and event identity are not inferred.

## Numerical outputs

All paths below are relative to `--work-root`. The commands produce CSV, parquet, NPZ and JSON numerical outputs, without plotting or typesetting.

| Calculation | Output location |
|---|---|
| Pooling comparison (SD01) | `results/paper_v1/pooling_simulation_v2/` |
| Graded departures (SD02) | `results/peer_extension/e14/` |
| Original replay, information and support (SD03, SD06, SD07, SD12) | `results/peer_extension/e15/` |
| Matched retention (SD04, SD10) | `results/peer_extension/e16/` |
| Response controls and private response targets (SD05, SD09) | `results/paper_v1/design_response/` |
| Reference retention (SD08) | `results/paper_v1/design_retention/` |
| Grouped retention (SD11) | `results/paper_v1/group_dropout/` |
| Paired repeated references (SD13) | `results/repeated_reference_20260924/` |

Some supplementary tables aggregate these outputs; retain zero-support units and the stated session-to-subject weighting. Exact retention calculations describe the specified loss mechanism. Conditional response/replay calculations additionally depend on the stated count model and do not establish biological responsiveness.
