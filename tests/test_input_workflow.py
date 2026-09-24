"""Small synthetic recordings exercise file input, numerical output and CLI wiring."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

import extract_external_counts as mouse
import private_data
from conditional_design import DESIGNS

REPO = Path(__file__).resolve().parents[1]

def mouse_fixture(path):
    anchors = np.array([0., 3., 6., 9., 12.])
    centers = np.asarray(DESIGNS['quadratic_equal']['centers_ms']) / 1000
    spikes = np.sort(np.concatenate([np.repeat(a + centers, [1, 3, 3, 1]) for a in anchors[1:-1]]))
    with h5py.File(path, 'w') as file:
        file['general/subject/subject_id'] = 'jlh33'
        file['general/session_id'] = 'synthetic_session'
        file['intervals/trials/start_time'] = anchors
        file['intervals/trials/pulse_number'] = np.ones(len(anchors), dtype=int)
        file['intervals/trials/id'] = np.arange(len(anchors))
        file['units/id'] = np.array([101, 102, 103])
        file['units/group'] = np.array([b'good', b'good', b'mua'])
        file['units/spike_times'] = spikes
        file['units/spike_times_index'] = np.array([len(spikes)] * 3)

def private_fixture(folder):
    base = pd.Timestamp('2025-01-01').value
    stim = pd.DataFrame({'time_of_stim': pd.to_datetime(base + np.array([0, 3, 6]) * 10**9),
                         'electrode': 0, 'nb_stim_pulse': 2, 'stim_shape': 0,
                         'post_trigger_delay': 0, 'a1': 1., 'a2': 1., 'pulse_train_period': 1000,
                         'd1': 1, 'd2': 1, 'stim_polarity': 1,
                         'post_stim_amp_settle': 0, 'post_stim_charge_recov_off': 0})
    centers = DESIGNS['quadratic_equal']['centers_ms']
    times = base + 3 * 10**9 + np.rint(np.asarray(centers) * 10**6).astype(np.int64)
    events = pd.DataFrame({'time_of_event': pd.to_datetime(times), 'electrode': 0, 'max_voltage_uv': -100.})
    package = folder / 'fs437_package.hdf5'
    stim.to_hdf(package, key='fs437_wholelife_stimulations', format='table')
    events.to_hdf(package, key='fs437_wholelife_events', format='table')
    raw_times = np.concatenate([t + np.arange(-3, 4) * 20000 for t in times])
    raw = pd.DataFrame({'time': pd.to_datetime(raw_times), 'electrode': 0,
                        'voltage_uv': np.tile([0., -20., -50., -100., -50., -20., 0.], len(times))})
    raw.to_hdf(folder / 'fs437_raw.hdf5', key='fs437_wholelife_raw', format='table', data_columns=['time'])
    return package

class InputWorkflowTests(unittest.TestCase):
    def test_public_manifest_selects_exact_fixed_inputs(self):
        import public_data
        import download_external
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(download_external, 'download', side_effect=lambda job: job):
                public_data.fetch(root)
            jobs = json.loads((root / 'reviewer_download_manifest.json').read_text())
            self.assertEqual(len(jobs), 18)
            self.assertEqual(sum(job['relative_path'].endswith('.nwb') for job in jobs), 10)
            self.assertEqual(sum(job['relative_path'].endswith('.raw') for job in jobs), 8)
            self.assertTrue(all(job['size'] > 0 for job in jobs))

    def test_organoid_counts_group_sums_and_matched_retention(self):
        import extract_organoid_counts as organoid
        import extract_group_counts as grouped
        import matched_retention as matched
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); base = root / 'results/paper_v1'; source = root / 'AP2_47plusDIV15.raw'
            source.write_bytes(b'synthetic voltage supplied by reader fixture')
            fs = 12500
            voltage = np.rint(np.random.default_rng(92).normal(size=(fs * 4, 2))).astype(np.int16)
            for channel in range(2):
                for center, height in zip([1.15, 1.4, 1.9, 2.15], [8, 12, 7, 14]):
                    voltage[int(center * fs), channel] = -height
            mapping = [dict(well='A1', electrode=11 + i, column=i) for i in range(2)]
            meta = dict(path=str(source), sampling_frequency=fs, voltage_scale=1e-6, duration_s=4.)
            pd.DataFrame([dict(recording_filename_well=source.stem + '_A1', age_DPD=15,
                               cell_line='synthetic', batch='fixture')]).to_csv(root / 'metadata.csv', index=False)
            output = base / 'organoid_counts'
            with patch.object(organoid, 'ROOT', root), patch.object(organoid, 'OUT', output), \
                 patch.object(organoid, 'open_voltage', return_value=(voltage, mapping, meta)):
                record = organoid.extract(source)
            (output / 'manifest.json').write_text(json.dumps([record]))
            group_out = base / 'group_counts'; group_out.mkdir()
            with patch.object(grouped, 'BASE', base), patch.object(grouped, 'OUT', group_out):
                rows = grouped.organoid()
            self.assertEqual(rows[0]['population_crosscheck'], 'passed')
            matched_out = root / 'matched'; matched_out.mkdir()
            with patch.object(matched, 'OUT', matched_out):
                frame = matched.recording((0, output / source.stem))
            self.assertEqual(len(frame), 12)
            self.assertTrue((frame.K <= frame.M).all())
            self.assertTrue(frame.pvalue_mc.between(0, 1).all())

    def test_mouse_extraction_and_analysis_entry_points(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'input.nwb'; work = root / 'work'
            mouse_fixture(source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            output = work / 'results/paper_v1/external_counts'
            with patch.object(mouse, 'OUT', output):
                metadata = mouse.extract(source)
            (output / 'manifest.json').write_text(json.dumps([metadata]))
            counts = np.load(output / 'synthetic_session/counts.npz')['quadratic_equal_shift0']
            np.testing.assert_array_equal(counts[0], np.tile([1, 3, 3, 1], (3, 1)))
            np.testing.assert_array_equal(counts[1], np.zeros((3, 4)))
            self.assertEqual(metadata['selected_units'], 2)
            command = [sys.executable, str(REPO / 'run_analysis.py'), 'analyze', '--work-root', str(work),
                       '--steps', 'retention', 'response', 'replay']
            run = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            replay = pd.read_csv(work / 'results/peer_extension/e15/unit_replay.csv')
            self.assertEqual(len(replay), 32)
            self.assertEqual(set(replay.unit_id), {101, 102})
            self.assertTrue(replay.loc[replay.unit_id == 102, 'zero_support'].all())
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_private_raw_to_waveform_and_design_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); work = root / 'work'
            package = private_fixture(root)
            sources = [package, root / 'fs437_raw.hdf5']
            before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
            private_data.extract(package, work)
            out = work / 'results/paper_v1/finalspark_design_counts/fs437'
            counts = np.load(out / 'cap300_counts.npz')['quadratic_equal_shift0']
            np.testing.assert_array_equal(counts, [[1, 1, 1, 1]])
            broad = np.load(out / 'broad_shift0_counts.npz')['quadratic_equal_shift0']
            np.testing.assert_array_equal(broad, counts)
            self.assertEqual(len(pd.read_parquet(out / 'trials.parquet')), 1)
            self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources])

    def test_missing_input_is_actionable_and_repo_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = subprocess.run([sys.executable, str(REPO / 'run_analysis.py'), 'analyze', '--work-root', temp,
                                  '--steps', 'replay'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('Run public extraction first', run.stderr)
        run = subprocess.run([sys.executable, str(REPO / 'run_analysis.py'), 'simulate', '--work-root', str(REPO)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn('outside this repository', run.stderr)

if __name__ == '__main__':
    unittest.main()
