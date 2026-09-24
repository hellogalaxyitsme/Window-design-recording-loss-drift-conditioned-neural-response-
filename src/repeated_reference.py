"""Paired retained-reference extension for the frozen E15 mouse replay.

This program is deliberately separate from E15.  It does not overwrite its
single-mask outputs, and it treats masks as Monte Carlo realizations rather
than biological replicates.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from conditional_design import DESIGNS
from replay_core import NAMES, count_times, replay_alternative, replay_distribution


ROOT = Path("results/paper_v1")
OUT = Path("results/repeated_reference_20260924")
BASE_SEED = 202609240
RESPONSES = (1.0, 1.25, 1.5, 2.0)


def stable_id(value):
    """Stable uint32 label; Python's process-randomized hash is not used."""
    return int.from_bytes(hashlib.sha256(str(value).encode("utf8")).digest()[:4], "little")


def rng_for(phase, mask_index, session, unit_id):
    phase_code = 1 if phase == "pilot" else 2
    seq = np.random.SeedSequence([BASE_SEED, phase_code, int(mask_index), stable_id(session), int(unit_id)])
    return np.random.default_rng(seq)


def retained_event_mask(phase, mask_index, session, unit_id, event_count, q):
    """One deterministic mask, subsequently shared by both designs and all R."""
    return rng_for(phase, mask_index, session, unit_id).random(event_count) < q

def assess_counts(counts, response):
    row = {}
    for name in NAMES:
        null, reject, informative, information = replay_distribution(counts[name], DESIGNS[name]["nodes"])
        p = null if response == 1.0 else replay_alternative(counts[name], DESIGNS[name]["nodes"], response)
        row[name] = dict(
            informative_trials=int(informative), conditional_information=float(information),
            zero_support=bool(informative == 0), rejection=float(p @ reject),
        )
    return row


def assess_counts_all(counts, responses):
    """Evaluate all specified response multipliers from one retained mask."""
    return {float(response): assess_counts(counts, float(response)) for response in responses}


def session_rows(meta, phase, mask_index, q, responses):
    folder = ROOT / "external_counts" / meta["session"]
    units = pd.read_parquet(folder / "units.parquet")
    anchors = pd.read_parquet(folder / "trials.parquet").anchor_s.to_numpy()
    source = Path(meta["source"])
    rows = []
    with h5py.File(source, "r") as handle:
        ends = handle["units/spike_times_index"][()]
        starts = np.r_[0, ends[:-1]]
        for unit in units.itertuples():
            times = np.sort(handle["units/spike_times"][int(starts[unit.source_row]):int(ends[unit.source_row])])
            retained = times[retained_event_mask(phase, mask_index, meta["session"], unit.unit_id, len(times), q)]
            counts = count_times(retained, anchors)
            # The one retained event sequence is deliberately reused for both
            # designs and every response multiplier in this mask.
            for response, summary in assess_counts_all(counts, responses).items():
                for design, result in summary.items():
                    rows.append(dict(
                        phase=phase, mask_index=int(mask_index), q=q, response=response,
                        subject=meta["subject"], session=meta["session"], split=meta["split"],
                        unit_id=int(unit.unit_id), design=design, **result,
                    ))
    return rows


def one_mask(manifest, phase, mask_index, q, responses, benchmark=False):
    began = time.perf_counter()
    rows = []
    for meta in manifest:
        rows.extend(session_rows(meta, phase, mask_index, q, responses))
        if benchmark:
            break
    frame = pd.DataFrame(rows)
    # Unit -> session -> subject.  The primary result gives seven evaluation
    # subjects equal weight, exactly as the frozen E15 main summary does.
    session = frame.groupby(["subject", "session", "split", "design", "q", "response"], as_index=False).agg(
        units=("unit_id", "size"), informative_trials=("informative_trials", "mean"),
        conditional_information=("conditional_information", "mean"), zero_support=("zero_support", "mean"),
        rejection=("rejection", "mean"),
    )
    subject = session.groupby(["subject", "split", "design", "q", "response"], as_index=False).agg(
        sessions=("session", "size"), informative_trials=("informative_trials", "mean"),
        conditional_information=("conditional_information", "mean"), zero_support=("zero_support", "mean"),
        rejection=("rejection", "mean"),
    )
    ev = subject[subject.split.eq("evaluation")]
    summaries = []
    for response, response_frame in ev.groupby("response", sort=True):
        wide = response_frame.set_index(["subject", "design"])["rejection"].unstack("design")
        if not benchmark and set(NAMES) != set(wide.columns):
            raise RuntimeError("An evaluation subject lacks a design result")
        if benchmark and wide.empty:
            delta = np.array([np.nan])
        elif not benchmark and len(wide) != 7:
            raise RuntimeError(f"Expected seven evaluation subjects, got {len(wide)}")
        else:
            delta = wide["quadratic_sparse"] - wide["quadratic_equal"]
        summaries.append(dict(
            phase=phase, mask_index=int(mask_index), q=q, response=float(response),
            evaluation_subjects=int(len(delta)) if not np.isnan(delta).all() else 0,
            paired_difference_equal_subject_mean=float(np.nanmean(delta)),
            paired_difference_subject_sd=float(np.nanstd(delta, ddof=1)), elapsed_seconds=time.perf_counter() - began,
        ))
    return frame, session, subject, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--q", type=float, default=0.2)
    parser.add_argument("--responses", nargs="+", type=float, default=[1.5],
                        help="one or more positive response multipliers; use all four for final masks")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.start < 0 or not 0 < args.q <= 1:
        raise ValueError("Invalid mask interval or q")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "external_counts" / "manifest.json").read_text())
    # Pilot and final indices are deliberately disjoint.  The caller records
    # the final count only after examining all ten pilot masks.
    if args.phase == "pilot" and args.start + args.count > 10:
        raise ValueError("Pilot masks are exactly indices 0..9")
    if args.phase == "final" and args.start < 10:
        raise ValueError("Final masks start at index 10")
    responses = tuple(float(x) for x in args.responses)
    if not responses or any(x <= 0 for x in responses):
        raise ValueError("Responses must be positive")
    summaries, sessions, subjects = [], [], []
    for index in range(args.start, args.start + args.count):
        _, session, subject, mask_summaries = one_mask(manifest, args.phase, index, args.q, responses, args.benchmark)
        summaries.extend(mask_summaries); sessions.append(session.assign(mask_index=index, phase=args.phase)); subjects.append(subject.assign(mask_index=index, phase=args.phase))
        for summary in mask_summaries:
            print(json.dumps(summary), flush=True)
    response_label = "allR" if responses == RESPONSES else "R" + "_".join(str(x).replace(".", "p") for x in responses)
    stem = f"{args.phase}_{args.start:03d}_{args.start + args.count - 1:03d}_q{int(args.q * 100):03d}_{response_label}"
    pd.DataFrame(summaries).to_csv(OUT / f"{stem}_mask_summary.csv", index=False)
    pd.concat(sessions, ignore_index=True).to_csv(OUT / f"{stem}_session_summary.csv", index=False)
    pd.concat(subjects, ignore_index=True).to_csv(OUT / f"{stem}_subject_summary.csv", index=False)
    meta = dict(program=Path(__file__).name, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                base_seed=BASE_SEED, phase=args.phase, start=args.start, count=args.count, q=args.q,
                responses=responses, mask_stream="SeedSequence([202609240, phase_code, mask_index, stable_session_sha256, unit_id])",
                aggregation="unit-to-session-to-subject; equal evaluation subjects; paired sparse minus equal", benchmark=args.benchmark)
    (OUT / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
