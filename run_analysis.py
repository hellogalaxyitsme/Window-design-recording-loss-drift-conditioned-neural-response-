"""Entry point for numerical analyses; inputs and outputs stay outside the repository."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
STEPS = ('retention', 'response', 'groups', 'replay', 'matched')

def external_directory(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT or path.is_relative_to(ROOT):
        raise argparse.ArgumentTypeError('Use a data/work directory outside this repository.')
    return path

def run(script, *args, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(SRC) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(SRC / script), *map(str, args)], cwd=cwd, env=env, check=True)

def require(path, message):
    if not path.exists():
        raise FileNotFoundError(f'{message}: {path}')

def analyze(args):
    base = args.work_root / 'results/paper_v1'
    flags = [] if args.include_private else ['--public-only']
    for step in args.steps:
        if step in ('retention', 'response', 'replay'):
            require(base / 'external_counts/manifest.json', 'Run public extraction first')
            if args.include_private:
                for ds in ('fs369', 'fs437'):
                    require(base / 'finalspark_design_counts' / ds / 'cap300_counts.npz', 'Run private extraction first')
            script = {'retention': 'analyze_design_retention.py', 'response': 'analyze_design_response.py', 'replay': 'conditional_replay.py'}[step]
            run(script, *flags, cwd=args.work_root)
        elif step == 'groups':
            require(base / 'organoid_counts/manifest.json', 'Run organoid extraction first')
            extra = flags
            if args.include_private:
                if args.private_root is None:
                    raise ValueError('--private-root is required for private grouped counts')
                extra = ['--private-root', args.private_root]
            run('extract_group_counts.py', *extra, cwd=args.work_root)
            run('analyze_group_dropout.py', cwd=args.work_root)
        else:
            require(base / 'organoid_counts/manifest.json', 'Run organoid extraction first')
            run('matched_retention.py', cwd=args.work_root)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('demo')
    sub.add_parser('test')
    p = sub.add_parser('public', help='Download and/or extract the fixed public recordings')
    p.add_argument('--data-root', type=external_directory, required=True)
    p.add_argument('--work-root', type=external_directory, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--extract', action='store_true')
    p = sub.add_parser('private', help='Reconstruct FinalSpark candidate, waveform and design counts')
    p.add_argument('--data-root', type=external_directory, required=True)
    p.add_argument('--work-root', type=external_directory, required=True)
    p = sub.add_parser('simulate', help='Run the fixed pooling and graded-departure simulations')
    p.add_argument('--work-root', type=external_directory, required=True)
    p = sub.add_parser('curves', help='Compute analytical support/information curves')
    p.add_argument('--work-root', type=external_directory, required=True)
    p = sub.add_parser('analyze', help='Analyze previously extracted counts')
    p.add_argument('--work-root', type=external_directory, required=True)
    p.add_argument('--steps', nargs='+', choices=STEPS, default=list(STEPS))
    p.add_argument('--include-private', action='store_true')
    p.add_argument('--private-root', type=external_directory)
    p = sub.add_parser('repeated-reference', help='Reproduce pilot or final paired retained references')
    p.add_argument('--work-root', type=external_directory, required=True)
    p.add_argument('--phase', choices=('pilot', 'final'), required=True)
    args = parser.parse_args()
    try:
        if hasattr(args, 'work_root'):
            args.work_root.mkdir(parents=True, exist_ok=True)
        if args.command == 'demo':
            with tempfile.TemporaryDirectory(prefix='neural-count-demo-') as temp:
                run('worked_example.py', cwd=temp)
                run('reviewer_demo.py', cwd=temp)
        elif args.command == 'test':
            env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests')], cwd=ROOT, env=env, check=True)
        elif args.command == 'public':
            if not (args.download or args.extract):
                parser.error('Specify --download and/or --extract')
            options = (['--download'] if args.download else []) + (['--extract'] if args.extract else [])
            run('public_data.py', '--data-root', args.data_root, '--work-root', args.work_root, *options, cwd=args.work_root)
        elif args.command == 'private':
            require(args.data_root, 'Private source directory not found')
            run('private_data.py', '--data-root', args.data_root, '--work-root', args.work_root, cwd=args.work_root)
        elif args.command == 'simulate':
            run('simulate_pooling.py', cwd=args.work_root)
            run('robustness.py', cwd=args.work_root)
        elif args.command == 'curves':
            run('analytical_curves.py', cwd=args.work_root)
        elif args.command == 'analyze':
            analyze(args)
        else:
            require(args.work_root / 'results/paper_v1/external_counts/manifest.json', 'Run public extraction first')
            cfg = json.loads((ROOT / 'config/final_study.json').read_text())['repeated_reference']
            interval = cfg['pilot_masks' if args.phase == 'pilot' else 'final_masks']
            responses = [1.5] if args.phase == 'pilot' else cfg['responses']
            run('repeated_reference.py', '--phase', args.phase, '--start', interval[0], '--count', interval[1] - interval[0] + 1,
                '--q', cfg['q'], '--responses', *responses, cwd=args.work_root)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f'Input error: {exc}\n')
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, 'Analysis failed; see the diagnostic above.\n')

if __name__ == '__main__':
    main()
