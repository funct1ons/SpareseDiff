"""Launch four recipes in-process; GPU/process selection belongs to Accelerate."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Keep direct `python training/launch.py` compatible with module launch.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.cli import resolve_mixed_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='JSON config path (relative to repository root)')
    parser.add_argument('--dry-run', action='store_true')
    args, overrides = parser.parse_known_args()
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    cfg = json.loads(path.read_text(encoding='utf-8'))
    cfa = cfg['cfa_mode']
    if cfa not in ('sparse', 'bayer') or cfg['light_mode'] not in ('normal', 'mixed'):
        parser.error('Invalid experiment axes')
    for arg in overrides:
        option = arg.split('=', 1)[0]
        if len(option) > 2 and option.startswith('--') and '--cfa_mode'.startswith(option):
            parser.error('CFA mode is selected by the recipe; choose another config instead')
    values = dict(cfg['training'], cfa_mode=cfa, light_mode=cfg['light_mode'])
    values.setdefault('output_dir', str(ROOT / 'outputs' / f"{cfa}_{cfg['light_mode']}"))
    values['mixed_precision'] = resolve_mixed_precision(values.get('mixed_precision', 'fp16'), overrides)
    training_argv = []
    for key, value in values.items():
        training_argv.extend([f'--{key}', str(value)])
    training_argv.extend(overrides)
    # Compatibility preview only: never spawn a child or lose Accelerate's rank.
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps([sys.executable, '-m', 'training.train_demosaic'] + training_argv,
                         ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    os.chdir(ROOT)
    from training.cli import parse_args
    from training.engine import run_training
    run_training(parse_args(training_argv))


if __name__ == '__main__':
    main()
