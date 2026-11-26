"""Run with python -m training.train_demosaic (or use training/launch.py)."""
import sys
from training.cli import parse_args, resolve_mixed_precision


def main():
    args = parse_args()
    args.mixed_precision = resolve_mixed_precision(args.mixed_precision, sys.argv[1:])
    from training.engine import run_training
    run_training(args)


if __name__ == '__main__':
    main()
