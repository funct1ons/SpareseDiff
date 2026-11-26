"""Command line entry point: python -m pipelines --help."""

import argparse
import os
from pathlib import Path

from training.evaluation.common import sha256, write_json
from training.evaluation.infer import read_config

MODELS = ('bayer_normal', 'bayer_mixed', 'sparse_normal', 'sparse_mixed')
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('expected an integer >= 1')
    return value


def nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError('expected an integer >= 0')
    return value


def illumination(text):
    try:
        left, slash, right = text.partition('/')
        value = float(left) / float(right) if slash else float(left)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError('expected an illumination such as 1 or 1/40') from exc
    if value not in (1.0, 1/30, 1/40, 1/50, 1/80, 1/100, 1/150):
        raise argparse.ArgumentTypeError('choose 1, 1/30, 1/40, 1/50, 1/80, 1/100 or 1/150')
    return value


def add_input(parser, *, staged=False):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--sample', choices=('1', '2'), help='Use a bundled RGB example')
    source.add_argument('--image', type=Path, help='Path to an 8-bit sRGB image')
    if staged:
        source.add_argument('--simulated', type=Path, help='Output directory of simulate')
    parser.add_argument('--crop', nargs=2, type=nonnegative, metavar=('X', 'Y'),
                        help='Top-left of an explicit 256x256 crop after EXIF orientation')
    parser.add_argument('--light-level', type=illumination, help='Default: 1 (normal light)')
    parser.add_argument('--seed', type=nonnegative, help='Default: 42')


def add_model(parser):
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--weights', type=Path, required=True, help='Path to the matching downloaded model.safetensors')
    parser.add_argument('--steps', type=positive, default=30)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')


def arguments():
    parser = argparse.ArgumentParser(description='SparseDiff RGB simulation, CFA sampling, restoration and evaluation')
    cmds = parser.add_subparsers(dest='command', required=True)
    sim = cmds.add_parser('simulate', help='Prepare simulated model inputs without weights')
    add_input(sim)
    sim.add_argument('--cfa', choices=('bayer', 'sparse'), required=True)
    cfa = cmds.add_parser('mosaic', help='Write a model-ready observation and CFA previews')
    add_input(cfa, staged=True)
    cfa.add_argument('--cfa', choices=('bayer', 'sparse'), help='Required when starting from RGB')
    restore = cmds.add_parser('restore', help='Restore an existing observation, without RGB or new noise')
    restore.add_argument('--observation', type=Path, required=True, help='Output directory of mosaic or demo')
    add_model(restore)
    demo = cmds.add_parser('demo', help='Run RGB input through simulation, mosaic and restoration')
    add_input(demo)
    add_model(demo)
    bench = cmds.add_parser('benchmark', help='Evaluate RGB-simulated observations from an image directory')
    bench.add_argument('--input-dir', type=Path, required=True,
                       help='Directory of 8-bit sRGB images (PNG, JPEG, BMP, TIFF, WebP), searched recursively')
    bench.add_argument('--light-level', type=illumination, default=1.0)
    bench.add_argument('--seed', type=nonnegative, default=42)
    bench.add_argument('--no-fsimc', action='store_true', help='Only compute PSNR and SSIM')
    add_model(bench)
    for sub in (sim, cfa, restore, demo, bench):
        sub.add_argument('--output', type=Path, required=True, help='A new output directory')
    return parser


def config_path(model):
    return MODEL_DIR / f'{model}.json'


def weights_hash(model, path):
    if not path.is_file() or path.suffix != '.safetensors':
        raise ValueError('--weights must point to a downloaded .safetensors file')
    for line in (MODEL_DIR / 'checksums.sha256').read_text(encoding='ascii').splitlines():
        expected, relative = line.split(maxsplit=1)
        if relative == f'{model}/model.safetensors':
            value = sha256(path)
            if value != expected:
                raise ValueError(f'Weights SHA-256 does not match {model}; check --model and the download')
            return value
    raise ValueError(f'No release checksum for {model}')


def ensure_model(config, cfa, light):
    if config['cfa_mode'] != cfa:
        raise ValueError('--model does not match the observation CFA')
    if config['light_mode'] == 'normal' and light != 1:
        raise ValueError('Choose a mixed checkpoint for simulated low-light inputs')


def restore_patch(args, condition, metadata, config, digest):
    import torch
    from PIL import Image
    from training.evaluation.infer import load_model, quantize_prediction, sample, sample_seed
    from .stages import PATCH_ID

    cfa, light = metadata['cfa'], metadata['light_level']
    denoiser = load_model(args.weights, config, args.device)
    sampler_seed = sample_seed(metadata['seed'], PATCH_ID, 'initial-noise')
    pred = sample(denoiser, condition[None].to(args.device), seed=sampler_seed, steps=args.steps,
                  method='euler', cfg_scale=0.0, precision=args.precision, constraint_schedule='final',
                  apply_constraint=light == 1 or cfa == 'sparse', constraint_strength=float(light == 1))
    if pred.shape != (1, 3, 256, 256) or not torch.isfinite(pred).all():
        raise ValueError('Invalid sampler output')
    target = args.output / 'restored.png'
    Image.fromarray(quantize_prediction(pred)).save(target)
    write_json(args.output / 'run.json', dict(schema_version=1, model=args.model,
               weights_sha256=digest, model_config_sha256=sha256(config_path(args.model)),
               observation_sha256=metadata['array_sha256'], light_level=light, seed=metadata['seed'],
               sampler_seed=sampler_seed, method='euler', steps=args.steps, cfg_scale=0,
               device=args.device, precision=args.precision, torch=torch.__version__,
               projection='final', projection_applied=(light == 1 or cfa == 'sparse'),
               projection_strength=float(light == 1), quantize='round', restored_sha256=sha256(target)))


def execute(args):
    from . import stages

    if os.path.lexists(args.output):
        raise FileExistsError(f'Output already exists: {args.output}')
    args.output = args.output.resolve()
    if os.path.lexists(args.output):
        raise FileExistsError(f'Output already exists: {args.output}')
    config = read_config(config_path(args.model)) if hasattr(args, 'model') else None
    if args.command == 'benchmark':
        from . import benchmark

        ensure_model(config, config['cfa_mode'], args.light_level)
        digest = weights_hash(args.model, args.weights)
        result = benchmark.run(args, config, config_path(args.model), digest)
        print(result['overall'])
        return
    if args.command == 'restore':
        condition, info = stages.read_artifact(args.observation, 'observation')
        ensure_model(config, info['cfa'], info['light_level'])
        digest = weights_hash(args.model, args.weights)
        args.output.mkdir(parents=True)
        restore_patch(args, condition, info, config, digest)
        return
    if getattr(args, 'simulated', None):
        if any(v is not None for v in (args.crop, args.light_level, args.seed)):
            raise ValueError('With --simulated, crop, light level and seed come from the artifact')
        data, info = stages.read_artifact(args.simulated, 'simulation')
        cfa = info['cfa']
        if args.cfa is not None and args.cfa != cfa:
            raise ValueError('--cfa does not match the simulation')
        args.output.mkdir(parents=True)
    else:
        cfa = config['cfa_mode'] if config is not None else args.cfa
        if cfa is None:
            raise ValueError('--cfa is required when mosaic starts from RGB')
        light = args.light_level if args.light_level is not None else 1.0
        seed = args.seed if args.seed is not None else 42
        if config is not None:
            ensure_model(config, cfa, light)
            digest = weights_hash(args.model, args.weights)
        noise_std = config['lowlight_noise_std'] if config is not None else 0.004
        source = stages.SAMPLES / f'sample{args.sample}.png' if args.sample else args.image
        image = stages.read_input(source, args.crop)
        info = stages.input_metadata(source, cfa, light, seed, noise_std, args.crop)
        data, w = stages.simulate(image, cfa, light, seed, noise_std)
        args.output.mkdir(parents=True)
        stages.write_simulation(args.output, image, data, w, info)
        info['input_sha256'] = sha256(args.output / 'input.png')
    if args.command != 'simulate':
        condition = stages.mosaic(data, cfa)
        stages.write_observation(args.output, condition, info)
    if args.command == 'demo':
        info['array_sha256'] = sha256(args.output / 'observation.npy')
        restore_patch(args, condition, info, config, digest)


def main():
    parser = arguments()
    args = parser.parse_args()
    # Avoid spawning an Inductor worker pool for this short-lived CLI. This
    # preserves compilation; users can override it for their environment.
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
    try:
        execute(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
