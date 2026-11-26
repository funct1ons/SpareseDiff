"""Run one 256×256 synthetic or user-supplied RGB patch through a released model."""

import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from training.evaluation.cached_cfa import masks
from training.evaluation.common import sha256, write_json
from training.evaluation.infer import read_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('bayer_normal', 'bayer_mixed', 'sparse_normal', 'sparse_mixed'))
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--image', type=Path, help='An RGB 256×256 patch; omit to use a synthetic gradient')
    parser.add_argument('--cfa-w', type=Path, help='Aligned masked grayscale W plane, for Sparse only')
    parser.add_argument('--output', required=True, type=Path, help='New prediction PNG path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', default='fp32', choices=('fp32', 'bf16'))
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--light-level', type=float, default=1.0)
    args = parser.parse_args()
    if args.weights.suffix != '.safetensors':
        parser.error('--weights must be a .safetensors file')
    if args.output.suffix.lower() != '.png' or args.output.exists():
        parser.error('--output must be a new PNG path')
    config_path = Path(__file__).resolve().parents[1] / 'models' / f'{args.model}.json'
    config = read_config(config_path)
    sparse = config['cfa_mode'] == 'sparse'
    if args.cfa_w and (not sparse or not args.image):
        parser.error('--cfa-w requires a Sparse model and an aligned --image')
    if sparse and args.image and not args.cfa_w:
        parser.error('A Sparse --image requires its aligned masked --cfa-w')
    with TemporaryDirectory(prefix='sparsediff-example-') as tmp:
        folder = Path(tmp)
        image = args.image.resolve() if args.image else folder / 'gradient.png'
        if not args.image:
            y, x = np.indices((256, 256), dtype=np.uint16)
            rgb = np.stack((x, y, (x + y) // 2), axis=-1).astype(np.uint8)
            Image.fromarray(rgb).save(image)
        patch = dict(id='example', dataset='example', gt=str(image), sha256=sha256(image),
                     source=str(image), xy=[0, 0], size=[256, 256])
        if sparse:
            if args.cfa_w:
                w = args.cfa_w.resolve()
            else:
                w = folder / 'synthetic_w.png'
                _, mask = masks(256, 256)
                gray = Image.fromarray(rgb).convert('L')
                Image.fromarray(np.asarray(gray) * mask).save(w)
                print('Using a synthetic W plane for the demo, not benchmark W.')
            patch.update(w=str(w), w_sha256=sha256(w))
        manifest = folder / 'manifest.json'
        write_json(manifest, dict(schema_version=1, patches=[patch]))
        result = run(manifest=manifest, checkpoint=args.weights, training_config=config_path,
                     provenance_config=config_path, output_dir=folder / 'predictions',
                     steps=args.steps, light_level=args.light_level, seed=42,
                     device=args.device, precision=args.precision, constraint_schedule='final',
                     w_source='cached' if sparse else None,
                     execute=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(result['predictions'][0]['path'], 'rb') as src, args.output.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        print(f"Checkpoint SHA-256: {result['config']['checkpoint_sha256']}")
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
