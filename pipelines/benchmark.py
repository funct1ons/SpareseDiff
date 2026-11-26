"""Public RGB-simulated benchmark; no prepared or cached CFA input."""

from hashlib import sha256 as hash_bytes
from io import BytesIO
import json
from pathlib import Path
from platform import python_version
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageCms, __version__ as pillow_version
import torch

from training.evaluation import infer, metrics
from training.evaluation.common import sha256, write_json
from . import stages

SOURCE = 'rgb-simulated'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def stage_rgb_patches(root, files, staging, ground_truth):
    size = stages.SIZE
    patches, sources = [], []
    for relative, source in files:
        payload = source.read_bytes()
        source_sha256 = hash_bytes(payload).hexdigest()
        try:
            image = stages.read_rgb(BytesIO(payload))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f'{source}: {exc}') from exc
        width, height = image.size
        right, bottom = width % size, height % size
        sources.append(dict(path=relative, sha256=source_sha256, size=[width, height],
                            dropped_right=right, dropped_bottom=bottom,
                            patch_count=(width // size) * (height // size)))
        for y in range(0, height - bottom, size):
            for x in range(0, width - right, size):
                filename = f'{len(patches):06d}.png'
                target = staging / filename
                crop = image.crop((x, y, x + size, y + size))
                crop.info.clear()  # Save only the original oriented RGB pixels, like demo input.png.
                crop.save(target)
                # Keep the whole filename (including suffix) as well as its relative directory.
                patches.append(dict(id=f'{relative}/{x}_{y}', dataset=root.name or 'rgb-input',
                                    gt=str(ground_truth / filename), sha256=sha256(target), source=str(source),
                                    source_relative=relative, source_sha256=source_sha256,
                                    xy=[x, y], size=[size, size]))
    return patches, sources


def prepare_rgb(input_dir, output, simulation, versions):
    root = input_dir.resolve()
    if not root.is_dir():
        raise ValueError(f'--input-dir must be an image directory: {root}')
    if output.is_relative_to(root):
        raise ValueError(f'--output must be outside --input-dir: {output}')
    files = sorted((p.relative_to(root).as_posix(), p) for p in root.rglob('*')
                   if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not files:
        raise ValueError(f'No supported RGB image files in {root}')

    size = stages.SIZE
    ground_truth = output / 'ground_truth'
    # Stage on the destination filesystem without creating output or its parents yet.
    staging_parent = next((parent for parent in output.parents if parent.is_dir()), output.parent)
    with TemporaryDirectory(prefix='.sparsediff-', dir=staging_parent) as temp:
        staging = Path(temp) / 'ground_truth'
        staging.mkdir()
        patches, sources = stage_rgb_patches(root, files, staging, ground_truth)
        if not patches:
            raise ValueError(f'No complete {size}x{size} RGB patches in {root}')
        output.mkdir(parents=True, exist_ok=False)
        staging.rename(ground_truth)

    manifest = output / 'manifest.json'
    write_json(manifest, dict(schema_version=1, source=SOURCE, input_dir=str(root),
                             edge='drop', patch_size=size,
                             rgb_convention='8-bit decoded RGB treated as sRGB; EXIF orientation applied before cropping',
                             order='root-relative POSIX path, then y, then x', simulation=simulation,
                             versions=versions, sources=sources, patches=patches))
    return manifest, patches


def run(args, config, recipe, checkpoint_sha256):
    cfa, light = config['cfa_mode'], args.light_level
    sparse = cfa == 'sparse'
    checkpoint = infer.checkpoint_info(args.weights, config, recipe)
    versions = dict(python=python_version(), torch=torch.__version__, numpy=np.__version__,
                    pillow=pillow_version, littlecms=ImageCms.core.littlecms_version if sparse else None,
                    torch_cuda=torch.version.cuda)
    simulation = dict(cfa=cfa, light_level=light, seed=args.seed, noise_std=config['lowlight_noise_std'],
                      w_source='pillow-imagecms-lab-l' if sparse else None,
                      method='mask RGB and Lab L before degradation' if sparse else 'degrade RGB before GBRG sampling',
                      w_sensitivity=3 if sparse else None)
    manifest, patches = prepare_rgb(args.input_dir, args.output, simulation, versions)
    sampler = dict(steps=args.steps, precision=args.precision, method='euler', cfg_scale=0.0,
                   constraint_schedule='final', apply_constraint=light == 1 or sparse,
                   constraint_strength=float(light == 1))
    projection_count = infer.expected_projection_count(sampler['apply_constraint'], args.steps,
                                                       sampler['constraint_schedule'])
    settings = dict(model=args.model, training_config=config, training_config_sha256=sha256(recipe),
                    checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256,
                    manifest=str(manifest), manifest_sha256=sha256(manifest), cfa_input_source=SOURCE,
                    simulation=simulation, versions=versions, device=args.device, label=config['num_classes'],
                    expected_projection_count=projection_count, quantize='round', **sampler)

    denoiser = infer.load_model(checkpoint, config, args.device)
    output = args.output / 'predictions'
    output.mkdir()
    predictions = []
    for index, patch in enumerate(patches):
        patch_id = patch['id']
        image = stages.read_input(patch['gt'])
        data, _ = stages.simulate(image, cfa, light, args.seed, simulation['noise_std'], patch_id=patch_id)
        condition = stages.mosaic(data, cfa)[None].to(args.device)
        sampler_seed = infer.sample_seed(args.seed, patch_id, 'initial-noise')
        pred = infer.sample(denoiser, condition, seed=sampler_seed, **sampler)
        if pred.shape != (1, 3, stages.SIZE, stages.SIZE) or not torch.isfinite(pred).all():
            raise ValueError('Invalid sampler output')
        path = output / f'{index:06d}.png'
        Image.fromarray(infer.quantize_prediction(pred)).save(path)
        predictions.append(dict(id=patch_id, path=str(path), gt_sha256=patch['sha256'], sha256=sha256(path),
                                degradation_seed=infer.sample_seed(args.seed, patch_id, 'degradation'),
                                sampler_seed=sampler_seed))
    pred_manifest = output / 'predictions.json'
    encoded_settings = json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    write_json(pred_manifest, dict(schema_version=1, config=settings, predictions=predictions, dry_run=False,
                                   config_sha256=hash_bytes(encoded_settings).hexdigest()))
    return metrics.run(manifest, pred_manifest, args.output / 'metrics.json', fsimc=not args.no_fsimc)
