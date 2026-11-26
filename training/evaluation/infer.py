"""Strict checkpoint inference. Dry-run is default; no random production model.

Sparse evaluation requires explicit --w-source lab-l (new simulated W, NOT
verified identical to training's cached W) or cached (patch w/w_sha256 fields).
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.evaluation.common import load_manifest, sha256, write_json

KEYS = ('cfa_mode', 'light_mode', 'model_name', 'resolution', 'num_classes',
        'noise_scale', 't_eps', 'lowlight_prob', 'lowlight_noise_std', 'num_layers')


def json_object(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError('Configuration must be an object')
    return obj


def read_config(path):
    obj = json_object(path)
    if 'training' in obj:
        obj = dict(obj['training'], cfa_mode=obj['cfa_mode'], light_mode=obj['light_mode'])
        # These are explicit production CLI defaults absent from launcher recipes.
        obj.setdefault('num_classes', 1000)
        obj.setdefault('t_eps', 1e-2)
    if any(k not in obj for k in KEYS):
        raise ValueError('Incomplete training config: ' + str([k for k in KEYS if k not in obj]))
    if obj['cfa_mode'] not in ('sparse', 'bayer') or obj['light_mode'] not in ('normal', 'mixed'):
        raise ValueError('Explicit CFA and normal/mixed training mode required')
    if obj['model_name'] != 'JiT-B/16' or obj['resolution'] != 256 or obj['num_classes'] != 1000 or obj['num_layers'] != 4:
        raise ValueError('Only current JiT-B/16 256px lightweight recipe supported')
    for key in ('noise_scale', 't_eps', 'lowlight_prob', 'lowlight_noise_std'):
        if type(obj[key]) not in (int, float) or not math.isfinite(obj[key]):
            raise ValueError('Invalid numeric training setting: ' + key)
    if obj['noise_scale'] <= 0 or not 0 < obj['t_eps'] < 1 or not 0 <= obj['lowlight_prob'] <= 1 or obj['lowlight_noise_std'] < 0:
        raise ValueError('Invalid diffusion/lowlight range')
    return obj


def checkpoint_info(checkpoint, config, provenance_config=None):
    path = Path(checkpoint).resolve()
    folder = path if path.is_dir() else path.parent
    if path.is_dir():
        candidates = [p for p in (path / 'model.safetensors', path / 'pytorch_model.bin') if p.is_file()]
        if len(candidates) != 1:
            raise ValueError('Checkpoint directory requires exactly one complete model file')
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    marker = folder / 'complete.json'
    sources = []
    if marker.is_file():
        metadata = json_object(marker)
        if metadata.get('checkpoint_schema') != 2:
            raise ValueError('Unsupported checkpoint metadata schema')
        source = metadata.get('training_config')
        if not isinstance(source, dict) or any(k not in source for k in KEYS):
            raise ValueError('Missing bound training config')
        files = metadata.get('files')
        if not isinstance(files, dict) or type(files.get(path.name)) is not int or files[path.name] <= 0 or path.stat().st_size != files[path.name]:
            raise ValueError('Incomplete checkpoint model file manifest')
        sources.append(source)
    else:
        if provenance_config is None:
            raise ValueError('Legacy weights require explicit provenance_config (source training configuration)')
        sources.append(read_config(provenance_config))
    for candidate in (folder / 'training_config.json', folder.parent / 'training_config.json'):
        if candidate.is_file():
            sources.append(read_config(candidate))
    if provenance_config is not None:
        sources.append(read_config(provenance_config))
    for source in sources:
        for key in KEYS:
            if source[key] != config[key]:
                raise ValueError('Checkpoint training config mismatch: ' + key)
    return path


def load_model(path, config, device):
    import torch
    from training.models.model_condition import create_condition_model
    from training.models.denoiser_demosaic import DemosaicDenoiser
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected complete tensor state_dict, not backbone/optimizer/wrapped checkpoint')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint parameters')
    model = create_condition_model(model_name=config['model_name'], img_size=256,
                                   num_classes=config['num_classes'], cfa_mode=config['cfa_mode'])
    # Accelerate stores the entire DemosaicDenoiser, including model. prefix.
    denoiser = DemosaicDenoiser(model, img_size=256, num_classes=config['num_classes'],
                              noise_scale=config['noise_scale'], t_eps=config['t_eps'])
    fixed = {k: v.detach().clone() for k, v in model.state_dict().items()
             if k.startswith('cfa_reconstructor.')}
    if all(k.startswith('model.') for k in state):
        denoiser.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(v, loaded[k]) for k, v in fixed.items()):
        raise ValueError('Checkpoint overwrites fixed CFA masks/kernels')
    for p in model.parameters():
        p.requires_grad = False
    model.unfreeze_pretrained_layers(num_layers=config['num_layers'])
    denoiser.to(device).eval()
    return denoiser


@contextmanager
def fixed_rng(seed, device='cpu'):
    """Restore caller RNG; use independent seeds for degradation and sampler."""
    import torch
    import numpy as np
    previous = np.random.get_state()
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            np.random.seed(seed % (2**32))
            yield
    finally:
        np.random.set_state(previous)


def sample_seed(seed, patch_id, stream):
    return int.from_bytes(hashlib.sha256(f'{seed}:{patch_id}:{stream}'.encode()).digest()[:8], 'big') % (2**63)


def make_condition(patch, config, light_level, seed, w_source=None):
    import torch
    import numpy as np
    from PIL import Image
    from training.data import SimulateLowLight
    from training.cfa import get_cfa_adapter
    with Image.open(patch['gt']) as im:
        rgb_arr = np.asarray(im).copy()
    rgb = torch.from_numpy(rgb_arr).permute(2, 0, 1).float() / 255
    adapter = get_cfa_adapter(config['cfa_mode'])
    simulator = SimulateLowLight(apply_prob=0, noise_std=config['lowlight_noise_std'])
    parameters = (light_level != 1.0, light_level)
    with fixed_rng(seed):
        if config['cfa_mode'] == 'bayer':
            degraded = simulator.apply_with_parameters(rgb, parameters, 'rgb')
            return adapter.make_conditions({'lgt_image': (degraded * 2 - 1).unsqueeze(0)})
        mask = adapter.pattern_mask().repeat(1, 64, 64)
        if w_source == 'cached':
            from training.evaluation.cached_cfa import validate_patch
            if not ('cached_cfa' in patch or 'cached_cfa_sha256' in patch):
                from training.evaluation.cached_cfa import validate_legacy_w
                validate_legacy_w(patch)
            cached = validate_patch(patch) if ('cached_cfa' in patch or 'cached_cfa_sha256' in patch) else None
            cached_rgb = (torch.from_numpy(np.asarray(cached['cfa']).copy()).permute(2, 0, 1).contiguous().float() / 255
                          if cached is not None else rgb * mask[:3])
            w_path = Path(patch.get('w', ''))
            if not w_path.is_absolute() or not w_path.is_file() or sha256(w_path) != patch.get('w_sha256'):
                raise ValueError('Cached W requires absolute w and verified w_sha256')
            with Image.open(w_path) as im:
                if im.mode != 'L' or im.size != (256, 256):
                    raise ValueError('Cached W must be aligned 256px L image')
                w = torch.from_numpy(np.asarray(im).copy()).unsqueeze(0).float() / 255
        elif w_source == 'lab-l':
            from skimage.color import rgb2lab
            # D65/2-degree sRGB LAB L; explicit quantization to emulate image cache.
            l = rgb2lab(rgb_arr, illuminant='D65', observer='2')[..., 0]
            w = torch.from_numpy((l * 255 / 100).astype(np.uint8).copy()).unsqueeze(0).float() / 255
            w = w * mask[3:4]
        else:
            raise ValueError('Sparse requires explicit cached or lab-l W source')
        # Match torchvision Normalize([0.5], [0.5]) operation order exactly.
        cfa = (simulator.apply_with_parameters(cached_rgb if w_source == 'cached' else rgb * mask[:3], parameters, 'rgb') - 0.5) / 0.5
        w = (simulator.apply_with_parameters(w, parameters, 'w') - 0.5) / 0.5
        return adapter.make_conditions({'cfa': cfa.unsqueeze(0), 'cfa_w': w.unsqueeze(0)})


def autocast(device, precision):
    import torch
    if precision == 'fp32':
        return nullcontext()
    return torch.autocast(torch.device(device).type, dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[precision])


def expected_projection_count(apply_constraint, steps, constraint_schedule):
    if not apply_constraint:
        return 0
    if constraint_schedule == 'final':
        return 1
    if constraint_schedule == 'per_step':
        return steps
    raise ValueError('constraint_schedule must be per_step or final')


def quantize_prediction(pred, mode='round'):
    """Convert a [-1, 1] NCHW RGB tensor to uint8 HWC. Official default is round."""
    import numpy as np
    x = pred[0].detach().float().cpu()
    if mode == 'round':
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    if mode == 'trunc':
        # Legacy conversion: normalize to [0, 1], scale, then truncate.
        img_np = ((x + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
        return (img_np * 255).astype(np.uint8)
    raise ValueError('quantize must be round or trunc')


def sample(denoiser, condition, *, seed, steps, method='euler', cfg_scale=0.0,
           apply_constraint=True, constraint_strength=1.0, precision='fp32',
           constraint_schedule='per_step'):
    import torch
    if constraint_schedule not in ('per_step', 'final'):
        raise ValueError('constraint_schedule must be per_step or final')
    device = condition.device
    labels = torch.full((1,), denoiser.num_classes, dtype=torch.long, device=device)
    generate_constraint = apply_constraint and constraint_schedule == 'per_step'
    with fixed_rng(seed, device), torch.inference_mode(), autocast(device, precision):
        pred = denoiser.generate(condition, labels, num_steps=steps, method=method,
                                 cfg_scale=cfg_scale, apply_constraint=generate_constraint,
                                 constraint_strength=constraint_strength)
        if apply_constraint and constraint_schedule == 'final':
            pred = denoiser._constraint_projection(pred, condition, strength=constraint_strength)
        return pred


def plan(*, manifest, checkpoint, training_config, output_dir, steps=30, light_level=1.0,
         seed=0, device='cpu', precision='fp32', provenance_config=None, cfa_mode=None,
         light_mode=None, w_source=None, method='euler', cfg_scale=0.0, constraint='auto',
         quantize='round', constraint_schedule='per_step'):
    config = read_config(training_config)
    if cfa_mode is not None and cfa_mode != config['cfa_mode'] or light_mode is not None and light_mode != config['light_mode']:
        raise ValueError('Requested CFA/training light mode mismatch')
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError('Positive steps and nonnegative integer seed required')
    if light_level not in (1.0, 1/30, 1/40, 1/50, 1/80, 1/100, 1/150):
        raise ValueError('Unsupported illumination')
    if config['light_mode'] == 'normal' and light_level != 1:
        raise ValueError('Normal-trained model is not authorized for lowlight evaluation')
    if precision not in ('fp32', 'bf16', 'fp16') or (device == 'cpu' and precision == 'fp16'):
        raise ValueError('Unsupported precision/device')
    if method not in ('euler', 'heun') or not math.isfinite(cfg_scale) or cfg_scale != 0 or constraint not in ('auto', 'on', 'off'):
        raise ValueError('Invalid sampler policy; neutral-label evaluation requires cfg_scale=0')
    if quantize not in ('round', 'trunc'):
        raise ValueError('quantize must be round or trunc')
    if constraint_schedule not in ('per_step', 'final'):
        raise ValueError('constraint_schedule must be per_step or final')
    if config['cfa_mode'] == 'sparse' and w_source not in ('cached', 'lab-l'):
        raise ValueError('Sparse requires explicit W source')
    patches = load_manifest(manifest)['patches']
    if any(p['size'] != [256, 256] for p in patches):
        raise ValueError('Only aligned 256px patches supported; no resizing')
    if config['cfa_mode'] == 'sparse' and w_source == 'cached':
        from training.evaluation.cached_cfa import validate_patch
        for patch in patches:
            if ('cached_cfa' in patch or 'cached_cfa_sha256' in patch):
                validate_patch(patch)
            else:
                from training.evaluation.cached_cfa import validate_legacy_w
                validate_legacy_w(patch)
    weights = checkpoint_info(checkpoint, config, provenance_config)
    normal = light_level == 1
    apply = normal or config['cfa_mode'] == 'sparse'
    strength = 1.0 if normal else 0.0
    if constraint != 'auto':
        apply, strength = constraint == 'on', float(constraint == 'on')
    settings = dict(training_config=config, training_config_sha256=sha256(training_config),
                    checkpoint=str(weights), checkpoint_sha256=sha256(weights),
                    provenance_config=str(Path(provenance_config).resolve()) if provenance_config else None,
                    manifest=str(Path(manifest).resolve()), manifest_sha256=sha256(manifest),
                    steps=steps, light_level=light_level, seed=seed, device=device, precision=precision,
                    method=method, cfg_scale=cfg_scale, constraint=constraint, apply_constraint=apply,
                    constraint_strength=strength, label=config['num_classes'], w_source=w_source,
                    w_warning='LAB-L generated W is not verified identical to training cached W' if w_source == 'lab-l' else None,
                    cfa_input_source='cached' if all('cached_cfa' in p for p in patches) else 'GT synthesized (legacy cached-W or lab-l)',
                    cached_w_warning='Legacy cached-W lacks source image_w alignment proof' if w_source == 'cached' and any('cached_cfa' not in p for p in patches) else None,
                    sampler_warning='Existing Heun endpoint and projection semantics preserved; Sparse debug checks synchronize and projection constructs/transfers masks each step',
                    quantize=quantize,
                    constraint_schedule=constraint_schedule,
                    expected_projection_count=expected_projection_count(apply, steps, constraint_schedule),
                    quantize_note='Round to nearest integer by default; trunc uses NumPy uint8 conversion after *255',
                    constraint_schedule_note='Official default per_step (model generate). final disables in-loop projection and applies DemosaicDenoiser._constraint_projection once after sampling; evaluation wrapper only, model algorithm unchanged')
    return patches, weights, settings


def run(*, execute=False, **kwargs):
    """Keyword API: plan's arguments plus execute=False. Returns prediction manifest."""
    patches, weights, settings = plan(**kwargs)
    output = Path(kwargs['output_dir']).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = dict(schema_version=1, config=settings,
                  config_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                  predictions=[], dry_run=not execute)
    if not execute:
        return result
    from training.evaluation.cached_cfa import validate_arrow_provenance
    validate_arrow_provenance(kwargs['manifest'])
    import numpy as np
    import torch
    from PIL import Image
    if settings['w_source'] == 'lab-l':
        import skimage
        settings['w_backend'] = dict(name='skimage.color.rgb2lab', version=skimage.__version__,
                                     illuminant='D65', observer='2', input='uint8 sRGB',
                                     output='L*/100 quantized uint8 then /255, CFA masked')
        result['config_sha256'] = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()
    denoiser = load_model(weights, settings['training_config'], settings['device'])
    output.mkdir(parents=True, exist_ok=False)
    for index, patch in enumerate(patches):
        condition = make_condition(patch, settings['training_config'], settings['light_level'],
                                   sample_seed(settings['seed'], patch['id'], 'degradation'), settings['w_source'])
        pred = sample(denoiser, condition.to(settings['device']),
                      seed=sample_seed(settings['seed'], patch['id'], 'initial-noise'),
                      **{k: settings[k] for k in ('steps', 'method', 'cfg_scale', 'apply_constraint',
                                                  'constraint_strength', 'precision', 'constraint_schedule')})
        if pred.shape != (1, 3, 256, 256) or not torch.isfinite(pred).all():
            raise ValueError('Invalid sampler output')
        pixels = quantize_prediction(pred, settings['quantize'])
        path = output / f'{index:06d}.png'
        with path.open('xb') as stream:
            Image.fromarray(np.asarray(pixels)).save(stream, format='PNG')
        result['predictions'].append(dict(id=patch['id'], path=str(path), gt_sha256=patch['sha256']))
    write_json(output / 'predictions.json', result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'checkpoint', 'training-config', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--provenance-config')
    p.add_argument('--cfa-mode', choices=('sparse', 'bayer'))
    p.add_argument('--light-mode', choices=('normal', 'mixed'))
    p.add_argument('--w-source', choices=('cached', 'lab-l'))
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--light-level', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='fp32')
    p.add_argument('--method', choices=('euler', 'heun'), default='euler')
    p.add_argument('--cfg-scale', type=float, default=0.0)
    p.add_argument('--constraint', choices=('auto', 'on', 'off'), default='auto')
    p.add_argument('--constraint-schedule', choices=('per_step', 'final'), default='per_step',
                   help='per_step projects during sampling; final projects once after sampling')
    p.add_argument('--quantize', choices=('round', 'trunc'), default='round',
                   help='round to nearest integer or truncate when converting to uint8')
    p.add_argument('--execute', action='store_true')
    return p


if __name__ == '__main__':
    args = vars(parser().parse_args())
    args['output_dir'] = args.pop('output')
    print(json.dumps(run(**args), indent=2, allow_nan=False))
