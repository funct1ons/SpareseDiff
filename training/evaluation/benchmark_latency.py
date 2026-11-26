"""Batch1 256px CUDA sampler latency (no I/O, loading or degradation timed)."""
import json
from pathlib import Path
import platform
import statistics
import sys
import time
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.evaluation import infer
from training.evaluation.common import write_json


def parameter_counts(denoiser):
    model = denoiser.model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = {}
    for name, p in model.named_parameters():
        group = 'encoder' if name.startswith('cond_encoder.') else 'adapters' if name.startswith('injection_layers.') else 'backbone' if name.startswith('jit_model.') else 'other'
        entry = groups.setdefault(group, dict(total=0, trainable=0))
        entry['total'] += p.numel()
        entry['trainable'] += p.numel() if p.requires_grad else 0
    return dict(total=total, trainable_training_recipe=trainable, frozen=total-trainable, groups=groups)


def benchmark(*, execute=False, warmup=3, repeats=20, **kwargs):
    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError('At least one warmup and two measured repeats required')
    if not str(kwargs.get('device', '')).startswith('cuda'):
        raise ValueError('Latency benchmark requires explicit CUDA device')
    patches, weights, settings = infer.plan(**kwargs)
    output = Path(kwargs['output_dir']).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = dict(schema_version=1, config=settings, dry_run=not execute, batch_size=1,
                  resolution=[256,256], warmup=warmup, repeats=repeats,
                  boundary='sample wrapper wall time incl RNG fork/reseed/restore, labels, Python dispatch, generate initial noise, condition encoding, denoising, debug synchronizations and projection mask transfers; excludes model loading, initial condition H2D, degradation, image I/O')
    if not execute:
        return result
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device(settings['device'])
    torch.cuda.set_device(device)
    denoiser = infer.load_model(weights, settings['training_config'], device)
    condition = infer.make_condition(patches[0], settings['training_config'], settings['light_level'],
                                     infer.sample_seed(settings['seed'], patches[0]['id'], 'degradation'), settings['w_source']).to(device)
    seed = infer.sample_seed(settings['seed'], patches[0]['id'], 'initial-noise')
    policy = {k: settings[k] for k in ('steps','method','cfg_scale','apply_constraint','constraint_strength','precision','constraint_schedule')}
    def generate():
        return infer.sample(denoiser, condition, seed=seed, **policy)
    for _ in range(warmup):
        generate()
    torch.cuda.synchronize(device)
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = generate()
        torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1000)
        if not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite benchmark prediction')
    ordered = sorted(durations)
    result.update(hardware=dict(gpu=torch.cuda.get_device_name(device),
                                capability=list(torch.cuda.get_device_capability(device)),
                                memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                                torch=torch.__version__, cuda=torch.version.cuda,
                                cudnn=torch.backends.cudnn.version(), platform=platform.platform(),
                                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                tf32_cudnn=torch.backends.cudnn.allow_tf32),
                  parameters=parameter_counts(denoiser),
                  latency_ms=dict(samples=durations, mean=statistics.mean(durations),
                                  median=statistics.median(durations), stdev=statistics.stdev(durations),
                                  minimum=min(durations), maximum=max(durations),
                                  p95=ordered[min(len(ordered)-1, int(0.95 * len(ordered)))]))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'latency.json', result)
    return result


if __name__ == '__main__':
    parser = infer.parser()
    parser.set_defaults(device='cuda')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    args = vars(parser.parse_args())
    args['output_dir'] = args.pop('output')
    print(json.dumps(benchmark(**args), indent=2, allow_nan=False))
