"""Build an explicit paper evaluation matrix; execution is opt-in and serial."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

DATASETS = ('CelebA', 'Kodak', 'McMaster', 'Urban100')
LEVELS = (30, 40, 50, 80, 100, 150)
STEPS = (1, 2, 3, 4, 5, 10, 15, 30)
GROUPS = ('sparse_normal', 'sparse_mixed', 'bayer_normal', 'bayer_mixed')


def build_plan(spec, output, seed=42):
    """Spec binds four checkpoint/config pairs and four benchmark manifests."""
    if set(spec['models']) != set(GROUPS):
        raise ValueError('models must explicitly bind all four CFA/light groups')
    if set(spec['benchmarks']) != set(DATASETS):
        raise ValueError('benchmarks must bind exactly the four named datasets')
    jobs = []
    output = Path(output).resolve()
    for group in GROUPS:
        model = spec['models'][group]
        if not {'checkpoint', 'training_config'} <= set(model) or set(model) - {'checkpoint', 'training_config', 'provenance_config', 'w_source'}:
            raise ValueError('each model requires checkpoint and training_config')
        cfa, light = group.split('_')
        if cfa == 'sparse' and model.get('w_source') not in ('cached', 'lab-l'):
            raise ValueError('Sparse models require explicit w_source cached or lab-l')
        if 'w_source' in model and model['w_source'] not in ('cached', 'lab-l'):
            raise ValueError('Invalid w_source')
        scenarios = [(dataset, 1.0, 30, 'normal') for dataset in DATASETS]
        if light == 'mixed':
            scenarios += [(dataset, 1 / level, 30, 'lowlight')
                          for level in LEVELS for dataset in DATASETS]
        scenarios += [('Kodak', level, steps, 'steps')
                      for level in ((1.0,) if light == 'normal' else
                                    (1.0, 1/30, 1/80, 1/100, 1/150))
                      for steps in STEPS]
        for dataset, level, steps, study in scenarios:
            denominator = 1 if level == 1 else round(1 / level)
            job_id = f'{group}_{study}_{dataset}_1-{denominator}_s{steps}'
            jobs.append(dict(id=job_id, group=group, cfa_mode=cfa,
                             light_mode=light, dataset=dataset,
                             light_level=level, steps=steps, seed=seed,
                             checkpoint=str(Path(model['checkpoint']).resolve()),
                             training_config=str(Path(model['training_config']).resolve()),
                             manifest=str(Path(spec['benchmarks'][dataset]).resolve()),
                             output=str(output / job_id),
                             **({'provenance_config': str(Path(model['provenance_config']).resolve())} if 'provenance_config' in model else {}),
                             **({'w_source': model['w_source']} if 'w_source' in model else {})))
    return dict(schema_version=1, status='planned', seed=seed, jobs=jobs)


def preflight(plan, fsimc=False):
    """Read-only validation of the entire matrix before any output is created."""
    from . import infer
    # Reuse production validation, including checkpoint directories/metadata.
    for job in plan['jobs']:
        for key in ('checkpoint', 'training_config', 'manifest', 'provenance_config'):
            if key in job and not Path(job[key]).exists():
                raise FileNotFoundError(job[key])
    from .cached_cfa import validate_arrow_provenance
    for manifest in dict.fromkeys(str(Path(job['manifest']).resolve()) for job in plan['jobs']):
        validate_arrow_provenance(manifest)
    checked_w = set()
    for job in plan['jobs']:
        patches, weights, settings = infer.plan(**infer_kwargs(job))
        cfa, light = job['group'].split('_')
        if (settings['training_config']['cfa_mode'], settings['training_config']['light_mode']) != (cfa, light):
            raise ValueError('Training CFA/light mismatch')
        if (job['cfa_mode'], job['light_mode']) != (cfa, light):
            raise ValueError('Job CFA/light differs from group')
        datasets = {p['dataset'] for p in patches}
        if datasets != {job['dataset']}:
            raise ValueError(f"Manifest dataset binding mismatch: {job['dataset']}: {datasets}")
        if cfa == 'sparse' and job.get('w_source') == 'cached':
            for patch in patches:
                key = (job['training_config'], patch['id'], patch.get('w'), patch.get('w_sha256'))
                if key not in checked_w:
                    # plan currently does not validate W; use its production consumer,
                    # not a second implementation of hash/image/shape validation.
                    infer.make_condition(patch, settings['training_config'], 1.0,
                                         infer.sample_seed(job['seed'], patch['id'], 'degradation'), 'cached')
                    checked_w.add(key)
    if fsimc:
        try:
            import torch
            import piq
            if not callable(piq.fsim):
                raise TypeError('piq.fsim is not callable')
        except Exception as exc:
            raise RuntimeError('FSIMc requires working torch/piq; no approximation') from exc


def infer_kwargs(job):
    return dict(manifest=job['manifest'], checkpoint=job['checkpoint'],
                training_config=job['training_config'], output_dir=job['output'],
                steps=job['steps'], light_level=job['light_level'], seed=job['seed'],
                cfa_mode=job['cfa_mode'], light_mode=job['light_mode'],
                **{k: job[k] for k in ('provenance_config', 'w_source') if k in job})


def infer_command(job, device):
    command = [sys.executable, '-m', 'training.evaluation.infer']
    for key, value in infer_kwargs(job).items():
        flag = 'output' if key == 'output_dir' else key.replace('_', '-')
        command.extend(['--' + flag, str(value)])
    return command + ['--device', device, '--execute']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fsimc', action='store_true')
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error('--execute and --dry-run are mutually exclusive')
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    plan = build_plan(spec, args.output, args.seed)
    plan['requested_metrics'] = ['psnr', 'ssim'] + (['fsimc'] if args.fsimc else [])
    plan['computed_metrics'] = []
    plan['preflight'] = 'not_run'
    if not args.execute:
        plan['status'] = 'planned_not_executed'
        print(json.dumps(plan, indent=2))
        return
    from .common import write_json
    if args.output.exists():
        raise FileExistsError(args.output)
    preflight(plan, args.fsimc)
    plan['preflight'] = 'passed'
    args.output.mkdir(parents=True)
    write_json(args.output / 'plan.json', plan)
    completed = []
    computed = {}
    try:
        for job in plan['jobs']:
            command = infer_command(job, args.device)
            subprocess.run(command, check=True)
            command = [sys.executable, '-m', 'training.evaluation.metrics',
                       '--gt-manifest', job['manifest'],
                       '--pred-manifest', str(Path(job['output']) / 'predictions.json'),
                       '--output', str(Path(job['output']) / 'metrics.json')]
            if args.fsimc:
                command.append('--fsimc')
            subprocess.run(command, check=True)
            result = json.loads((Path(job['output']) / 'metrics.json').read_text(encoding='utf-8'))
            rows = result.get('patches', [])
            actual = [m for m in ('psnr', 'ssim', 'fsimc')
                      if rows and all(m in row for row in rows)]
            computed[job['id']] = actual
            if not set(plan['requested_metrics']) <= set(actual):
                raise ValueError(f"Requested metrics missing: {job['id']}")
            completed.append(job['id'])
    finally:
        write_json(args.output / 'results_manifest.json',
                   dict(schema_version=1,
                        status='execution_complete' if len(completed) == len(plan['jobs']) else 'incomplete',
                        execution_complete=len(completed) == len(plan['jobs']),
                        paper_metrics_complete=(len(completed) == len(plan['jobs']) and
                                                all(set(('psnr', 'ssim', 'fsimc')) <= set(computed[j['id']])
                                                    for j in plan['jobs'])),
                        requested_metrics=plan['requested_metrics'], computed_metrics=computed,
                        completed=completed, plan=plan))


if __name__ == '__main__':
    main()
