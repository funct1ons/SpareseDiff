"""Import real metric reports, never run or fabricate baseline methods.

CLI: python -m training.evaluation.summarize_results --config runs.json --output summary.json
Config schema_version=1, runs=[{method,condition,gt_manifest,pred_manifest,metrics}].
Paths in config are absolute. Each condition must have the same methods and GT.
Dataset means are patch-weighted; dataset_mean is equal-weight across datasets.
"""
import argparse
import math
import os
from pathlib import Path
from .common import image_size, load_manifest, read_json, sha256, write_json


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('Expected absolute path')
    return value


def predictions(path, ground):
    rows = read_json(path).get('predictions')
    if not isinstance(rows, list):
        raise ValueError('predictions must be a list')
    result, paths = {}, set()
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ('id', 'path', 'gt_sha256')):
            raise ValueError('Incomplete prediction')
        key = os.path.normcase(str(Path(absolute(row['path'])).resolve()))
        if row['id'] in result or key in paths:
            raise ValueError('Duplicate prediction ID/path')
        result[row['id']] = row
        paths.add(key)
    if set(result) != {p['id'] for p in ground['patches']}:
        raise ValueError('Prediction ID set differs from GT')
    for p in ground['patches']:
        r = result[p['id']]
        if r['gt_sha256'] != p['sha256'] or image_size(r['path']) != p['size']:
            raise ValueError('Prediction GT hash/size mismatch')
    return result


def number(value, metric):
    if value == 'Infinity' and metric == 'psnr':
        return math.inf
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Invalid/nonfinite metric')
    if metric in ('ssim', 'fsimc') and not -1 <= value <= 1:
        raise ValueError('Similarity outside [-1,1]')
    return value


def mean(values):
    return 'Infinity' if any(v == 'Infinity' for v in values) else math.fsum(values) / len(values)


def aggregate(rows, metrics):
    return dict(count=len(rows), **{m: mean([r[m] for r in rows]) for m in metrics})


def validated_report(path, ground):
    report = read_json(path)
    rows = report.get('patches')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing metric patches')
    expected = {p['id']: p for p in ground['patches']}
    seen = set()
    metrics = ['psnr', 'ssim'] + (['fsimc'] if 'fsimc' in rows[0] else [])
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get('id'), str) or r['id'] in seen or r['id'] not in expected:
            raise ValueError('Duplicate/unknown metric ID')
        seen.add(r['id'])
        if r.get('dataset') != expected[r['id']]['dataset'] or ('fsimc' in r) != ('fsimc' in metrics):
            raise ValueError('Metric dataset/coverage mismatch')
        for m in metrics:
            number(r.get(m), m)
    if seen != set(expected):
        raise ValueError('Missing metric IDs')
    datasets = {d: aggregate([r for r in rows if r['dataset'] == d], metrics) for d in sorted({p['dataset'] for p in ground['patches']})}
    for supplied, computed in ((report.get('datasets'), datasets), (report.get('overall'), aggregate(rows, metrics))):
        if not isinstance(supplied, dict) or supplied.keys() != computed.keys():
            raise ValueError('Metric summary keys differ')
        def check(a, b):
            for k, v in b.items():
                if isinstance(v, dict):
                    if not isinstance(a.get(k), dict) or a[k].keys() != v.keys():
                        raise ValueError('Dataset summary keys differ')
                    check(a[k], v)
                elif k == 'count':
                    if type(a.get(k)) is not int or a[k] != v:
                        raise ValueError('Summary count mismatch')
                elif v == 'Infinity':
                    if a.get(k) != v:
                        raise ValueError('Summary Infinity mismatch')
                elif not math.isclose(number(a.get(k), k), v, rel_tol=1e-9, abs_tol=1e-10):
                    raise ValueError('Summary mean mismatch')
        check(supplied, computed)
    return report, metrics


def run(config, output):
    if Path(output).exists():
        raise FileExistsError(output)
    specs = read_json(config).get('runs')
    if not isinstance(specs, list) or not specs:
        raise ValueError('runs must be nonempty')
    result, seen, signatures, groups = [], set(), {}, {}
    protocol = None
    metric_set = None
    for spec in specs:
        if not isinstance(spec, dict) or not all(isinstance(spec.get(k), str) and spec[k] for k in ('method', 'condition')):
            raise ValueError('Missing method/condition')
        method, condition = spec['method'], spec['condition']
        if (method, condition) in seen:
            raise ValueError('Duplicate method/condition')
        seen.add((method, condition))
        ground = load_manifest(absolute(spec['gt_manifest']))
        predictions(absolute(spec['pred_manifest']), ground)
        signature = sorted((p['id'], p['dataset'], p['sha256'], tuple(p['size']), p['gt']) for p in ground['patches'])
        if signatures and signature != next(iter(signatures.values())):
            raise ValueError('Runs do not share identical GT IDs/hash/size/path')
        signatures[condition] = signature
        report, metrics = validated_report(absolute(spec['metrics']), ground)
        if metric_set is not None and (metrics != metric_set or report.get('protocol') != protocol):
            raise ValueError('Incompatible metric coverage/protocol')
        metric_set, protocol = metrics, report.get('protocol')
        groups.setdefault(condition, set()).add(method)
        result.append(dict(method=method, condition=condition, datasets=report['datasets'], overall=report['overall'],
                           dataset_mean={m: mean([v[m] for v in report['datasets'].values()]) for m in metrics},
                           sources={k: {'path': spec[k], 'sha256': sha256(spec[k])} for k in ('gt_manifest', 'pred_manifest', 'metrics')}))
    if any(g != next(iter(groups.values())) for g in groups.values()):
        raise ValueError('Missing methods in condition grid')
    summary = dict(schema_version=1, protocol=protocol, metrics=metric_set, runs=result,
                   aggregation={'datasets': 'arithmetic patch mean', 'overall': 'arithmetic patch mean', 'dataset_mean': 'equal-weight arithmetic dataset mean'})
    write_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
