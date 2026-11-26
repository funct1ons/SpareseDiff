"""Real-result bar/line charts and explicit fixed-coordinate cross-method ROI grids.

python -m training.evaluation.make_figures chart --summary S --metric ssim --kind line --output F.png
python -m training.evaluation.make_figures roi --config R --output F.png
ROI schema_version=1: gt_manifest, methods=[{name,pred_manifest}],
rois=[{id,xy:[x,y],size:[w,h]}], zoom=2 (positive integer, nearest-neighbor).
Chart line order follows summary condition order; no inferred numeric illumination.
Infinity PSNR is rejected explicitly rather than clipped or silently omitted.
"""
import argparse
import io
from pathlib import Path
from .common import load_manifest, read_json, sha256, write_json
from .summarize_results import absolute, number, predictions


def save_bytes(output, data, provenance):
    path = Path(output)
    sidecar = Path(str(path) + '.json')
    if path.exists() or sidecar.exists() or path.is_symlink() or sidecar.is_symlink():
        raise FileExistsError(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    write_json(sidecar, provenance)


def chart(summary_path, output, metric='ssim', kind='bar', dataset=None, condition=None):
    # Revalidate source manifests and reports, not only the cached summary.
    from .summarize_results import validated_report, mean
    summary = read_json(summary_path)
    rows = summary.get('runs')
    if not isinstance(rows, list) or not rows or kind not in ('bar', 'line'):
        raise ValueError('Invalid chart input')
    for r in rows:
        for source in r['sources'].values():
            if sha256(source['path']) != source['sha256']:
                raise ValueError('Summary source hash mismatch')
        ground = load_manifest(r['sources']['gt_manifest']['path'])
        predictions(r['sources']['pred_manifest']['path'], ground)
        report, metrics = validated_report(r['sources']['metrics']['path'], ground)
        if metric not in metrics or r['datasets'] != report['datasets'] or r['overall'] != report['overall'] or r['dataset_mean'] != {m: mean([v[m] for v in report['datasets'].values()]) for m in metrics}:
            raise ValueError('Summary differs from source report')
    methods = list(dict.fromkeys(r['method'] for r in rows))
    conditions = list(dict.fromkeys(r['condition'] for r in rows))
    if len({(r['method'], r['condition']) for r in rows}) != len(rows):
        raise ValueError('Duplicate chart run')
    indexed = {(r['method'], r['condition']): r for r in rows}
    if set(indexed) != {(m, c) for m in methods for c in conditions}:
        raise ValueError('Incomplete chart grid')
    if condition is not None and condition not in conditions:
        raise ValueError('Unknown condition')
    conditions = [condition] if condition else conditions
    # Bar compares datasets for one condition; line compares conditions, optionally one dataset.
    if kind == 'bar' and len(conditions) != 1:
        raise ValueError('Bar requires explicit --condition when multiple conditions exist')
    labels = list(rows[0]['datasets']) if kind == 'bar' else conditions
    values = {}
    for method in methods:
        values[method] = []
        for label in labels:
            r = indexed[method, conditions[0] if kind == 'bar' else label]
            target = r['datasets'][label] if kind == 'bar' else (r['datasets'][dataset] if dataset else r['dataset_mean'])
            value = number(target.get(metric), metric)
            if value == float('inf'):
                raise ValueError('Infinite PSNR cannot be plotted; choose a finite metric (no clipping)')
            values[method].append(value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        for i, method in enumerate(methods):
            if kind == 'line':
                ax.plot(range(len(labels)), values[method], marker='o', label=method)
            else:
                width = .8 / len(methods)
                ax.bar([j - .4 + width*(i+.5) for j in range(len(labels))], values[method], width=width, label=method)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylabel(metric.upper())
        ax.set_title(dataset or ('Equal-weight dataset means' if kind == 'line' else conditions[0]))
        ax.legend()
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        save_bytes(output, buffer.getvalue(), dict(schema_version=1, kind=kind, metric=metric, labels=labels, values=values,
                   summary={'path': str(Path(summary_path).resolve()), 'sha256': sha256(summary_path)},
                   aggregation='equal-weight dataset means' if kind == 'line' and not dataset else 'arithmetic patch means'))
    finally:
        plt.close(fig)


def roi(config_path, output):
    from PIL import Image, ImageDraw
    config = read_json(config_path)
    ground = load_manifest(absolute(config['gt_manifest']))
    gt = {p['id']: p for p in ground['patches']}
    methods, names = [], {'GT'}
    specs = config.get('methods')
    if not isinstance(specs, list) or not specs:
        raise ValueError('methods must be nonempty')
    for spec in specs:
        name = spec.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Duplicate/invalid method name')
        names.add(name)
        methods.append((name, predictions(absolute(spec['pred_manifest']), ground)))
    rois = config.get('rois')
    zoom = config.get('zoom', 2)
    if not isinstance(rois, list) or not rois or type(zoom) is not int or zoom < 1:
        raise ValueError('Explicit ROIs/positive integer zoom required')
    seen = set()
    for r in rois:
        if not isinstance(r, dict) or not isinstance(r.get('id'), str) or r['id'] not in gt or r['id'] in seen:
            raise ValueError('Unknown/duplicate ROI ID')
        seen.add(r['id'])
        for key, minimum in (('xy', 0), ('size', 1)):
            v = r.get(key)
            if not isinstance(v, list) or len(v) != 2 or any(type(x) is not int or x < minimum for x in v):
                raise ValueError('Invalid ROI coordinates')
        x, y = r['xy']; w, h = r['size']
        if x+w > gt[r['id']]['size'][0] or y+h > gt[r['id']]['size'][1]:
            raise ValueError('ROI outside patch')
    cell_w = max(max(gt[r['id']]['size'][0], r['size'][0]*zoom) for r in rois)
    heights = [gt[r['id']]['size'][1] + r['size'][1]*zoom + 48 for r in rois]
    canvas = Image.new('RGB', (cell_w*(len(methods)+1), sum(heights)), 'white')
    draw = ImageDraw.Draw(canvas)
    top = 0
    for r, height in zip(rois, heights):
        p = gt[r['id']]
        x,y = r['xy']; w,h = r['size']
        for col, (name, path) in enumerate([('GT', p['gt'])] + [(name, pred[r['id']]['path']) for name, pred in methods]):
            with Image.open(path) as image:
                full = image.copy()
                crop = image.crop((x,y,x+w,y+h)).resize((w*zoom,h*zoom), Image.Resampling.NEAREST)
            ImageDraw.Draw(full).rectangle((x,y,x+w-1,y+h-1), outline='red', width=1)
            left = col*cell_w
            draw.text((left, top), name + ' | ' + r['id'], fill='black')
            canvas.paste(full, (left, top+24))
            canvas.paste(crop, (left, top+24+p['size'][1]))
        top += height
    buffer = io.BytesIO(); canvas.save(buffer, format='PNG')
    save_bytes(output, buffer.getvalue(), dict(schema_version=1, kind='roi', config=config,
               config_sha256=sha256(config_path), image_hashes={p: sha256(p) for p in [v['gt'] for v in gt.values()] + [v['path'] for _, pred in methods for v in pred.values()]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('chart')
    c.add_argument('--summary', required=True); c.add_argument('--output', required=True)
    c.add_argument('--metric', choices=['psnr','ssim','fsimc'], default='ssim')
    c.add_argument('--kind', choices=['bar','line'], default='bar')
    c.add_argument('--dataset'); c.add_argument('--condition')
    r = sub.add_parser('roi')
    r.add_argument('--config', required=True); r.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.command == 'chart':
        chart(a.summary, a.output, a.metric, a.kind, a.dataset, a.condition)
    else:
        roi(a.config, a.output)


if __name__ == '__main__':
    main()
