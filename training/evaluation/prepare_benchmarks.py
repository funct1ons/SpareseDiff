"""Prepare deterministic 256px non-overlapping RGB benchmark patches."""
import argparse
import hashlib
from pathlib import Path

from .common import image_size, sha256, write_json

DATASETS = ('CelebA', 'Kodak', 'McMaster', 'Urban100')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def prepare(roots, output, edge):
    """roots maps an explicit subset of the four datasets to absolute directories.

    edge='pad' uses black padding on right/bottom. No resizing or RGB conversion.
    A fresh output directory is required. Returns manifest.json's absolute path.
    """
    from PIL import Image
    if not roots or not set(roots) <= set(DATASETS) or edge not in ('drop', 'error', 'pad'):
        raise ValueError('Specify known dataset roots and edge=drop/error/pad')
    output = Path(output).resolve()
    plans = []
    for dataset in DATASETS:
        if dataset not in roots:
            continue
        root = Path(roots[dataset])
        if not root.is_absolute() or not root.is_dir():
            raise ValueError(f'Invalid absolute dataset root: {root}')
        files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f'No images in {dataset}')
        count = 0
        for source in files:
            w, h = image_size(source)
            if edge == 'error' and (w % 256 or h % 256):
                raise ValueError(f'Incomplete edge: {source}')
            stop_w, stop_h = (w, h) if edge == 'pad' else (w // 256 * 256, h // 256 * 256)
            for y in range(0, stop_h, 256):
                for x in range(0, stop_w, 256):
                    token = hashlib.sha256(source.relative_to(root).as_posix().encode()).hexdigest()
                    plans.append((dataset, source.resolve(), token, x, y))
                    count += 1
        if not count:
            raise ValueError(f'No complete patches for {dataset}; choose pad explicitly')
    output.mkdir(parents=True, exist_ok=False)
    patches = []
    source_hashes = {}
    for dataset, source, token, x, y in plans:
        identifier = f'{dataset}/{token}_{x}_{y}'
        target = output / (identifier + '.png')
        target.parent.mkdir(exist_ok=True)
        with Image.open(source) as image:
            image.crop((x, y, x + 256, y + 256)).save(target)
        if source not in source_hashes:
            source_hashes[source] = sha256(source)
        patches.append(dict(id=identifier, dataset=dataset, gt=str(target), sha256=sha256(target),
                            source=str(source), source_sha256=source_hashes[source], xy=[x, y], size=[256, 256]))
    manifest = output / 'manifest.json'
    write_json(manifest, dict(schema_version=1, edge=edge, padding='black' if edge == 'pad' else None,
                             patch_size=256, roots={k: str(v) for k, v in roots.items()}, patches=patches))
    return manifest


def prepare_arrow(roots, output, ranges, expected_splits=None):
    """Explicit per-dataset [start,stop); six columns share nonoverlapping crops."""
    from .cached_cfa import arrow_rows, COLUMNS
    import importlib.metadata
    import re
    if not roots or not set(roots) <= set(DATASETS) or set(ranges) != set(roots):
        raise ValueError('Explicit dataset roots and sample ranges required')
    from .arrow_inventory import inventory
    if not isinstance(expected_splits, dict) or set(expected_splits) != set(roots):
        raise ValueError('Explicit expected_splits per dataset required')
    arrow_files = [row for dataset, root in roots.items() for row in inventory(root, expected_splits[dataset])]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    # Validate the selected rows before publishing anything.
    for dataset, root in roots.items():
        for index, source, images in arrow_rows(root, *ranges[dataset]):
            pass
    def selected_rows():
        for dataset, root in roots.items():
            for index, source, images in arrow_rows(root, *ranges[dataset]):
                yield dataset, index, source, images
    output.mkdir(parents=True, exist_ok=False)
    patches, sources, source_hashes = [], [], {}
    for dataset, index, source, images in selected_rows():
        if source not in source_hashes:
            source_hashes[source] = sha256(source)
        width, height = images['image'].size
        sources.append(dict(dataset=dataset, row=index, local_row=images['_local_row'], arrow=str(source), arrow_sha256=source_hashes[source],
                            split=re.sub(r'-\d+-of-\d+$', '', source.stem).rsplit('-',1)[-1],
                            cache_version=str(source.parent),
                            size=[width, height], dropped_right=width % 256,
                            dropped_bottom=height % 256))
        for y in range(0, height // 256 * 256, 256):
            for x in range(0, width // 256 * 256, 256):
                identifier = f'{dataset}/{index:06d}_{x}_{y}'
                record = dict(id=identifier, dataset=dataset, source=str(source),
                              source_sha256=source_hashes[source], source_row=index,
                              source_local_row=images['_local_row'], xy=[x,y], size=[256,256])
                for column in COLUMNS:
                    target = output / (identifier + '_' + column + '.png')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    images[column].crop((x,y,x+256,y+256)).save(target)
                    field = {'image':'gt', 'cfa':'cached_cfa', 'cfa_w':'w'}.get(column, column)
                    record[field] = str(target)
                    record['sha256' if field == 'gt' else field + '_sha256'] = sha256(target)
                patches.append(record)
    if not patches:
        raise ValueError('No complete 256px patches in selected rows')
    manifest = output / 'manifest.json'
    write_json(manifest, dict(schema_version=1, edge='drop', patch_size=256,
                             cache_columns=list(COLUMNS), sample_ranges=ranges,
                             expected_splits=expected_splits, arrow_files=arrow_files,
                             arrow_roots={d:str(Path(p).resolve()) for d,p in roots.items()},
                             versions={name: importlib.metadata.version(name) for name in ('pyarrow', 'Pillow')},
                             selected_files=[dict(path=str(p), sha256=h) for p,h in source_hashes.items()],
                             sources=sources, patches=patches))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in DATASETS:
        parser.add_argument('--' + dataset.lower() + '-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--edge', choices=('drop', 'error', 'pad'), required=True)
    parser.add_argument('--arrow', action='store_true')
    parser.add_argument('--expected-splits', help='JSON object: dataset -> train/test/validation')
    parser.add_argument('--sample-ranges', help='JSON object: dataset -> [start,stop)')
    args = parser.parse_args()
    if args.arrow:
        import json
        if not args.sample_ranges or not args.expected_splits or args.edge != 'drop':
            parser.error('Arrow requires --sample-ranges and --edge drop')
        print(prepare_arrow({d: getattr(args, d.lower() + '_root') for d in DATASETS if getattr(args, d.lower() + '_root')}, args.output, json.loads(args.sample_ranges), json.loads(args.expected_splits)))
        return
    print(prepare({d: getattr(args, d.lower() + '_root') for d in DATASETS if getattr(args, d.lower() + '_root')}, args.output, args.edge))


if __name__ == '__main__':
    main()
