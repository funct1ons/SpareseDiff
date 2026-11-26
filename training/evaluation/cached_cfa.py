"""Offline Arrow decoding and strict Sparse cache validation (no datasets/hub)."""
import io
from pathlib import Path
from .common import sha256

COLUMNS = ('image', 'cfa', 'cfa_inter', 'image_w', 'cfa_w', 'cfa_inter_w')


def masks(height, width):
    import numpy as np
    mask = np.zeros((height, width, 3), dtype=bool)
    mask[1::4, 0::4, 0] = True
    mask[0::4, 0::4, 1] = True
    mask[1::4, 1::4, 1] = True
    mask[0::4, 1::4, 2] = True
    return mask, ~mask.any(axis=2)


def validate_images(images):
    import numpy as np
    size = images['image'].size
    for column in COLUMNS:
        image = images[column]
        if image.mode != ('RGB' if column in COLUMNS[:3] else 'L') or image.size != size:
            raise ValueError('Cache mode/size mismatch: ' + column)
    rgb, wm = masks(size[1], size[0])
    if not np.array_equal(np.asarray(images['cfa']), np.asarray(images['image']) * rgb):
        raise ValueError('Cached CFA mask/GT alignment mismatch')
    if not np.array_equal(np.asarray(images['cfa_w']), np.asarray(images['image_w']) * wm):
        raise ValueError('Cached W mask/image_w alignment mismatch')


def arrow_rows(root, start, stop):
    import pyarrow as pa
    from PIL import Image
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError('Explicit nonempty [start, stop) sample range required')
    root = Path(root)
    if not root.is_absolute() or not root.exists():
        raise ValueError('Absolute existing Arrow root required')
    files = [root] if root.is_file() else sorted(root.rglob('*.arrow'))
    if not files:
        raise ValueError('No Arrow IPC files')
    import re
    splits = {re.sub(r'-\d+-of-\d+$', '', p.stem).rsplit('-', 1)[-1] for p in files}
    if len({p.parent for p in files}) != 1 or len(splits) != 1:
        raise ValueError('Arrow root mixes cache versions or splits; select one version/split directory or file')
    index = 0
    for path in files:
        shard_start = index
        with pa.memory_map(str(path), 'r') as source:
            try:
                reader = pa.ipc.open_stream(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_file(source)
            batches = reader if hasattr(reader, '__iter__') else (reader.get_batch(i) for i in range(reader.num_record_batches))
            for batch in batches:
                if not set(COLUMNS) <= set(batch.schema.names):
                    raise ValueError('Missing six cache columns')
                for row_index in range(batch.num_rows):
                    if start <= index < stop:
                        row = batch.slice(row_index, 1).to_pylist()[0]
                        images = {}
                        for column in COLUMNS:
                            value = row[column]
                            if not isinstance(value, dict) or not value.get('bytes'):
                                raise ValueError('Embedded HF image bytes required: ' + column)
                            with Image.open(io.BytesIO(value['bytes'])) as image:
                                image.load()
                                images[column] = image.copy()
                        validate_images(images)
                        images['_local_row'] = index - shard_start
                        yield index, path, images
                    index += 1
                    if index == stop:
                        return
    raise ValueError('Sample range exceeds Arrow rows')


def validate_patch(patch):
    from PIL import Image
    images = {}
    for column, field in (('image', 'gt'), ('cfa', 'cached_cfa'), ('cfa_w', 'w'), ('image_w', 'image_w'), ('cfa_inter', 'cfa_inter'), ('cfa_inter_w', 'cfa_inter_w')):
        path = Path(patch.get(field, ''))
        digest = patch.get('sha256' if field == 'gt' else field + '_sha256')
        if not path.is_absolute() or not path.is_file() or sha256(path) != digest:
            raise ValueError('Cached W requires absolute path and verified hash (including cached CFA/GT/image_w): ' + field)
        with Image.open(path) as image:
            image.load()
            images[column] = image.copy()
    validate_images(images)
    if images['image'].size != (256, 256):
        raise ValueError('Cache requires aligned 256px patches')
    return images


def validate_legacy_w(patch):
    """Compatibility only: verify file and mask, without missing source alignment."""
    import numpy as np
    from PIL import Image
    path = Path(patch.get('w', ''))
    if not path.is_absolute() or not path.is_file() or sha256(path) != patch.get('w_sha256'):
        raise ValueError('Cached W requires absolute path and verified hash')
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != (256, 256):
            raise ValueError('Cached W must be aligned 256px L image')
        _, wm = masks(256, 256)
        if np.any(np.asarray(image)[~wm]):
            raise ValueError('Cached W off-mask values')


def validate_arrow_provenance(manifest):
    """Audit strict six-column Arrow selection once, before production side effects.

    RGB-only and legacy cached-W manifests remain supported with explicit
    inference warnings; declared Arrow manifests require the strict six columns.
    """
    from .common import load_manifest, read_json
    from .arrow_inventory import inventory
    import numpy as np
    obj = read_json(manifest)
    if 'cache_columns' not in obj:
        if 'arrow_roots' in obj or any('source_row' in p for p in obj.get('patches', [])):
            raise ValueError('Arrow manifest requires strict six-column schema')
        return
    obj = load_manifest(manifest)
    roots = obj.get('arrow_roots')
    ranges = obj.get('sample_ranges')
    splits = obj.get('expected_splits')
    if (obj['cache_columns'] != list(COLUMNS) or obj.get('edge') != 'drop'
            or obj.get('patch_size') != 256 or not isinstance(roots, dict) or not roots
            or not isinstance(ranges, dict) or set(ranges) != set(roots)
            or not isinstance(splits, dict) or set(splits) != set(roots)):
        raise ValueError('Strict Arrow roots/ranges/splits/drop256 schema required')
    inventories = {}
    for dataset, root in roots.items():
        # A sharded file is not a complete inventory, even if its neighbours exist.
        import re
        if Path(root).is_file() and re.search(r'-\d+-of-\d+\.arrow$', str(root)):
            raise ValueError('Sharded file entry rejected; select exact version directory')
        inventories[dataset] = inventory(root, splits[dataset])
        span = ranges[dataset]
        if (not isinstance(span, list) or len(span) != 2
                or any(type(v) is not int for v in span) or not 0 <= span[0] < span[1]):
            raise ValueError('Explicit nonempty [start,stop) sample range required')
    if [f for files in inventories.values() for f in files] != obj.get('arrow_files'):
        raise ValueError('Arrow ordered inventory/split/version/hash mismatch')
    grouped = {}
    for patch in obj['patches']:
        dataset, row = patch['dataset'], patch.get('source_row')
        if dataset not in roots or type(row) is not int or not ranges[dataset][0] <= row < ranges[dataset][1]:
            raise ValueError('Arrow dataset/range binding mismatch')
        grouped.setdefault((dataset, row), []).append(patch)
    sources = []
    selected = {}
    for dataset, root in roots.items():
        hashes = {f['path']: f['sha256'] for f in inventories[dataset]}
        # Ordered shard traversal counts every preceding row, binding global/local.
        for row, source, originals in arrow_rows(root, *ranges[dataset]):
            source = str(source.resolve())
            local = originals['_local_row']
            width, height = originals['image'].size
            sources.append(dict(dataset=dataset, row=row, local_row=local, arrow=source,
                                arrow_sha256=hashes[source], split=splits[dataset],
                                cache_version=str(Path(source).parent), size=[width, height],
                                dropped_right=width % 256, dropped_bottom=height % 256))
            selected[source] = hashes[source]
            coordinates = {(x, y) for y in range(0, height // 256 * 256, 256)
                           for x in range(0, width // 256 * 256, 256)}
            seen = set()
            for patch in grouped.pop((dataset, row), []):
                if (patch['source'] != source or patch.get('source_sha256') != hashes[source]
                        or type(patch.get('source_local_row')) is not int
                        or patch['source_local_row'] != local):
                    raise ValueError('Arrow source hash/global-local/dataset mismatch')
                xy = tuple(patch['xy'])
                if xy not in coordinates or xy in seen or patch['size'] != [256, 256]:
                    raise ValueError('Arrow duplicate/extra/unaligned crop')
                seen.add(xy)
                images = validate_patch(patch)
                x, y = xy
                for column in COLUMNS:
                    crop = originals[column].crop((x, y, x + 256, y + 256))
                    if not np.array_equal(np.asarray(crop), np.asarray(images[column])):
                        raise ValueError('Arrow source row/crop mismatch: ' + column)
            if seen != coordinates:
                raise ValueError('Arrow selected row missing drop256 patches')
    if grouped or sources != obj.get('sources'):
        raise ValueError('Arrow complete selected source rows mismatch')
    if [dict(path=p, sha256=h) for p, h in selected.items()] != obj.get('selected_files'):
        raise ValueError('Arrow selected files mismatch')
