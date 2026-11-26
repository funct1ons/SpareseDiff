"""Strict, versioned evaluation manifests; paths are absolute and never resized."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    """Exclusive creation, including rejection of existing files/symlinks."""
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict) or type(obj.get('schema_version')) is not int or obj['schema_version'] != 1:
        raise ValueError('Expected schema_version=1 object')
    return obj


def image_size(path):
    from PIL import Image
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f'Expected existing absolute image path: {path}')
    with Image.open(path) as image:
        image.load()
        if image.mode != 'RGB':
            raise ValueError(f'Expected RGB image: {path} ({image.mode})')
        return list(image.size)


def load_manifest(path):
    obj = read_json(path)
    patches = obj.get('patches')
    if not isinstance(patches, list) or not patches:
        raise ValueError('patches must be a nonempty list')
    ids, paths = set(), set()
    for patch in patches:
        if not isinstance(patch, dict) or not {'id', 'dataset', 'gt', 'sha256', 'source', 'xy', 'size'} <= patch.keys():
            raise ValueError('Incomplete patch record')
        for key in ('id', 'dataset', 'gt', 'sha256', 'source'):
            if not isinstance(patch[key], str) or not patch[key]:
                raise ValueError(f'Invalid {key}')
        key = os.path.normcase(str(Path(patch['gt']).resolve()))
        if patch['id'] in ids or key in paths:
            raise ValueError('Duplicate patch ID or GT path')
        ids.add(patch['id']); paths.add(key)
        for field, minimum in (('xy', 0), ('size', 1)):
            value = patch[field]
            if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int or v < minimum for v in value):
                raise ValueError(f'Invalid {field}')
        if not Path(patch['source']).is_absolute():
            raise ValueError('source must be an absolute path')
        if image_size(patch['gt']) != patch['size']:
            raise ValueError('GT size does not match manifest')
        if sha256(patch['gt']) != patch['sha256']:
            raise ValueError('GT hash mismatch')
    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    args = parser.parse_args()
    print(f"Validated {len(load_manifest(args.manifest)['patches'])} patches")


if __name__ == '__main__':
    main()
