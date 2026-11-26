"""Explicit single-builder/single-version Arrow shard inventory."""
from pathlib import Path
import re
from .common import sha256


def inventory(root, expected_split):
    root = Path(root)
    if expected_split not in ('train', 'test', 'validation') or not root.is_absolute():
        raise ValueError('Explicit absolute builder/version path and expected split required')
    if 'downloads' in root.parts:
        raise ValueError('Downloads directory is not an Arrow builder cache')
    files = [root] if root.is_file() else sorted(root.glob('*.arrow'))
    if not files or any(p.parent != files[0].parent for p in files):
        raise ValueError('Select the exact single builder/version directory (not cache root)')
    pattern = re.compile(r'^(.*)-' + expected_split + r'(?:-(\d+)-of-(\d+))?\.arrow$')
    parsed = [pattern.fullmatch(p.name) for p in files]
    if any(m is None for m in parsed) or len({m[1] for m in parsed}) != 1:
        raise ValueError('Arrow expected split/builder mismatch')
    totals = {m[3] for m in parsed}
    if len(totals) != 1:
        raise ValueError('Inconsistent Arrow shard totals')
    total = next(iter(totals))
    if total is None:
        if len(files) != 1:
            raise ValueError('Multiple unsharded Arrow files')
    elif len(files) != int(total) or [int(m[2]) for m in parsed] != list(range(int(total))):
        raise ValueError('Missing/duplicate/noncontinuous Arrow shards')
    return [dict(path=str(p.resolve()), sha256=sha256(p), split=expected_split,
                 cache_version=str(p.parent.resolve())) for p in files]
