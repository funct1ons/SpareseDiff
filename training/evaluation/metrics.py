"""Strict RGB PSNR/SSIM evaluation; optional standard phase-congruency FSIMc."""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path

from .common import image_size, load_manifest, read_json, write_json

OFFICIAL_PSNR = 'RGB uint8 images cast to float64 before MSE; 10*log10(255^2/mse)'
UINT8_OVERFLOW_PSNR = (
    'diagnostic only, never official; numpy uint8 wraparound subtract-and-square, '
    'then mean as float; reproduces a legacy uint8 overflow in PSNR'
)


def psnr_from_mse(mse, max_val=255.0):
    if mse == 0:
        return 'Infinity'
    return 10 * math.log10((max_val ** 2) / mse)


def mse_float64(first, second):
    import numpy as np
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f'Image shapes differ: {a.shape} != {b.shape}')
    return float(np.mean((a - b) ** 2))


def mse_uint8_overflow(first, second):
    """Wraparound uint8 MSE used only to diagnose the old assessor formula."""
    import numpy as np
    a = np.asarray(first)
    b = np.asarray(second)
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('uint8 overflow diagnostic requires uint8 arrays')
    if a.shape != b.shape:
        raise ValueError(f'Image shapes differ: {a.shape} != {b.shape}')
    return float(np.mean((a - b) ** 2))


def run(gt_manifest, pred_manifest, output, fsimc=False, diagnostic_uint8_psnr=False):
    import numpy as np
    from PIL import Image
    if Path(output).exists():
        raise FileExistsError(output)
    ground = load_manifest(gt_manifest)
    predicted = read_json(pred_manifest).get('predictions')
    if not isinstance(predicted, list):
        raise ValueError('predictions must be a list')
    predictions, paths = {}, set()
    for row in predicted:
        if not isinstance(row, dict) or not {'id', 'path', 'gt_sha256'} <= row.keys():
            raise ValueError('Incomplete prediction record')
        if not all(isinstance(row[k], str) and row[k] for k in ('id', 'path', 'gt_sha256')):
            raise ValueError('Invalid prediction fields')
        path = os.path.normcase(str(Path(row['path']).resolve()))
        if row['id'] in predictions or path in paths:
            raise ValueError('Duplicate prediction ID or path')
        predictions[row['id']] = row; paths.add(path)
    if set(predictions) != {p['id'] for p in ground['patches']}:
        raise ValueError('Prediction and GT ID sets differ')
    for patch in ground['patches']:
        row = predictions[patch['id']]
        if row['gt_sha256'] != patch['sha256'] or image_size(row['path']) != patch['size']:
            raise ValueError('Prediction GT hash or size mismatch')
    versions = {}
    for package in ('numpy', 'Pillow', 'torch', 'piq'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if fsimc:
        try:
            import torch
            import piq
        except Exception as exc:
            raise RuntimeError(f'FSIMc requires working torch/piq (no approximation); versions={versions}') from exc
    coords = np.arange(-5, 6)
    window = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * 1.5 ** 2))
    window /= window.sum()

    def ssim(a, b):
        scores = []
        for channel in range(3):
            x, y = a[:, :, channel] / 255, b[:, :, channel] / 255
            def smooth(z):
                # Gaussian is separable: the same 11x11 reference window,
                # implemented in NumPy without an additional SciPy dependency.
                kernel = window.sum(axis=0)
                padded = np.pad(z, 5, mode='reflect')
                horizontal = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), 1, padded)
                return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), 0, horizontal)
            mx, my = smooth(x), smooth(y)
            vx, vy, cov = smooth(x*x)-mx*mx, smooth(y*y)-my*my, smooth(x*y)-mx*my
            scores.append(float(np.mean((2*mx*my+0.01**2)*(2*cov+0.03**2) /
                                         ((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2)))))
        return float(np.mean(scores))

    rows = []
    for patch in ground['patches']:
        with Image.open(patch['gt']) as image:
            a_u8 = np.asarray(image)
            a = np.asarray(image, dtype=np.float64)
        with Image.open(predictions[patch['id']]['path']) as image:
            b_u8 = np.asarray(image)
            b = np.asarray(image, dtype=np.float64)
        mse = mse_float64(a, b)
        row = dict(id=patch['id'], dataset=patch['dataset'], psnr=psnr_from_mse(mse), ssim=ssim(a, b))
        if diagnostic_uint8_psnr:
            if a_u8.dtype != np.uint8 or b_u8.dtype != np.uint8:
                raise ValueError('uint8 overflow diagnostic requires uint8 PNG pixels')
            row['psnr_uint8_overflow'] = psnr_from_mse(mse_uint8_overflow(a_u8, b_u8))
        if fsimc:
            x = torch.from_numpy(a.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255
            y = torch.from_numpy(b.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255
            with torch.no_grad():
                value = float(piq.fsim(x, y, data_range=1., chromatic=True).item())
            if not math.isfinite(value):
                raise ValueError('Nonfinite FSIMc')
            row['fsimc'] = value
        rows.append(row)

    metric_names = ('psnr', 'ssim') + (('fsimc',) if fsimc else ()) + (
        ('psnr_uint8_overflow',) if diagnostic_uint8_psnr else ())

    def summarize(items):
        result = {'count': len(items)}
        for metric in metric_names:
            values = [item[metric] for item in items]
            result[metric] = 'Infinity' if 'Infinity' in values else float(np.mean(values))
        return result
    protocol = {'color': 'RGB', 'data_range': 255,
                'psnr': OFFICIAL_PSNR, 'psnr_role': 'official_float64',
                'ssim': '11x11 Gaussian sigma=1.5, population covariance, reflect padding, channel mean',
                'fsimc': 'piq.fsim chromatic=True standard phase congruency' if fsimc else None,
                'aggregation': 'arithmetic patch mean; infinite PSNR encoded as string Infinity',
                'psnr_uint8_overflow': UINT8_OVERFLOW_PSNR if diagnostic_uint8_psnr else None}
    result = dict(schema_version=1, versions=versions, protocol=protocol,
                  patches=rows, datasets={d: summarize([r for r in rows if r['dataset'] == d]) for d in sorted({r['dataset'] for r in rows})},
                  overall=summarize(rows))
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-manifest', required=True)
    parser.add_argument('--pred-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fsimc', action='store_true')
    parser.add_argument('--diagnostic-uint8-psnr', action='store_true',
                        help='Also record wraparound uint8 PSNR; official psnr stays float64')
    args = parser.parse_args()
    run(args.gt_manifest, args.pred_manifest, args.output, args.fsimc, args.diagnostic_uint8_psnr)


if __name__ == '__main__':
    main()
