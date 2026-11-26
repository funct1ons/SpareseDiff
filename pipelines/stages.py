"""Single-patch RGB conditioning and lossless stage artifacts."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, __version__ as pillow_version
import torch

from training.cfa import get_cfa_adapter
from training.data import SimulateLowLight
from training.evaluation.common import read_json, sha256, write_json
from training.evaluation.infer import fixed_rng, sample_seed

SIZE = 256
LEVELS = (1.0, 1/30, 1/40, 1/50, 1/80, 1/100, 1/150)
MODES = ('bayer', 'sparse')
PATCH_ID = 'rgb-patch'
SAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'samples'


def read_rgb(source):
    """Decode and orient as in the demo, interpreting RGB values as sRGB."""
    with Image.open(source) as decoded:
        image = ImageOps.exif_transpose(decoded)
        if image.mode != 'RGB':
            raise ValueError('Expected an 8-bit RGB image; convert grayscale/RGBA to RGB first')
        return image


def read_input(path, crop=None):
    image = read_rgb(path)
    if crop is not None:
        x, y = crop
        if min(x, y) < 0 or x + SIZE > image.width or y + SIZE > image.height:
            raise ValueError('256x256 crop is outside the input image')
        image = image.crop((x, y, x + SIZE, y + SIZE))
    if image.size != (SIZE, SIZE):
        raise ValueError('Expected 256x256 RGB; use --crop X Y for a larger image')
    return Image.frombytes('RGB', image.size, image.tobytes())


@lru_cache(maxsize=1)
def _lab_transform():
    return ImageCms.buildTransform(ImageCms.createProfile('sRGB'),
                                   ImageCms.createProfile('LAB'), 'RGB', 'LAB')


def lab_w(rgb_image):
    """8-bit Lab L; the Pillow/LittleCMS path matched the sampled training cache."""
    return ImageCms.applyTransform(rgb_image, _lab_transform()).getchannel('L')


def tensor(image):
    arr = np.array(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255


def grid(cfa):
    adapter = get_cfa_adapter(cfa)
    n = SIZE // adapter.tile_size
    return adapter.pattern_mask().bool().repeat(1, n, n)


def simulate(rgb_image, cfa, level, seed, noise_std, *, patch_id=PATCH_ID):
    """Bayer returns dense degraded RGB. Sparse returns degraded sampled RGBW."""
    rgb = tensor(rgb_image)
    mask = grid(cfa)
    w = lab_w(rgb_image) if cfa == 'sparse' else None
    low_light = SimulateLowLight(apply_prob=0, noise_std=noise_std)
    with fixed_rng(sample_seed(seed, patch_id, 'degradation')):
        params = (level != 1.0, level)
        if w is None:
            return low_light.apply_with_parameters(rgb, params, 'rgb'), None
        # Match Sparse training: mask both planes before attenuation/noise.
        rgb_cfa = low_light.apply_with_parameters((rgb * mask[:3]).contiguous(), params, 'rgb')
        w_cfa = low_light.apply_with_parameters(tensor(w) * mask[3:], params, 'w')
        return torch.cat((rgb_cfa, w_cfa)), w


def mosaic(simulated, cfa):
    """Return the exact normalized model condition (Bayer=RGB, Sparse=RGBW)."""
    adapter = get_cfa_adapter(cfa)
    if cfa == 'bayer':
        return adapter.make_conditions({'lgt_image': (simulated * 2 - 1)[None]})[0]
    scaled = (simulated - 0.5) / 0.5
    return adapter.make_conditions({'cfa': scaled[:3][None], 'cfa_w': scaled[3:][None]})[0]


def preview(path, value, gain=1.0):
    pixels = (value * gain).clamp(0, 1).mul(255).round().byte().numpy()
    pixels = pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0)
    Image.fromarray(pixels).save(path)


def write_simulation(output, image, data, w, metadata):
    image.save(output / 'input.png')
    if w is not None:
        w.save(output / 'input_w.png')
    # Preserve the actual simulated brightness. Sparse preview shows sampled RGB sites.
    mask = grid(metadata['cfa'])
    rgb = data[:3] if w is None else data[:3] * mask[:3]
    preview(output / 'simulated_rgb.png', rgb)
    if w is not None:
        preview(output / 'simulated_w.png', data[3:] * mask[3:])
    write_artifact(output, 'simulation', data, dict(metadata, input_sha256=sha256(output / 'input.png')))


def write_observation(output, condition, metadata):
    mask = grid(metadata['cfa'])
    values = (condition + 1) / 2
    gain = 1.0 if metadata['light_level'] == 1 else 1 / metadata['light_level']
    preview(output / 'cfa_rgb.png', values[:3] * mask[:3], gain)
    preview(output / 'cfa_rgb_mask.png', mask[:3].float())
    w_gain = None
    if metadata['cfa'] == 'sparse':
        w_gain = 1.0 if gain == 1 else gain / 3
        preview(output / 'cfa_w.png', values[3:] * mask[3:], w_gain)
        preview(output / 'cfa_w_mask.png', mask[3:].float())
    write_artifact(output, 'observation', condition, dict(metadata, rgb_preview_gain=gain, w_preview_gain=w_gain))


def write_artifact(folder, name, value, metadata):
    file = folder / f'{name}.npy'
    np.save(file, value.contiguous().numpy(), allow_pickle=False)
    write_json(folder / f'{name}.json', dict(metadata, schema_version=1, kind=name,
                                           array_sha256=sha256(file)))


def read_artifact(folder, name):
    info = read_json(folder / f'{name}.json')
    file = folder / f'{name}.npy'
    n = 3 if info.get('cfa') == 'bayer' else 4
    floor = 0 if name == 'simulation' else -1
    if (info.get('schema_version') != 1 or info.get('kind') != name
            or info.get('cfa') not in MODES or info.get('light_level') not in LEVELS
            or type(info.get('seed')) is not int or info['seed'] < 0
            or file.stat().st_size > n * SIZE**2 * 4 + 4096 or sha256(file) != info.get('array_sha256')):
        raise ValueError(f'Invalid {name} metadata or checksum')
    array = np.load(file, allow_pickle=False, mmap_mode='r')
    if (array.shape != (n, SIZE, SIZE) or array.dtype != np.float32 or not np.isfinite(array).all()
            or not np.all((array >= floor) & (array <= 1))):
        raise ValueError(f'Invalid {name} array shape, dtype or range')
    return torch.from_numpy(array.copy()), info


def input_metadata(path, cfa, light_level, seed, noise_std, crop):
    return dict(cfa=cfa, light_level=light_level, seed=seed, noise_std=noise_std,
                source_sha256=sha256(path), crop_xy=crop,
                w_source='pillow-imagecms-lab-l' if cfa == 'sparse' else None,
                pillow=pillow_version,
                littlecms=ImageCms.core.littlecms_version if cfa == 'sparse' else None)
