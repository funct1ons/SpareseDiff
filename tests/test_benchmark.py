"""Public RGB directory benchmark tests with real simulation and RGB metrics."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from pipelines import __main__ as cli
from training.evaluation import infer
from training.evaluation.common import read_json, sha256

REPO = Path(__file__).resolve().parents[1]


def fixture_weights(folder, model):
    models = folder / 'models'
    models.mkdir(exist_ok=True)
    shutil.copyfile(REPO / 'models' / f'{model}.json', models / f'{model}.json')
    weights = folder / 'model.safetensors'
    weights.write_bytes(b'mocked release weights')
    (models / 'checksums.sha256').write_text(f'{sha256(weights)}  {model}/model.safetensors\n', encoding='ascii')
    return models, weights


def command(input_dir, output, model, weights, *extra):
    return ['pipelines', 'benchmark', '--input-dir', str(input_dir), '--output', str(output),
            '--model', model, '--weights', str(weights), *extra]


class BenchmarkTests(unittest.TestCase):
    def assert_cli_error(self, argv, text):
        err = StringIO()
        with patch.object(sys, 'argv', argv), redirect_stderr(err), self.assertRaises(SystemExit) as raised:
            cli.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(text, err.getvalue())

    def make_directory_alias(self, alias, target):
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            if getattr(exc, 'winerror', None) != 1314:
                raise
            # Windows may deny symlinks without Developer Mode, but permit junctions.
            subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(alias), str(target)],
                           check=True, capture_output=True)

    def test_rgb_benchmark_without_dataset_dependencies(self):
        # A fresh interpreter prevents a previously imported dataset stack from hiding an eager import.
        script = '''
import importlib.abc
import runpy
import sys

class NoDatasetImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'datasets', 'pyarrow', 'skimage'}:
            raise AssertionError(f'Unexpected public benchmark import: {fullname}')

sys.meta_path.insert(0, NoDatasetImports())
runpy.run_path('tests/test_benchmark.py')['BenchmarkTests']()._check_rgb_benchmark()
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=REPO, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _check_rgb_benchmark(self):
        import torch
        from pipelines import stages

        torch.set_num_threads(1)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / 'my photos'
            (inputs / 'z').mkdir(parents=True)
            Image.new('RGB', (256, 256), 'green').save(inputs / 'z' / 'same.png')
            (inputs / 'a').mkdir()
            wide = Image.new('RGB', (519, 517), 'magenta')
            for x, y, color in ((0, 0, 'red'), (256, 0, 'yellow'), (0, 256, 'green'), (256, 256, 'blue')):
                wide.paste(color, (x, y, x + 256, y + 256))
            wide.save(inputs / 'a' / 'same.png')
            Image.new('RGB', (256, 256), (64, 64, 64)).save(inputs / 'a' / 'same.bmp')
            exif = Image.Exif()
            exif[274] = 6  # Rotate clockwise before calculating the patch grid.
            rotated = Image.new('RGB', (256, 520), 'white')
            rotated.paste('blue', (0, 264, 256, 520))
            rotated.save(inputs / 'rotate.PNG', exif=exif)
            Image.new('RGB', (100, 300)).save(inputs / 'small.png')
            (inputs / 'sidecar.json').write_text('{}', encoding='ascii')

            ids = ['a/same.bmp/0_0', 'a/same.png/0_0', 'a/same.png/256_0',
                   'a/same.png/0_256', 'a/same.png/256_256', 'rotate.PNG/0_0',
                   'rotate.PNG/256_0', 'z/same.png/0_0']
            first_crops = None
            for cfa, level in (('bayer', 1.0), ('sparse', 1/40)):
                if cfa == 'sparse':
                    inputs = shutil.copytree(inputs, root / 'relocated')
                model = cfa + '_mixed'
                output = root / 'results' / cfa
                models, weights = fixture_weights(root, model)
                piq = sys.modules['piq'] = ModuleType('piq')
                piq.fsim = Mock(return_value=torch.tensor(0.75))
                index = 0

                def sample(denoiser, condition, **kwargs):
                    nonlocal index
                    patch_id = ids[index]
                    relative, xy = patch_id.rsplit('/', 1)
                    crop = stages.read_input(inputs / relative, tuple(map(int, xy.split('_'))))
                    data, _ = stages.simulate(crop, cfa, level, 57, 0.004, patch_id=patch_id)
                    self.assertTrue(torch.equal(condition, stages.mosaic(data, cfa)[None]))
                    self.assertEqual(kwargs, dict(seed=infer.sample_seed(57, patch_id, 'initial-noise'),
                                                  steps=2, method='euler', cfg_scale=0.0, precision='fp32',
                                                  apply_constraint=True, constraint_schedule='final',
                                                  constraint_strength=float(level == 1)))
                    index += 1
                    if index == 1:
                        return torch.full((1, 3, 256, 256), -1.)
                    return (stages.tensor(crop) * 2 - 1)[None]

                extra = ['--seed', '57', '--steps', '2']
                extra.extend(['--no-fsimc'] if cfa == 'bayer' else ['--light-level', '1/40'])
                with patch.object(cli, 'MODEL_DIR', models), patch.object(infer, 'load_model') as load, \
                     patch.object(infer, 'sample', side_effect=sample) as sampler, \
                     patch.object(sys, 'argv', command(inputs, output, model, weights, *extra)), redirect_stdout(StringIO()):
                    cli.main()
                load.assert_called_once_with(weights.resolve(), infer.read_config(models / f'{model}.json'), 'cpu')
                self.assertEqual(sampler.call_count, len(ids))

                grid = read_json(output / 'manifest.json')
                pred = read_json(output / 'predictions' / 'predictions.json')
                scores = read_json(output / 'metrics.json')
                self.assertEqual([row['id'] for row in grid['patches']], ids)
                self.assertEqual([row['id'] for row in pred['predictions']], ids)
                self.assertEqual([row['id'] for row in scores['patches']], ids)
                self.assertEqual([row['path'] for row in grid['sources']],
                                 ['a/same.bmp', 'a/same.png', 'rotate.PNG', 'small.png', 'z/same.png'])
                self.assertEqual(grid['source'], 'rgb-simulated')
                self.assertEqual(grid['input_dir'], str(inputs.resolve()))
                self.assertEqual(grid['edge'], 'drop')
                self.assertEqual(grid['patch_size'], 256)
                self.assertEqual(grid['sources'][1]['dropped_right'], 7)
                self.assertEqual(grid['sources'][1]['dropped_bottom'], 5)
                self.assertEqual(grid['sources'][2]['size'], [520, 256])
                self.assertEqual(grid['sources'][3]['patch_count'], 0)
                self.assertEqual([row['size'] for row in grid['patches']], [[256, 256]] * len(ids))
                self.assertEqual([row['xy'] for row in grid['patches'][1:5]], [[0, 0], [256, 0], [0, 256], [256, 256]])
                checksums = [row['sha256'] for row in grid['patches']]
                if first_crops is None:
                    first_crops = checksums
                else:
                    self.assertEqual(checksums, first_crops)
                for gt, result in zip(grid['patches'], pred['predictions']):
                    self.assertEqual(Path(gt['gt']).parent, output.resolve() / 'ground_truth')
                    self.assertEqual(Path(result['path']).parent, output.resolve() / 'predictions')
                    self.assertEqual(gt['source_sha256'], sha256(inputs / gt['source_relative']))
                    self.assertEqual(result['gt_sha256'], sha256(gt['gt']))
                    self.assertEqual(result['sha256'], sha256(result['path']))
                    self.assertEqual(result['degradation_seed'], infer.sample_seed(57, gt['id'], 'degradation'))
                    self.assertEqual(result['sampler_seed'], infer.sample_seed(57, gt['id'], 'initial-noise'))
                with Image.open(grid['patches'][5]['gt']) as crop:
                    self.assertEqual(crop.getpixel((0, 0)), (0, 0, 255))
                self.assertEqual(pred['config']['checkpoint_sha256'], sha256(weights))
                self.assertEqual(pred['config']['training_config_sha256'], sha256(models / f'{model}.json'))
                self.assertEqual(pred['config']['manifest_sha256'], sha256(output / 'manifest.json'))
                self.assertEqual(pred['config']['cfa_input_source'], 'rgb-simulated')
                self.assertEqual(pred['config']['simulation'], grid['simulation'])
                self.assertEqual(grid['simulation']['cfa'], cfa)
                self.assertEqual(grid['simulation']['light_level'], level)
                self.assertEqual(grid['simulation']['w_source'], 'pillow-imagecms-lab-l' if cfa == 'sparse' else None)
                self.assertEqual(bool(grid['versions']['littlecms']), cfa == 'sparse')
                self.assertTrue(grid['versions']['pillow'])
                self.assertTrue(grid['versions']['torch'])
                self.assertEqual(scores['overall']['count'], len(ids))
                self.assertEqual(set(scores['datasets']), {inputs.name})
                self.assertAlmostEqual(scores['patches'][0]['psnr'], 20 * math.log10(255 / 64))
                self.assertLess(scores['patches'][0]['ssim'], 1)
                self.assertEqual(scores['patches'][1]['psnr'], 'Infinity')
                self.assertAlmostEqual(scores['patches'][1]['ssim'], 1)
                if cfa == 'sparse':
                    self.assertEqual(piq.fsim.call_count, len(ids))
                    x, y = piq.fsim.call_args_list[0].args
                    torch.testing.assert_close(x, torch.full((1, 3, 256, 256), 64/255))
                    self.assertEqual(torch.count_nonzero(y).item(), 0)
                    self.assertEqual(piq.fsim.call_args.kwargs, dict(data_range=1., chromatic=True))
                    self.assertEqual(scores['overall']['fsimc'], 0.75)
                else:
                    piq.fsim.assert_not_called()
                    self.assertNotIn('fsimc', scores['overall'])

    def test_benchmark_rejects_bad_inputs_and_weights_before_loading(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = 'bayer_normal'
            models, weights = fixture_weights(root, model)
            inputs, output = root / 'inputs', root / 'output'
            argv = command(inputs, output, model, weights, '--no-fsimc')

            def fails(text, argv=argv):
                self.assert_cli_error(argv, text)
                self.assertFalse(output.exists())

            with patch.object(cli, 'MODEL_DIR', models), patch.object(infer, 'load_model') as load:
                fails('must be an image directory')
                inputs.mkdir()
                fails('No supported RGB image files')
                image = inputs / 'image.png'
                Image.new('RGB', (255, 256)).save(image)
                fails('No complete 256x256 RGB patches')
                Image.new('L', (256, 256)).save(image)
                fails('Expected an 8-bit RGB image')
                image.write_bytes(b'not a PNG')
                fails('cannot identify image file')
                fails('Choose a mixed checkpoint', argv + ['--light-level', '1/40'])
                weights.write_bytes(b'wrong checksum')
                fails('Weights SHA-256 does not match')
                fails('unrecognized arguments: --manifest old.json', argv + ['--manifest', 'old.json'])
                load.assert_not_called()

    def test_benchmark_rejects_output_in_resolved_input_tree_before_discovery(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = 'bayer_normal'
            models, weights = fixture_weights(root, model)
            inputs, alias = root / 'inputs', root / 'alias'
            inputs.mkdir()
            self.make_directory_alias(alias, inputs)
            output = inputs / 'new' / 'result'
            with patch.object(cli, 'MODEL_DIR', models), patch.object(Path, 'rglob') as glob:
                for source, destination in ((inputs, output), (inputs, alias / 'new' / 'result'), (alias, output)):
                    with self.subTest(source=source, destination=destination):
                        self.assert_cli_error(command(source, destination, model, weights),
                                              '--output must be outside --input-dir')
                glob.assert_not_called()
            self.assertFalse(output.parent.exists())

    def test_benchmark_rejects_existing_requested_output_before_resolving(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = 'bayer_normal'
            models, weights = fixture_weights(root, model)
            inputs, live, dangling = root / 'inputs', root / 'live', root / 'dangling'
            inputs.mkdir()
            Image.new('RGB', (256, 256)).save(inputs / 'image.png')
            existing_dir, existing_file, missing = root / 'existing_dir', root / 'existing_file', root / 'missing'
            existing_dir.mkdir()
            existing_file.write_bytes(b'keep')
            self.make_directory_alias(live, existing_dir)
            self.make_directory_alias(dangling, missing)
            self.assertFalse(dangling.exists())
            with patch.object(cli, 'MODEL_DIR', models):
                for output in (existing_file, existing_dir, live, dangling):
                    with self.subTest(output=output):
                        self.assert_cli_error(command(inputs, output, model, weights), f'Output already exists: {output}')
            self.assertTrue(os.path.lexists(dangling))
            self.assertFalse(missing.exists())
            self.assertEqual(list(existing_dir.iterdir()), [])
            self.assertEqual(existing_file.read_bytes(), b'keep')

    def test_benchmark_late_bad_rgb_input_leaves_no_requested_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = 'bayer_normal'
            models, weights = fixture_weights(root, model)
            inputs, output = root / 'inputs', root / 'not-yet-created' / 'output'
            inputs.mkdir()
            Image.new('RGB', (256, 256)).save(inputs / 'a-valid.png')
            late = inputs / 'z-invalid.png'
            before = set(root.iterdir())
            with patch.object(cli, 'MODEL_DIR', models), patch.object(infer, 'load_model') as load:
                late.write_bytes(b'corrupt PNG')
                self.assert_cli_error(command(inputs, output, model, weights), 'cannot identify image file')
                self.assertEqual(set(root.iterdir()), before)
                self.assertFalse(os.path.lexists(output))
                Image.new('RGBA', (256, 256)).save(late)
                self.assert_cli_error(command(inputs, output, model, weights), 'Expected an 8-bit RGB image')
                self.assertEqual(set(root.iterdir()), before)
                self.assertFalse(os.path.lexists(output))
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
