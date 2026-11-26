"""CPU checks for the public pipeline; no weights, cache, or downloads needed."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from examples.generate_samples import color_chart
from pipelines import stages
from training.evaluation.common import sha256
from training.evaluation.infer import make_condition


class PipelineTests(unittest.TestCase):
    def test_conditions_match_existing_evaluator(self):
        rgb = color_chart()
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            w = stages.lab_w(rgb)
            mask = stages.grid('sparse')
            fields = dict(gt=rgb, image_w=w, cfa_inter=rgb, cfa_inter_w=w)
            for name, value in (('cached_cfa', stages.tensor(rgb) * mask[:3]),
                                ('w', stages.tensor(w) * mask[3:])):
                stages.preview(folder / f'{name}.png', value)
            patch = {}
            for name in (*fields, 'cached_cfa', 'w'):
                path = folder / f'{name}.png'
                if name in fields:
                    fields[name].save(path)
                patch[name] = str(path)
                patch['sha256' if name == 'gt' else name + '_sha256'] = sha256(path)
            for cfa in stages.MODES:
                for level in (1, 1/40):
                    with self.subTest(cfa=cfa, level=level):
                        simulated, _ = stages.simulate(rgb, cfa, level, 42, 0.004)
                        result = stages.mosaic(simulated, cfa)
                        seed = stages.sample_seed(42, stages.PATCH_ID, 'degradation')
                        expected = make_condition(patch, dict(cfa_mode=cfa, lowlight_noise_std=0.004), level, seed,
                                                  'cached' if cfa == 'sparse' else None)[0]
                        self.assertTrue(torch.equal(result, expected))
                        self.assertTrue(torch.equal(stages.grid(cfa).sum(0), torch.ones((256, 256))))

    def test_observation_round_trip_and_hash(self):
        image = color_chart()
        data, _ = stages.simulate(image, 'sparse', 1/40, 42, 0.004)
        condition = stages.mosaic(data, 'sparse')
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            stages.write_observation(folder, condition, dict(cfa='sparse', light_level=1/40, seed=42))
            actual, metadata = stages.read_artifact(folder, 'observation')
            self.assertEqual(metadata['array_sha256'], sha256(folder / 'observation.npy'))
            self.assertTrue(torch.equal(condition, actual))
            with (folder / 'observation.npy').open('ab') as stream:
                stream.write(b'!')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                stages.read_artifact(folder, 'observation')

    def test_explicit_crop(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / 'big.png'
            Image.new('RGB', (320, 300), '#8a86df').save(file)
            with self.assertRaisesRegex(ValueError, '256x256'):
                stages.read_input(file)
            result = stages.read_input(file, (64, 44))
            self.assertEqual(np.asarray(result).shape, (256, 256, 3))
            with self.assertRaisesRegex(ValueError, 'outside'):
                stages.read_input(file, (65, 44))


if __name__ == '__main__':
    unittest.main()
