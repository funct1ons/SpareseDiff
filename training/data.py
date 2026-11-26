"""Shared low-light simulation and synthetic wiring fixtures (not production W)."""
import torch
import numpy as np
from PIL import Image


class SimulateLowLight:
    """
    对模型的输入（CFA图像）进行低光照模拟，考虑不同通道的感光效率差异。
    """

    def __init__(self, *, apply_prob, lowlight_levels=None, noise_std=0.004):
        if lowlight_levels is None:
            lowlight_levels = [1 / 30, 1 / 40, 1 / 50, 1 / 80, 1 / 100, 1 / 150]
        self.lowlight_levels = lowlight_levels
        self.noise_std = noise_std  # 统一的噪声标准差
        self.apply_prob = apply_prob

    def _sample_parameters(self):
        """
        为【一个样本】采样一组统一的随机参数。
        返回: (apply_lowlight, rgb_scale)，其中rgb_scale在apply_lowlight为False时为1.0。
        """
        apply_lowlight = (torch.rand(1).item() < self.apply_prob)
        rgb_scale = np.random.choice(self.lowlight_levels) if apply_lowlight else 1.0
        return apply_lowlight, rgb_scale

    def apply_with_parameters(self, img_tensor, parameters, channel_type='rgb'):
        """
        使用预先采样好的参数，对单张图像应用低光模拟。
        考虑不同通道的感光效率：W通道的降光系数是RGB的1/3（即更亮）

        参数:
            img_tensor: 输入张量 (值域[0,1])
            parameters: 由 _sample_parameters() 返回的 (apply_lowlight, rgb_scale)
            channel_type: 'rgb' 或 'w'
        返回:
            处理后的张量
        """
        apply_lowlight, rgb_scale = parameters

        if apply_lowlight:
            # 计算各通道的降光系数
            if channel_type == 'rgb':
                scale = rgb_scale
            else:  # 'w' 通道
                # W通道感光效率更高，降光系数是RGB的3倍，但不能超过1
                scale = min(rgb_scale * 3, 1.0)

            # 1. 根据通道类型应用不同的降光系数
            lowlight_tensor = img_tensor * scale

            # 2. 添加统一的噪声（强度相同，但按降光后的亮度比例影响）
            noise = torch.randn_like(lowlight_tensor) * self.noise_std
            lowlight_tensor = lowlight_tensor + noise
        else:
            # 正常光照，不降光但可能仍添加少量噪声（可选）
            lowlight_tensor = img_tensor

        lowlight_tensor = torch.clamp(lowlight_tensor, 0, 1)
        return lowlight_tensor


def make_dataset(size, cfa_mode):
    from datasets import Dataset

    rng = np.random.default_rng(1234)
    records = {'image': []}
    if cfa_mode == 'sparse':
        from training.cfa.sparse import CFAPattern
        mask = CFAPattern.get_rggbw_pattern().numpy()
        mask = np.tile(mask, (1, size // 4, size // 4))
        records.update(cfa=[], cfa_w=[])
    for _ in range(8):
        rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        records['image'].append(Image.fromarray(rgb))
        if cfa_mode == 'sparse':
            records['cfa'].append(Image.fromarray((rgb * mask[:3].transpose(1, 2, 0)).astype(np.uint8)))
            # Test-only luminance proxy, not a replacement for the original cached W channel.
            w = rgb.mean(axis=2) * mask[3]
            records['cfa_w'].append(Image.fromarray(w.astype(np.uint8)))
    return Dataset.from_dict(records)
