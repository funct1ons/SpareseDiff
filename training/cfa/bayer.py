"""
Bayer CFA Pattern处理工具 - GPU优化版本
针对RGGB Bayer马赛克模式的向量化处理函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import torchvision.transforms.functional as TF
from training.cfa.base import CFAAdapter


def rgb_to_bayer_cfa(rgb_tensor: torch.Tensor, bayer_pattern: str = "GBRG") -> torch.Tensor:
    """
    将RGB图像转换为Bayer CFA模拟结果

    Args:
        rgb_tensor: RGB图像张量，形状为 (C, H, W)，其中 C=3，值范围[0, 1]或[-1, 1]
        bayer_pattern: Bayer模式，支持 "RGGB", "GRBG", "GBRG", "BGGR"
                         "GBRG" 模式:
                        G B
                        R G

    Returns:
        Bayer CFA模拟结果张量，形状为 (C, H, W)
        每个通道只在对应Bayer位置有值，其他位置为-1
    """
    # 验证输入形状
    if rgb_tensor.dim() != 3:
        raise ValueError(f"输入张量必须是3维 (C, H, W)，但得到 {rgb_tensor.dim()} 维")

    C, H, W = rgb_tensor.shape
    if C != 3:
        raise ValueError(f"输入张量必须有3个通道，但得到 {C} 个通道")

    # 确保高度和宽度是2的倍数
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"高度和宽度必须是2的倍数，但得到 H={H}, W={W}")

    # 获取设备
    device = rgb_tensor.device
    dtype = rgb_tensor.dtype

    # 创建输出的Bayer CFA张量（初始化为-1）
    bayer_cfa = torch.full(rgb_tensor.shape, -1.00)

    # 根据不同的Bayer模式进行处理
    if bayer_pattern == "GBRG":  # 默认要求的模式
        # 模式:
        # G B
        # R G
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                # 第一个2x2块
                # (i, j): G通道
                bayer_cfa[1, i, j] = rgb_tensor[1, i, j]  # G
                # (i, j+1): B通道
                bayer_cfa[2, i, j + 1] = rgb_tensor[2, i, j + 1]  # B
                # (i+1, j): R通道
                bayer_cfa[0, i + 1, j] = rgb_tensor[0, i + 1, j]  # R
                # (i+1, j+1): G通道
                bayer_cfa[1, i + 1, j + 1] = rgb_tensor[1, i + 1, j + 1]  # G

    elif bayer_pattern == "RGGB":  # 标准模式
        # 模式:
        # R G
        # G B
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                # 第一个2x2块
                # (i, j): R通道
                bayer_cfa[0, i, j] = rgb_tensor[0, i, j]  # R
                # (i, j+1): G通道
                bayer_cfa[1, i, j + 1] = rgb_tensor[1, i, j + 1]  # G
                # (i+1, j): G通道
                bayer_cfa[1, i + 1, j] = rgb_tensor[1, i + 1, j]  # G
                # (i+1, j+1): B通道
                bayer_cfa[2, i + 1, j + 1] = rgb_tensor[2, i + 1, j + 1]  # B

    elif bayer_pattern == "GRBG":  # 另一种常见模式
        # 模式:
        # G R
        # B G
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                # 第一个2x2块
                # (i, j): G通道
                bayer_cfa[1, i, j] = rgb_tensor[1, i, j]  # G
                # (i, j+1): R通道
                bayer_cfa[0, i, j + 1] = rgb_tensor[0, i, j + 1]  # R
                # (i+1, j): B通道
                bayer_cfa[2, i + 1, j] = rgb_tensor[2, i + 1, j]  # B
                # (i+1, j+1): G通道
                bayer_cfa[1, i + 1, j + 1] = rgb_tensor[1, i + 1, j + 1]  # G

    elif bayer_pattern == "BGGR":  # 另一种常见模式
        # 模式:
        # B G
        # G R
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                # 第一个2x2块
                # (i, j): B通道
                bayer_cfa[2, i, j] = rgb_tensor[2, i, j]  # B
                # (i, j+1): G通道
                bayer_cfa[1, i, j + 1] = rgb_tensor[1, i, j + 1]  # G
                # (i+1, j): G通道
                bayer_cfa[1, i + 1, j] = rgb_tensor[1, i + 1, j]  # G
                # (i+1, j+1): R通道
                bayer_cfa[0, i + 1, j + 1] = rgb_tensor[0, i + 1, j + 1]  # R

    else:
        raise ValueError(f"不支持的Bayer模式: {bayer_pattern}，支持的格式: 'RGGB', 'GRBG', 'GBRG', 'BGGR'")

    return bayer_cfa


def rgb_to_bayer_cfa_vectorized(rgb_tensor: torch.Tensor, bayer_pattern: str = "GBRG") -> torch.Tensor:
    """
    向量化版本的RGB到Bayer CFA转换函数（GPU优化）

    Args:
        rgb_tensor: RGB图像张量，形状为 (C, H, W) 或 (B, C, H, W)
        bayer_pattern: Bayer模式，支持 "RGGB", "GRBG", "GBRG", "BGGR"

    Returns:
        Bayer CFA模拟结果张量，形状与输入相同
    """
    # 处理批量输入
    if rgb_tensor.dim() == 4:
        B, C, H, W = rgb_tensor.shape
        batch_mode = True
    elif rgb_tensor.dim() == 3:
        C, H, W = rgb_tensor.shape
        batch_mode = False
        rgb_tensor = rgb_tensor.unsqueeze(0)  # 添加批次维度
        B = 1
    else:
        raise ValueError(f"输入张量必须是3维或4维，但得到 {rgb_tensor.dim()} 维")

    if C != 3:
        raise ValueError(f"输入张量必须有3个通道，但得到 {C} 个通道")

    # 确保高度和宽度是2的倍数
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"高度和宽度必须是2的倍数，但得到 H={H}, W={W}")

    device = rgb_tensor.device
    dtype = rgb_tensor.dtype

    # 创建输出的Bayer CFA张量（初始化为-1）
    bayer_cfa = torch.full(rgb_tensor.shape, -1.00,device=device, dtype=dtype)

    # 创建Bayer掩码
    if bayer_pattern == "GBRG": # 默认的模式
        # 创建各通道的掩码
        g_mask = torch.zeros(1, 1, 2, 2, device=device, dtype=dtype)
        b_mask = torch.zeros(1, 1, 2, 2, device=device, dtype=dtype)
        r_mask = torch.zeros(1, 1, 2, 2, device=device, dtype=dtype)

        # G通道掩码 (位置(0,0)和(1,1))
        g_mask[0, 0, 0, 0] = 1.0
        g_mask[0, 0, 1, 1] = 1.0

        # B通道掩码 (位置(0,1))
        b_mask[0, 0, 0, 1] = 1.0

        # R通道掩码 (位置(1,0))
        r_mask[0, 0, 1, 0] = 1.0

    elif bayer_pattern == "RGGB":  # 标准模式
        # 创建各通道的掩码
        r_mask = torch.zeros(1, 1, 2, 2, device=device, dtype=dtype)
        g_mask = torch.zeros(1, 1, 2, 2, device=device, dtype=dtype)
        b_mask = torch.zeros(1, 1, 2, 2, device=device, dtype=dtype)

        # R通道掩码 (位置(0,0))
        r_mask[0, 0, 0, 0] = 1.0

        # G通道掩码 (位置(0,1)和(1,0))
        g_mask[0, 0, 0, 1] = 1.0
        g_mask[0, 0, 1, 0] = 1.0

        # B通道掩码 (位置(1,1))
        b_mask[0, 0, 1, 1] = 1.0

    else:
        raise ValueError(f"向量化版本目前仅支持 'RGGB' 和 'GBRG' 模式")

    # 重复掩码以匹配图像尺寸
    h_repeats = H // 2
    w_repeats = W // 2

    r_mask_full = r_mask.repeat(B, 1, h_repeats, w_repeats)
    g_mask_full = g_mask.repeat(B, 1, h_repeats, w_repeats)
    b_mask_full = b_mask.repeat(B, 1, h_repeats, w_repeats)

    # 应用掩码
    bayer_cfa[:, 0:1] = rgb_tensor[:, 0:1] * r_mask_full + (-1.00 * (1 - r_mask_full) )   # R通道
    bayer_cfa[:, 1:2] = rgb_tensor[:, 1:2] * g_mask_full + (-1.00 * (1 - g_mask_full) )   # G通道
    bayer_cfa[:, 2:3] = rgb_tensor[:, 2:3] * b_mask_full + (-1.00 * (1 - b_mask_full) )   # B通道

    # 移除批次维度（如果是单张图像）
    if not batch_mode:
        bayer_cfa = bayer_cfa.squeeze(0)

    return bayer_cfa

class CFAPattern:
    """
    CFA Pattern定义类
    定义RGGB Bayer CFA模式的布局
    """

    def __init__(self, pattern_mask: Optional[torch.Tensor] = None):
        """
        CFA Pattern定义类
        定义RGGB Bayer CFA模式的布局

        注意: pattern_mask的形状应为 [3, 2, 2]
            第0维: 通道索引 (0:R, 1:G, 2:B)
            第1维: 高度 (2x2块中的行，0-1)
            第2维: 宽度 (2x2块中的列，0-1)
        """
        if pattern_mask is None:
            pattern_mask = CFAPattern.get_bayer_pattern()

        # 形状断言检查
        assert pattern_mask.dim() == 3, f"pattern_mask必须是3维张量，得到{pattern_mask.dim()}维"
        assert pattern_mask.shape == (3, 2, 2), f"pattern_mask形状必须为(3,2,2)，得到{pattern_mask.shape}"
        assert torch.all(torch.isin(pattern_mask, torch.tensor([0., 1.]))), "pattern_mask必须只包含0和1"

        self.pattern_mask = pattern_mask.float()

    @staticmethod
    def get_bayer_pattern() -> torch.Tensor:
        """
        获取RGGB Bayer CFA pattern掩码
        返回形状为 [3, 2, 2] 的张量，其中:
        - 第一维: 通道索引 (0:R, 1:G, 2:B)
        - 第二维: 高度 (2x2块中的行)
        - 第三维: 宽度 (2x2块中的列)
        """
        pattern = torch.zeros(3, 2, 2, dtype=torch.float32)

        # R通道: 位置(1,0)
        pattern[0, 1, 0] = 1.0

        # G通道: 位置(0,0)和(1,1)
        pattern[1, 0, 0] = 1.0
        pattern[1, 1, 1] = 1.0

        # B通道: 位置(1,0)
        pattern[2, 0, 1] = 1.0

        return pattern


class CFAPatternReconstructor(nn.Module):
    """
    极致优化的CFA Pattern重构器 - GPU加速版本
    使用卷积和转置卷积操作实现完全向量化的RGGB Bayer马赛克重构

    主要优化：
    1. 完全消除所有Python循环
    2. 使用卷积操作代替手动循环
    3. 预计算所有核和掩码
    4. 支持批量处理和高性能GPU加速
    """

    def __init__(self, pattern_mask: Optional[torch.Tensor] = None):
        super().__init__()

        if pattern_mask is None:
            pattern_mask = CFAPattern.get_bayer_pattern()

        # 验证pattern_mask
        assert pattern_mask.dim() == 3, f"pattern_mask必须是3维张量，得到{pattern_mask.dim()}维"
        assert pattern_mask.shape == (3, 2, 2), f"pattern_mask形状必须为(3,2,2)，得到{pattern_mask.shape}"

        # 注册为buffer，自动处理设备移动
        self.register_buffer('pattern_mask', pattern_mask.float())

        # 创建卷积核和掩码
        self._create_convolutional_kernels()

        # 创建上采样层
        self._create_upsample_layers()

        # 验证配置
        self._validate_config()

    def _create_convolutional_kernels(self):
        """创建卷积操作所需的所有核和掩码"""
        # 提取各通道的掩码
        r_mask = self.pattern_mask[0]  # [2, 2]
        g_mask = self.pattern_mask[1]  # [2, 2]
        b_mask = self.pattern_mask[2]  # [2, 2]

        # ========== 1. 计算平均值所需的卷积核 ==========

        # R通道卷积核 [1, 1, 2, 2]
        r_kernel = r_mask / r_mask.sum().clamp_min(1e-5)
        self.register_buffer('r_kernel', r_kernel.view(1, 1, 2, 2))

        # G通道有两个位置，需要分别处理
        g_positions = torch.nonzero(g_mask)
        if len(g_positions) != 2:
            raise ValueError("G通道应该有2个已知像素位置")

        # G1通道卷积核 (位置0,1)
        g1_kernel = torch.zeros_like(g_mask)
        g1_kernel[g_positions[0, 0], g_positions[0, 1]] = 1.0
        self.register_buffer('g1_kernel', g1_kernel.view(1, 1, 2, 2))

        # G2通道卷积核 (位置1,0)
        g2_kernel = torch.zeros_like(g_mask)
        g2_kernel[g_positions[1, 0], g_positions[1, 1]] = 1.0
        self.register_buffer('g2_kernel', g2_kernel.view(1, 1, 2, 2))

        # B通道卷积核
        b_kernel = b_mask / b_mask.sum().clamp_min(1e-5)
        self.register_buffer('b_kernel', b_kernel.view(1, 1, 2, 2))

        # ========== 2. 验证掩码 ==========
        self._validate_masks(r_mask, g_mask, b_mask)

    def _create_upsample_layers(self):
        """创建上采样层（转置卷积）"""

        self.upsample_r = nn.ConvTranspose2d(
            1, 1, kernel_size=2, stride=2, bias=False
        )
        self.upsample_g1 = nn.ConvTranspose2d(
            1, 1, kernel_size=2, stride=2, bias=False
        )
        self.upsample_g2 = nn.ConvTranspose2d(
            1, 1, kernel_size=2, stride=2, bias=False
        )
        self.upsample_b = nn.ConvTranspose2d(
            1, 1, kernel_size=2, stride=2, bias=False
        )

        # 初始化上采样核为全1（固定，不训练）
        with torch.no_grad():
            for upsample_layer in [self.upsample_r, self.upsample_g1,
                                   self.upsample_g2, self.upsample_b]:
                upsample_layer.weight.fill_(1.0)
                upsample_layer.weight.requires_grad = False

    def _validate_masks(self, r_mask, g_mask, b_mask):
        """验证掩码的正确性"""
        # 确保掩码是二值的
        for name, mask in [('R', r_mask), ('G', g_mask), ('B', b_mask)]:
            unique_vals = torch.unique(mask)
            if not torch.all(torch.isin(unique_vals, torch.tensor([0., 1.]))):
                raise ValueError(f"{name}掩码应该只包含0和1")

        # 确保没有重叠
        total_mask = r_mask + g_mask + b_mask
        if torch.any(total_mask > 1):
            raise ValueError("掩码存在重叠")

        # 确保覆盖整个2x2块
        if not torch.allclose(total_mask, torch.ones_like(total_mask)):
            raise ValueError("掩码没有完全覆盖2x2块")

        # 打印统计信息
        if torch.is_grad_enabled():  # 只在训练时打印
            print(f"CFA Pattern验证通过:")
            print(f"  R像素数: {r_mask.sum().item()}")
            print(f"  G像素数: {g_mask.sum().item()}")
            print(f"  B像素数: {b_mask.sum().item()}")
            print(f"  总像素数: {total_mask.sum().item()}/4")

    def _validate_config(self):
        """验证配置"""
        # 检查所有buffer都已正确注册
        required_buffers = ['r_kernel', 'g1_kernel', 'g2_kernel', 'b_kernel']
        for buf_name in required_buffers:
            if not hasattr(self, buf_name):
                raise AttributeError(f"缺少必需的buffer: {buf_name}")

        # 检查上采样层
        required_layers = ['upsample_r', 'upsample_g1', 'upsample_g2', 'upsample_b']
        for layer_name in required_layers:
            if not hasattr(self, layer_name):
                raise AttributeError(f"缺少必需的上采样层: {layer_name}")

    def forward(self, mosaic: torch.Tensor) -> torch.Tensor:
        """
        向量化的前向传播 - 使用卷积操作

        Args:
            mosaic: RGGB Bayer马赛克，形状为 [B, 3, H, W]
                    通道顺序: [R, G, B]

        Returns:
            4通道稠密表示，形状为 [B, 4, H, W]
            通道顺序: [R_dense, G1_dense, G2_dense, B_dense]
        """
        B, C, H, W = mosaic.shape

        # 验证输入
        assert C == 3, f"输入应该有3个通道，但得到{C}"
        assert H % 2 == 0 and W % 2 == 0, \
            f"高度和宽度应该是2的倍数，但得到{H}x{W}"

        # ========== 1. 分离各通道 ==========
        r_channel = mosaic[:, 0:1]  # [B, 1, H, W]
        g_channel = mosaic[:, 1:2]  # [B, 1, H, W]
        b_channel = mosaic[:, 2:3]  # [B, 1, H, W]

        # ========== 2. 计算每个2x2块的平均值 ==========
        # 使用stride=2卷积，每个2x2块输出一个值

        # R通道
        r_avg = F.conv2d(
            r_channel, self.r_kernel,
            stride=2, padding=0
        )

        # G1通道平均值 (位置0,1)
        g1_avg = F.conv2d(
            g_channel, self.g1_kernel,
            stride=2, padding=0
        )

        # G2通道平均值 (位置1,0)
        g2_avg = F.conv2d(
            g_channel, self.g2_kernel,
            stride=2, padding=0
        )

        # B通道
        b_avg = F.conv2d(
            b_channel, self.b_kernel,
            stride=2, padding=0
        )

        # ========== 3. 上采样到原始大小 ==========
        # 使用转置卷积将每个块的平均值扩展到整个2x2块

        r_dense = self.upsample_r(r_avg)  # [B, 1, H, W]
        g1_dense = self.upsample_g1(g1_avg)  # [B, 1, H, W]
        g2_dense = self.upsample_g2(g2_avg)  # [B, 1, H, W]
        b_dense = self.upsample_b(b_avg)  # [B, 1, H, W]

        # ========== 4. 堆叠所有通道 ==========
        dense_output = torch.cat([
            r_dense, g1_dense, g2_dense, b_dense
        ], dim=1)  # [B, 4, H, W]

        return dense_output

    def reconstruct_dense(self, mosaic: torch.Tensor) -> torch.Tensor:
        """
        重构为稠密表示（向后兼容接口）

        Args:
            mosaic: 稀疏RGGB Bayer马赛克，形状为 [B, 3, H, W]

        Returns:
            4通道稠密表示，形状为 [B, 4, H, W]
        """
        return self.forward(mosaic)

    def create_sparse_mosaic(self, rgb_image: torch.Tensor,
                             noise_level: float = 0.0) -> torch.Tensor:
        """
        从RGB图像创建稀疏RGGB Bayer马赛克（向量化版本）

        Args:
            rgb_image: RGB图像，形状为 [B, 3, H, W]，值范围[-1, 1]或[0, 1]
            noise_level: 添加的高斯噪声标准差

        Returns:
            稀疏RGGB Bayer马赛克，形状为 [B, 3, H, W]
        """
        B, C, H, W = rgb_image.shape
        assert C == 3, f"输入应该是RGB图像，但得到{C}通道"
        assert H % 2 == 0 and W % 2 == 0, \
            f"高度和宽度应该是2的倍数，但得到{H}x{W}"

        device = rgb_image.device

        # 确保图像在[0, 1]范围
        if rgb_image.min() < -0.5:  # 假设是[-1, 1]范围
            rgb_image = (rgb_image + 1) / 2

        # 初始化马赛克
        mosaic = torch.zeros(B, 3, H, W, device=device, dtype=rgb_image.dtype)

        # ========== 向量化创建稀疏马赛克 ==========

        # 1. 分离RGB通道
        r_channel = rgb_image[:, 0:1]  # [B, 1, H, W]
        g_channel = rgb_image[:, 1:2]  # [B, 1, H, W]
        b_channel = rgb_image[:, 2:3]  # [B, 1, H, W]

        # 2. 创建RGB掩码
        r_mask_full = self._create_channel_mask('R', H, W, device).repeat(B, 1, 1, 1)
        g_mask_full = self._create_channel_mask('G', H, W, device).repeat(B, 1, 1, 1)
        b_mask_full = self._create_channel_mask('B', H, W, device).repeat(B, 1, 1, 1)

        # 3. 应用掩码创建稀疏通道
        mosaic[:, 0:1] = r_channel * r_mask_full  # R通道
        mosaic[:, 1:2] = g_channel * g_mask_full  # G通道
        mosaic[:, 2:3] = b_channel * b_mask_full  # B通道

        # 4. 添加噪声（可选）
        if noise_level > 0:
            noise = torch.randn_like(mosaic) * noise_level
            mosaic = mosaic + noise
            mosaic = torch.clamp(mosaic, 0, 1)

        return mosaic

    def dense_to_sparse(self, dense: torch.Tensor) -> torch.Tensor:
        """
        将4通道稠密表示转换回稀疏表示（向量化版本）

        Args:
            dense: 4通道稠密表示 [B, 4, H, W]

        Returns:
            稀疏RGGB Bayer马赛克 [B, 3, H, W]
        """
        B, C, H, W = dense.shape
        assert C == 4, f"输入应该有4个通道，但得到{C}"
        assert H % 2 == 0 and W % 2 == 0

        device = dense.device

        # 创建通道掩码
        r_mask_full = self._create_channel_mask('R', H, W, device).repeat(B, 1, 1, 1)
        g_mask_full = self._create_channel_mask('G', H, W, device).repeat(B, 1, 1, 1)
        b_mask_full = self._create_channel_mask('B', H, W, device).repeat(B, 1, 1, 1)

        # 初始化稀疏马赛克
        sparse_mosaic = torch.zeros(B, 3, H, W, device=device)

        # 应用掩码提取稀疏值
        # R通道
        sparse_mosaic[:, 0:1] = dense[:, 0:1] * r_mask_full

        # G通道：需要合并G1和G2
        g1_mask = self._create_specific_g_mask('G1', H, W, device).repeat(B, 1, 1, 1)
        g2_mask = self._create_specific_g_mask('G2', H, W, device).repeat(B, 1, 1, 1)

        g_sparse = (dense[:, 1:2] * g1_mask +
                    dense[:, 2:3] * g2_mask)
        sparse_mosaic[:, 1:2] = g_sparse

        # B通道
        sparse_mosaic[:, 2:3] = dense[:, 3:4] * b_mask_full

        return sparse_mosaic

    def _create_channel_mask(self, channel: str, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        创建指定通道的掩码 - 使用重复而不是插值

        Args:
            channel: 通道名称 ('R', 'G', 'B')
            H: 高度
            W: 宽度
            device: 设备

        Returns:
            通道掩码，形状为 [1, 1, H, W]
        """
        # 通道索引映射
        channel_map = {'R': 0, 'G': 1, 'B': 2}

        if channel not in channel_map:
            raise ValueError(f"无效的通道名称: {channel}，应为 'R', 'G' 或 'B'")

        # 获取2x2的通道掩码
        channel_idx = channel_map[channel]
        base_mask = self.pattern_mask[channel_idx]  # [2, 2]

        # 使用 repeat 操作在空间维度上重复基础掩码
        h_repeats = H // 2
        w_repeats = W // 2

        mask_full = base_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]
        mask_full = mask_full.repeat(1, 1, h_repeats, w_repeats)  # [1, 1, H, W]

        return mask_full.to(device)

    def _create_specific_g_mask(self, g_type: str, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        创建特定的G通道掩码（G1或G2）- 使用重复而不是插值

        Args:
            g_type: G类型 ('G1' 或 'G2')
            H: 高度
            W: 宽度
            device: 设备

        Returns:
            G通道特定位置的掩码，形状为 [1, 1, H, W]
        """
        if g_type not in ['G1', 'G2']:
            raise ValueError(f"无效的G类型: {g_type}，应为 'G1' 或 'G2'")

        # 获取对应的2x2掩码核
        if g_type == 'G1':
            base_mask = self.g1_kernel.squeeze()  # [2, 2]
        else:  # G2
            base_mask = self.g2_kernel.squeeze()  # [2, 2]

        # 使用 repeat 操作在空间维度上重复基础掩码
        h_repeats = H // 2
        w_repeats = W // 2

        # 创建完整掩码 [1, 1, H, W]
        mask_full = base_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]
        mask_full = mask_full.repeat(1, 1, h_repeats, w_repeats)  # [1, 1, H, W]

        return mask_full.to(device)


class BayerAdapter(CFAAdapter):
    mode = 'bayer'
    condition_channels = 3
    encoder_channels = 4
    tile_size = 2
    required_columns = ('image',)
    batch_columns = ('image', 'lgt_image')
    debug_checks = False
    samples_per_light = 6
    lowlight_apply_constraint = False

    def pattern_mask(self):
        return CFAPattern().pattern_mask

    def create_reconstructor(self, pattern_mask=None):
        return CFAPatternReconstructor(pattern_mask)

    def make_conditions(self, batch, dtype=torch.float32):
        return rgb_to_bayer_cfa_vectorized(batch['lgt_image'].to(dtype), bayer_pattern='GBRG')

    def visualization_inputs(self, batch, device):
        conditions = rgb_to_bayer_cfa_vectorized(batch['lgt_image'].to(device), bayer_pattern='GBRG')
        return conditions, conditions, None

    def preprocess(self, examples, *, resolution, lowlight_simulator, base_transform):
        self.validate_images(examples, resolution)
        # 1. 处理目标图像 (image): 正常光照，直接变换
        images = [base_transform(image.convert("RGB")) for image in examples["image"]]

        # 2. 处理输入图像 cfa
        lgt_images = []

        # 3. 添加光照水平标签
        light_levels = []  # 记录每个样本的光照水平

        for img in examples["image"]:
            # 【关键步骤】：为这一对 (cfa, cfa_w) 采样【同一组】随机参数
            apply_lowlight, rgb_scale = lowlight_simulator._sample_parameters()

            # 记录RGB通道的光照水平
            if not apply_lowlight:
                light_levels.append(1.0)  # 正常光照
            else:
                light_levels.append(rgb_scale)  # RGB通道的光照水平

            # 对CFA (RGB) 应用这组参数
            img_tensor = TF.to_tensor(img.convert("RGB"))
            img_tensor_lgt = lowlight_simulator.apply_with_parameters(img_tensor, (apply_lowlight, rgb_scale),
                                                                  channel_type='rgb')
            img_tensor_lgt = TF.normalize(img_tensor_lgt, [0.5], [0.5])
            lgt_images.append(img_tensor_lgt)


        return {
            "image": images,
            "lgt_image": lgt_images,
            "light_level": light_levels  # 添加光照水平标签（基于RGB通道）
        }
