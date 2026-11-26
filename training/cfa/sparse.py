"""
CFA Pattern处理工具 - GPU优化版本
针对RGGBW马赛克模式的向量化处理函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import torchvision.transforms.functional as TF
from training.cfa.base import CFAAdapter


class CFAPattern:
    """
    CFA Pattern定义类
    定义RGGBW CFA模式的布局
    """

    def __init__(self, pattern_mask: Optional[torch.Tensor] = None):
        """
        CFA Pattern定义类
        定义RGGBW CFA模式的布局

        注意: pattern_mask的形状应为 [4, 4, 4]
            第0维: 通道索引 (0:R, 1:G, 2:B, 3:W)
            第1维: 高度 (4x4块中的行，0-3)
            第2维: 宽度 (4x4块中的列，0-3)
        """
        if pattern_mask is None:
            pattern_mask = CFAPattern.get_rggbw_pattern()

        # 形状断言检查
        assert pattern_mask.dim() == 3, f"pattern_mask必须是3维张量，得到{pattern_mask.dim()}维"
        assert pattern_mask.shape == (4, 4, 4), f"pattern_mask形状必须为(4,4,4)，得到{pattern_mask.shape}"
        assert torch.all(torch.isin(pattern_mask, torch.tensor([0., 1.]))), "pattern_mask必须只包含0和1"

        self.pattern_mask = pattern_mask.float()

    @staticmethod
    def get_rggbw_pattern() -> torch.Tensor:
        """
        获取RGGBW CFA pattern掩码
        返回形状为 [4, 4, 4] 的张量，其中:
        - 第一维: 通道索引 (0:R, 1:G, 2:B, 3:W)
        - 第二维: 高度 (4x4块中的行)
        - 第三维: 宽度 (4x4块中的列)
        """
        pattern = torch.zeros(4, 4, 4, dtype=torch.float32)

        # R通道: 位置(1,0)
        pattern[0, 1, 0] = 1.0

        # G通道: 位置(0,0)和(1,1)
        pattern[1, 0, 0] = 1.0
        pattern[1, 1, 1] = 1.0

        # B通道: 位置(1,0)
        pattern[2, 0, 1] = 1.0

        # W通道: 其余12个位置
        # 创建全1掩码，然后减去RGB位置
        w_mask = torch.ones(4, 4)
        w_mask[0, 1] = 0  # R位置
        w_mask[0, 0] = 0  # G1位置
        w_mask[1, 1] = 0  # G2位置
        w_mask[1, 0] = 0  # B位置
        pattern[3] = w_mask

        return pattern


class CFAPatternReconstructor(nn.Module):
    """
    极致优化的CFA Pattern重构器 - GPU加速版本
    使用卷积和转置卷积操作实现完全向量化的RGGBW马赛克重构

    主要优化：
    1. 完全消除所有Python循环
    2. 使用卷积操作代替手动循环
    3. 预计算所有核和掩码
    4. 支持批量处理和高性能GPU加速
    """

    def __init__(self, pattern_mask: Optional[torch.Tensor] = None):
        super().__init__()

        if pattern_mask is None:
            pattern_mask = CFAPattern.get_rggbw_pattern()

        # 验证pattern_mask
        assert pattern_mask.dim() == 3, f"pattern_mask必须是3维张量，得到{pattern_mask.dim()}维"
        assert pattern_mask.shape == (4, 4, 4), f"pattern_mask形状必须为(4,4,4)，得到{pattern_mask.shape}"

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
        r_mask = self.pattern_mask[0]  # [4, 4]
        g_mask = self.pattern_mask[1]  # [4, 4]
        b_mask = self.pattern_mask[2]  # [4, 4]
        w_mask = self.pattern_mask[3]  # [4, 4]

        # ========== 1. 计算平均值所需的卷积核 ==========

        # R通道卷积核 [1, 1, 4, 4]
        r_kernel = r_mask / r_mask.sum().clamp_min(1e-5)
        self.register_buffer('r_kernel', r_kernel.view(1, 1, 4, 4))

        # G通道有两个位置，需要分别处理
        g_positions = torch.nonzero(g_mask)
        if len(g_positions) != 2:
            raise ValueError("G通道应该有2个已知像素位置")

        # G1通道卷积核 (位置0,0)
        g1_kernel = torch.zeros_like(g_mask)
        g1_kernel[g_positions[0, 0], g_positions[0, 1]] = 1.0
        self.register_buffer('g1_kernel', g1_kernel.view(1, 1, 4, 4))

        # G2通道卷积核 (位置1,1)
        g2_kernel = torch.zeros_like(g_mask)
        g2_kernel[g_positions[1, 0], g_positions[1, 1]] = 1.0
        self.register_buffer('g2_kernel', g2_kernel.view(1, 1, 4, 4))

        # B通道卷积核
        b_kernel = b_mask / b_mask.sum().clamp_min(1e-5)
        self.register_buffer('b_kernel', b_kernel.view(1, 1, 4, 4))

        # print(f"r_mask: {r_mask}")
        print(f"self.r_kernel: {self.r_kernel}")
        print(f"self.g1_kernel: {self.g1_kernel}")
        print(f"self.g2_kernel: {self.g2_kernel}")
        print(f"self.b_kernel: {self.b_kernel}")


        # ========== 2. W通道掩码核 ==========
        # W通道需要在整个4x4块上应用掩码
        self.register_buffer('w_mask_kernel', w_mask.view(1, 1, 4, 4))

        # 用于生成完整W掩码的核
        self.register_buffer('w_full_mask_kernel', w_mask.view(1, 1, 4, 4))

        # ========== 3. 验证掩码 ==========
        self._validate_masks(r_mask, g_mask, b_mask, w_mask)

    def _create_upsample_layers(self):
        """创建上采样层（转置卷积）"""

        self.upsample_r = nn.ConvTranspose2d(
            1, 1, kernel_size=4, stride=4, bias=False
        )
        self.upsample_g1 = nn.ConvTranspose2d(
            1, 1, kernel_size=4, stride=4, bias=False
        )
        self.upsample_g2 = nn.ConvTranspose2d(
            1, 1, kernel_size=4, stride=4, bias=False
        )
        self.upsample_b = nn.ConvTranspose2d(
            1, 1, kernel_size=4, stride=4, bias=False
        )

        # 初始化上采样核为全1（固定，不训练）
        with torch.no_grad():
            for upsample_layer in [self.upsample_r, self.upsample_g1,
                                   self.upsample_g2, self.upsample_b]:
                upsample_layer.weight.fill_(1.0)
                upsample_layer.weight.requires_grad = False

        # 创建用于生成完整W掩码的转置卷积
        self.upsample_w_mask = nn.ConvTranspose2d(
            1, 1, kernel_size=4, stride=4, bias=False
        )
        with torch.no_grad():
            self.upsample_w_mask.weight.fill_(1.0)
            self.upsample_w_mask.weight.requires_grad = False

    def _validate_masks(self, r_mask, g_mask, b_mask, w_mask):
        """验证掩码的正确性"""
        # 确保掩码是二值的
        for name, mask in [('R', r_mask), ('G', g_mask),
                           ('B', b_mask), ('W', w_mask)]:
            unique_vals = torch.unique(mask)
            if not torch.all(torch.isin(unique_vals, torch.tensor([0., 1.]))):
                raise ValueError(f"{name}掩码应该只包含0和1")

        # 确保没有重叠
        total_mask = r_mask + g_mask + b_mask + w_mask
        if torch.any(total_mask > 1):
            raise ValueError("掩码存在重叠")

        # 确保覆盖整个4x4块
        if not torch.allclose(total_mask, torch.ones_like(total_mask)):
            raise ValueError("掩码没有完全覆盖4x4块")

        # 打印统计信息
        if torch.is_grad_enabled():  # 只在训练时打印
            print(f"CFA Pattern验证通过:")
            print(f"  R像素数: {r_mask.sum().item()}")
            print(f"  G像素数: {g_mask.sum().item()}")
            print(f"  B像素数: {b_mask.sum().item()}")
            print(f"  W像素数: {w_mask.sum().item()}")
            print(f"  总像素数: {total_mask.sum().item()}/16")

    def _validate_config(self):
        """验证配置"""
        # 检查所有buffer都已正确注册
        required_buffers = ['r_kernel', 'g1_kernel', 'g2_kernel',
                            'b_kernel', 'w_mask_kernel', 'w_full_mask_kernel']
        for buf_name in required_buffers:
            if not hasattr(self, buf_name):
                raise AttributeError(f"缺少必需的buffer: {buf_name}")

        # 检查上采样层
        required_layers = ['upsample_r', 'upsample_g1', 'upsample_g2',
                           'upsample_b', 'upsample_w_mask']
        for layer_name in required_layers:
            if not hasattr(self, layer_name):
                raise AttributeError(f"缺少必需的上采样层: {layer_name}")

    def forward(self, mosaic: torch.Tensor) -> torch.Tensor:
        """
        向量化的前向传播 - 使用卷积操作

        Args:
            mosaic: 稀疏RGGBW马赛克，形状为 [B, 4, H, W]
                    通道顺序: [R, G, B, W]

        Returns:
            5通道稠密表示，形状为 [B, 5, H, W]
            通道顺序: [R_dense, G1_dense, G2_dense, B_dense, W_dense]
        """
        B, C, H, W = mosaic.shape

        # 验证输入
        assert C == 4, f"输入应该有4个通道，但得到{C}"
        assert H % 4 == 0 and W % 4 == 0, \
            f"高度和宽度应该是4的倍数，但得到{H}x{W}"

        # ========== 1. 分离各通道 ==========
        r_channel = mosaic[:, 0:1]  # [B, 1, H, W]
        g_channel = mosaic[:, 1:2]  # [B, 1, H, W]
        b_channel = mosaic[:, 2:3]  # [B, 1, H, W]
        w_channel = mosaic[:, 3:4]  # [B, 1, H, W]

        # ========== 2. 计算每个4x4块的平均值 ==========
        # 使用stride=4卷积，每个4x4块输出一个值

        # R通道
        r_avg = F.conv2d(
            r_channel, self.r_kernel,
            stride=4, padding=0
        )

        # G1通道平均值 (位置0,0)
        g1_avg = F.conv2d(
            g_channel, self.g1_kernel,
            stride=4, padding=0
        )

        # G2通道
        g2_avg = F.conv2d(
            g_channel, self.g2_kernel,
            stride=4, padding=0
        )

        # B通道
        b_avg = F.conv2d(
            b_channel, self.b_kernel,
            stride=4, padding=0
        )

        # ========== 3. 上采样到原始大小 ==========
        # 使用转置卷积将每个块的平均值扩展到整个4x4块

        r_dense = self.upsample_r(r_avg)  # [B, 1, H, W]
        g1_dense = self.upsample_g1(g1_avg)  # [B, 1, H, W]
        g2_dense = self.upsample_g2(g2_avg)  # [B, 1, H, W]
        b_dense = self.upsample_b(b_avg)  # [B, 1, H, W]

        # ========== 4. 处理W通道 ==========
        # W通道需要保持原始值，但在RGB位置补0

        # 首先创建完整的W掩码 [B, 1, H, W]
        # 通过转置卷积将4x4的W掩码扩展到整个图像
        w_mask_full = self._create_w_mask(B, H, W, w_channel.device)
        # print(f"w_mask_full:\n{w_mask_full}")

        # 应用掩码：只在W位置保留原始值
        w_dense = w_channel * w_mask_full

        # ========== 5. 堆叠所有通道 ==========
        dense_output = torch.cat([
            r_dense, g1_dense, g2_dense, b_dense, w_dense
        ], dim=1)  # [B, 5, H, W]

        return dense_output

    def reconstruct_dense(self, mosaic: torch.Tensor) -> torch.Tensor:
        """
        重构为稠密表示（向后兼容接口）

        Args:
            mosaic: 稀疏RGGBW马赛克，形状为 [B, 4, H, W]

        Returns:
            5通道稠密表示，形状为 [B, 5, H, W]
        """
        return self.forward(mosaic)

    def create_sparse_mosaic(self, rgb_image: torch.Tensor,
                             noise_level: float = 0.0) -> torch.Tensor:
        """
        从RGB图像创建稀疏RGGBW马赛克（向量化版本）

        Args:
            rgb_image: RGB图像，形状为 [B, 3, H, W]，值范围[-1, 1]或[0, 1]
            noise_level: 添加的高斯噪声标准差

        Returns:
            稀疏RGGBW马赛克，形状为 [B, 4, H, W]
        """
        B, C, H, W = rgb_image.shape
        assert C == 3, f"输入应该是RGB图像，但得到{C}通道"
        assert H % 4 == 0 and W % 4 == 0, \
            f"高度和宽度应该是4的倍数，但得到{H}x{W}"

        device = rgb_image.device

        # 确保图像在[0, 1]范围
        if rgb_image.min() < -0.5:  # 假设是[-1, 1]范围
            rgb_image = (rgb_image + 1) / 2

        # 初始化马赛克
        mosaic = torch.zeros(B, 4, H, W, device=device, dtype=rgb_image.dtype)

        # ========== 向量化创建稀疏马赛克 ==========

        # 1. 分离RGB通道
        r_channel = rgb_image[:, 0:1]  # [B, 1, H, W]
        g_channel = rgb_image[:, 1:2]  # [B, 1, H, W]
        b_channel = rgb_image[:, 2:3]  # [B, 1, H, W]

        # 2. 创建RGB掩码
        r_mask_full = self._create_channel_mask('R', H, W, device).repeat(B, 1, 1, 1)
        g_mask_full = self._create_channel_mask('G', H, W, device).repeat(B, 1, 1, 1)
        b_mask_full = self._create_channel_mask('B', H, W, device).repeat(B, 1, 1, 1)
        w_mask_full = self._create_channel_mask('W', H, W, device).repeat(B, 1, 1, 1)

        # 3. 应用掩码创建稀疏通道
        mosaic[:, 0:1] = r_channel * r_mask_full  # R通道
        mosaic[:, 1:2] = g_channel * g_mask_full  # G通道
        mosaic[:, 2:3] = b_channel * b_mask_full  # B通道

        # 4. W通道：RGB加权平均
        # 权重: R:0.299, G:0.587, B:0.114
        w_value = (0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel)
        mosaic[:, 3:4] = w_value * w_mask_full

        # 5. 添加噪声（可选）
        if noise_level > 0:
            noise = torch.randn_like(mosaic) * noise_level
            mosaic = mosaic + noise
            mosaic = torch.clamp(mosaic, 0, 1)

        return mosaic

    def dense_to_sparse(self, dense: torch.Tensor) -> torch.Tensor:
        """
        将5通道稠密表示转换回稀疏表示（向量化版本）

        Args:
            dense: 5通道稠密表示 [B, 5, H, W]

        Returns:
            稀疏RGGBW马赛克 [B, 4, H, W]
        """
        B, C, H, W = dense.shape
        assert C == 5, f"输入应该有5个通道，但得到{C}"
        assert H % 4 == 0 and W % 4 == 0

        device = dense.device

        # 创建通道掩码
        r_mask_full = self._create_channel_mask('R', H, W, device).repeat(B, 1, 1, 1)
        g_mask_full = self._create_channel_mask('G', H, W, device).repeat(B, 1, 1, 1)
        b_mask_full = self._create_channel_mask('B', H, W, device).repeat(B, 1, 1, 1)
        w_mask_full = self._create_channel_mask('W', H, W, device).repeat(B, 1, 1, 1)

        # 初始化稀疏马赛克
        sparse_mosaic = torch.zeros(B, 4, H, W, device=device)

        # 应用掩码提取稀疏值
        # R通道
        sparse_mosaic[:, 0:1] = dense[:, 0:1] * r_mask_full

        # G通道：需要合并G1和G2
        # 注意：G1和G2在原始G掩码的不同位置
        g1_mask = self._create_specific_g_mask('G1', H, W, device).repeat(B, 1, 1, 1)
        g2_mask = self._create_specific_g_mask('G2', H, W, device).repeat(B, 1, 1, 1)

        g_sparse = (dense[:, 1:2] * g1_mask +
                    dense[:, 2:3] * g2_mask)
        sparse_mosaic[:, 1:2] = g_sparse

        # B通道
        sparse_mosaic[:, 2:3] = dense[:, 3:4] * b_mask_full

        # W通道
        sparse_mosaic[:, 3:4] = dense[:, 4:5] * w_mask_full

        return sparse_mosaic

    def _create_w_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        创建W通道的全掩码 - 使用重复而不是插值

        Args:
            B: 批量大小
            H: 高度
            W: 宽度
            device: 设备

        Returns:
            W通道掩码，形状为 [B, 1, H, W]
        """
        # 获取W通道的4x4基础掩码
        w_base_mask = self.pattern_mask[3]  # [4, 4]

        # 使用 repeat 操作在空间维度上重复基础掩码
        h_repeats = H // 4
        w_repeats = W // 4

        # 创建完整掩码 [1, 1, H, W]
        w_mask_full = w_base_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        w_mask_full = w_mask_full.repeat(1, 1, h_repeats, w_repeats)  # [1, 1, H, W]

        # 扩展批量维度
        w_mask_full = w_mask_full.repeat(B, 1, 1, 1)  # [B, 1, H, W]

        return w_mask_full.to(device)

    def _create_channel_mask(self, channel: str, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        创建指定通道的掩码 - 使用重复而不是插值

        Args:
            channel: 通道名称 ('R', 'G', 'B', 'W')
            H: 高度
            W: 宽度
            device: 设备

        Returns:
            通道掩码，形状为 [1, 1, H, W]
        """
        # 通道索引映射
        channel_map = {'R': 0, 'G': 1, 'B': 2, 'W': 3}

        if channel not in channel_map:
            raise ValueError(f"无效的通道名称: {channel}，应为 'R', 'G', 'B' 或 'W'")

        # 获取4x4的通道掩码
        channel_idx = channel_map[channel]
        base_mask = self.pattern_mask[channel_idx]  # [4, 4]

        # 使用 repeat 操作在空间维度上重复基础掩码
        # 注意：H 和 W 必须是 4 的倍数
        h_repeats = H // 4
        w_repeats = W // 4

        # 方法1: 使用 torch.tile (推荐，更直观)
        mask_full = base_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        mask_full = mask_full.repeat(1, 1, h_repeats, w_repeats)  # [1, 1, H, W]

        # 方法2: 使用 repeat_interleave (也可以)
        # mask_full = base_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        # mask_full = mask_full.repeat_interleave(h_repeats, dim=2).repeat_interleave(w_repeats, dim=3)

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

        # 获取对应的4x4掩码核
        if g_type == 'G1':
            base_mask = self.g1_kernel.squeeze()  # [4, 4]
        else:  # G2
            base_mask = self.g2_kernel.squeeze()  # [4, 4]

        # 使用 repeat 操作在空间维度上重复基础掩码
        h_repeats = H // 4
        w_repeats = W // 4

        # 创建完整掩码 [1, 1, H, W]
        mask_full = base_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        mask_full = mask_full.repeat(1, 1, h_repeats, w_repeats)  # [1, 1, H, W]

        return mask_full.to(device)

    def verify_module_accuracy(self) -> Dict[str, Any]:
        """
        验证模块的重构准确性

        Returns:
            包含验证结果的字典，包括：
            - 'passed': 是否通过所有测试
            - 'tests': 各个测试的结果详情
            - 'errors': 各测试的最大误差
            - 'summary': 测试总结
        """
        results = {
            'passed': True,
            'tests': {},
            'errors': {},
            'summary': ''
        }

        print("=" * 50)
        print("CFA处理器准确性验证")
        print("=" * 50)

        # 创建测试设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print(f"使用设备: {device}")

        try:
            # 创建一个1×4×8×8的马赛克图块
            B, C, H, W = 1, 4, 8, 8
            mosaic = torch.zeros(B, C, H, W, device=device)

            # 根据CFA模式填充值
            pattern = CFAPattern.get_rggbw_pattern()

            # 遍历每个4x4块
            for i in range(0, H, 4):
                for j in range(0, W, 4):
                    # 在每个4x4块中设置CFA模式
                    for c in range(4):
                        for y in range(4):
                            for x in range(4):
                                if pattern[c, y, x] == 1:
                                    # 设置唯一值以便识别
                                    value = c * 10 + y * 2 + x + 1
                                    mosaic[0, c, i + y, j + x] = value

            print("\n原始稀疏马赛克 (形状: 1×4×8×8):")
            print("通道0 (R通道):")
            print(mosaic[0, 0].cpu().numpy())
            print("\n通道1 (G通道):")
            print(mosaic[0, 1].cpu().numpy())
            print("\n通道2 (B通道):")
            print(mosaic[0, 2].cpu().numpy())
            print("\n通道3 (W通道):")
            print(mosaic[0, 3].cpu().numpy())

            # 测试1: 重构为稠密表示
            print("\n重构后的稠密表示 (形状: 1×5×8×8):")
            dense = self.forward(mosaic)

            print("通道0 (R稠密):")
            print(dense[0, 0].cpu().detach().numpy())
            print("\n通道1 (G1稠密):")
            print(dense[0, 1].cpu().detach().numpy())
            print("\n通道2 (G2稠密):")
            print(dense[0, 2].cpu().detach().numpy())
            print("\n通道3 (B稠密):")
            print(dense[0, 3].cpu().detach().numpy())
            print("\n通道4 (W稠密):")
            print(dense[0, 4].cpu().detach().numpy())

            # 验证稠密表示的正确性
            r_dense = dense[0, 0]
            g1_dense = dense[0, 1]
            g2_dense = dense[0, 2]
            b_dense = dense[0, 3]
            w_dense = dense[0, 4]

            # 检查每个4x4块
            test_passed = True
            for i in range(0, H, 4):
                for j in range(0, W, 4):
                    # R通道：应该在整个4x4块上是常数
                    r_block = r_dense[i:i + 4, j:j + 4]
                    r_value = mosaic[0, 0, i, j + 1]  # R在(0,1)位置
                    if not torch.allclose(r_block, torch.full_like(r_block, r_value), rtol=1e-5):
                        print(f"R通道在块({i},{j})不一致")
                        test_passed = False

                    # G1通道：应该在整个4x4块上是常数
                    g1_block = g1_dense[i:i + 4, j:j + 4]
                    g1_value = mosaic[0, 1, i, j]  # G1在(0,0)位置
                    if not torch.allclose(g1_block, torch.full_like(g1_block, g1_value), rtol=1e-5):
                        print(f"G1通道在块({i},{j})不一致")
                        test_passed = False

                    # G2通道：应该在整个4x4块上是常数
                    g2_block = g2_dense[i:i + 4, j:j + 4]
                    g2_value = mosaic[0, 1, i + 1, j + 1]  # G2在(1,1)位置
                    if not torch.allclose(g2_block, torch.full_like(g2_block, g2_value), rtol=1e-5):
                        print(f"G2通道在块({i},{j})不一致")
                        test_passed = False

                    # B通道：应该在整个4x4块上是常数
                    b_block = b_dense[i:i + 4, j:j + 4]
                    b_value = mosaic[0, 2, i + 1, j]  # B在(1,0)位置
                    if not torch.allclose(b_block, torch.full_like(b_block, b_value), rtol=1e-5):
                        print(f"B通道在块({i},{j})不一致")
                        test_passed = False

                    # W通道：应该保持原始值
                    w_block = w_dense[i:i + 4, j:j + 4]
                    w_original = mosaic[0, 3, i:i + 4, j:j + 4]
                    if not torch.allclose(w_block, w_original, rtol=1e-5):
                        print(f"W通道在块({i},{j})不一致")
                        test_passed = False

            results['tests']['reconstruction_accuracy'] = '通过' if test_passed else '失败'
            results['errors']['reconstruction_accuracy'] = 0.0

            # 验证重构器参数不参与训练
            print("\n验证参数不参与训练:")
            param_trainable = False
            for name, param in self.named_parameters():
                print(f"{name}: requires_grad = {param.requires_grad}")
                if param.requires_grad:
                    print(f"警告: {name} 参数参与训练!")
                    param_trainable = True

            results['tests']['parameter_fixed'] = '通过' if not param_trainable else '失败'
            results['errors']['parameter_fixed'] = 1.0 if param_trainable else 0.0

            # 生成总结
            passed_tests = sum(1 for test in results['tests'].values() if test == '通过')
            total_tests = len(results['tests'])

            results['summary'] = f"通过 {passed_tests}/{total_tests} 个测试"
            results['passed'] = (passed_tests == total_tests)

            print("\n" + "=" * 50)
            print("验证完成!")
            print(f"结果: {results['summary']}")
            print(f"所有测试通过: {results['passed']}")

            return results

        except Exception as e:
            print(f"验证失败，错误: {e}")
            import traceback
            traceback.print_exc()

            results['passed'] = False
            results['summary'] = f"验证失败: {str(e)}"
            return results


def create_rggbw_condition(cfa_image, cfa_w_image):
    """
    创建4通道稀疏RGGBW条件输入

    返回4通道 [R, G, B, W] 稀疏马赛克

    注意：cfa_image已经是马赛克RGB（3通道，已知位置有值）
          cfa_w_image是马赛克W（1通道，W位置有值）
    """
    # 确保cfa_w_image是单通道
    if cfa_w_image.shape[1] == 3:
        cfa_w_image = cfa_w_image.mean(dim=1, keepdim=True)

    # 直接拼接为4通道
    condition = torch.cat([
        cfa_image,  # [B, 3, H, W] - 稀疏RGB
        cfa_w_image  # [B, 1, H, W] - 稀疏W
    ], dim=1)  # 总共4通道

    return condition


class SparseAdapter(CFAAdapter):
    mode = 'sparse'
    condition_channels = 4
    encoder_channels = 5
    tile_size = 4
    required_columns = ('image', 'cfa', 'cfa_w')
    batch_columns = ('image', 'cfa', 'cfa_w')
    debug_checks = True
    samples_per_light = 2
    lowlight_apply_constraint = True

    def pattern_mask(self):
        return CFAPattern().pattern_mask

    def create_reconstructor(self, pattern_mask=None):
        return CFAPatternReconstructor(pattern_mask)

    def make_conditions(self, batch, dtype=torch.float32):
        return create_rggbw_condition(batch['cfa'].to(dtype), batch['cfa_w'].to(dtype))

    def visualization_inputs(self, batch, device):
        cfa = batch['cfa'].to(device)
        w = batch['cfa_w'].to(device)
        return create_rggbw_condition(cfa, w), cfa, w

    def after_model_init(self, model):
        results = model.cfa_reconstructor.reconstructor.verify_module_accuracy()
        if results['passed']:
            print('模块准确性验证通过！')
        else:
            print('模块准确性验证失败！')
            print('详细结果:', results)

    def preprocess(self, examples, *, resolution, lowlight_simulator, base_transform):
        self.validate_images(examples, resolution)
        # 1. 处理目标图像 (image): 正常光照，直接变换
        images = [base_transform(image.convert("RGB")) for image in examples["image"]]

        # 2. 处理输入图像 (cfa, cfa_w): 确保成对状态一致
        cfa_images = []
        cfa_w_images = []

        # 3. 添加光照水平标签
        light_levels = []  # 记录每个样本的光照水平

        for cfa_img, cfa_w_img in zip(examples["cfa"], examples["cfa_w"]):
            # 【关键步骤】：为这一对 (cfa, cfa_w) 采样【同一组】随机参数
            apply_lowlight, rgb_scale = lowlight_simulator._sample_parameters()

            # 记录RGB通道的光照水平
            if not apply_lowlight:
                light_levels.append(1.0)  # 正常光照
            else:
                light_levels.append(rgb_scale)  # RGB通道的光照水平

            # 对CFA (RGB) 应用这组参数
            cfa_tensor = TF.to_tensor(cfa_img.convert("RGB"))
            cfa_tensor = lowlight_simulator.apply_with_parameters(cfa_tensor, (apply_lowlight, rgb_scale),
                                                                  channel_type='rgb')
            cfa_tensor = TF.normalize(cfa_tensor, [0.5], [0.5])
            cfa_images.append(cfa_tensor)

            # 对CFA_W (W通道) 应用【同一组】参数，但W通道的降光系数不同
            cfa_w_tensor = TF.to_tensor(cfa_w_img.convert("L"))
            cfa_w_tensor = lowlight_simulator.apply_with_parameters(cfa_w_tensor, (apply_lowlight, rgb_scale),
                                                                    channel_type='w')
            cfa_w_tensor = TF.normalize(cfa_w_tensor, [0.5], [0.5])
            cfa_w_images.append(cfa_w_tensor)

        return {
            "image": images,
            "cfa": cfa_images,
            "cfa_w": cfa_w_images,
            "light_level": light_levels  # 添加光照水平标签（基于RGB通道）
        }
