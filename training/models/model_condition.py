"""
Sparse/Bayer 共享条件扩散模型；CFA 重构器与输入通道显式选择
基于ControlNet思想，使用预训练JiT作为骨干网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.models.model_jit import JiT_models
from training.cfa import get_cfa_adapter


class CFAPatternReconstructorModule(nn.Module):
    """
    CFA Pattern重构模块（神经网络模块版本）
    选择对应 CFA 重构器：Sparse 4→5 通道，Bayer 3→4 通道
    """

    def __init__(self, cfa_mode, pattern_mask=None):
        super().__init__()
        # 在初始化时就创建重构器并缓存
        self.reconstructor = get_cfa_adapter(cfa_mode).create_reconstructor(pattern_mask)

    def forward(self, mosaic):
        """
        按模型的 CFA 模式重构稀疏马赛克

        Args:
            mosaic: [B, C, H, W]，Sparse C=4，Bayer C=3

        Returns:
            [B, C_dense, H, W]，Sparse C_dense=5，Bayer C_dense=4
        """
        # 直接将重构器移动到输入设备（如果尚未移动）
        if self.reconstructor.pattern_mask.device != mosaic.device:
            self.reconstructor = self.reconstructor.to(mosaic.device)

        return self.reconstructor(mosaic)


class ResDownBlock(nn.Module):
    """带残差连接和 SE 注意力的下采样块"""

    def __init__(self, in_ch, out_ch, stride=2, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

        # Skip connection：如果维度不匹配，用 1×1 卷积投影
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.GroupNorm(8, out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

        # SE 注意力：显式建模通道重要性
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch // 16, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch // 16, out_ch, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv(x)

        if self.use_se:
            x = x * self.se(x)  # 通道加权

        return x + residual  # 残差连接


class MultiScaleConditionEncoder(nn.Module):
    """
    多尺度条件编码器
    输入: CFA 重构后的图像；Sparse 5 通道，Bayer 4 通道
    输出: 浅层条件token [B, N, C] + 深层条件token [B, N, C]
    """

    def __init__(self, in_channels=5, hidden_size=768, patch_size=16, img_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.img_size = img_size

        # 计算patch数量
        self.num_patches = (img_size // patch_size) ** 2

        # 1. 共享浅层特征提取 (2层CNN)
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )

        # 2. 浅层分支 (纹理/边缘特征)
        self.shallow_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True)
        )

        # 3. 深层分支 (语义/全局特征)
        self.deep_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(inplace=True)
        )

        # 4. 分块嵌入投影 (将特征图投影到hidden_size)
        # 仅针对特定 patch_size 16
        self.shallow_patch_embed = nn.Sequential(
            ResDownBlock(128, 256, stride=2),
            ResDownBlock(256, 384, stride=2),
            ResDownBlock(384, hidden_size // 2, stride=2),  # 256
            ResDownBlock(hidden_size // 2, hidden_size, stride=2, use_se=False),  # 最后层可不加 SE
        )

        self.deep_patch_embed = nn.Sequential(
            ResDownBlock(512, hidden_size // 2, stride=2),  # 256
            ResDownBlock(hidden_size // 2, hidden_size, stride=2),
            ResDownBlock(hidden_size, hidden_size, stride=2, use_se=False),
            ResDownBlock(hidden_size, hidden_size, stride=2, use_se=False),
        )

        # 5. 可学习的位置编码（与JiT的位置编码对齐）
        self.shallow_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        self.deep_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        # 位置编码初始化（使用正态分布）
        nn.init.normal_(self.shallow_pos_embed, std=0.02)
        nn.init.normal_(self.deep_pos_embed, std=0.02)

        # 卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, in_channels, H, W]，由 CFA 模式选择输入通道

        Returns:
            shallow_tokens: 浅层条件token [B, N, C]
            deep_tokens: 深层条件token [B, N, C]
        """
        B, C, H, W = x.shape

        # 1. 共享浅层特征提取
        shared_features = self.shared_encoder(x)  # [B, 32, H, W]

        # 2. 双分支特征提取
        shallow_features = self.shallow_branch(shared_features)  # [B, 128, H, W]
        deep_features = self.deep_branch(shared_features)        # [B, 512, H, W]

        # 3. 分块嵌入投影
        # 展平为token序列: [B, C, H, W] -> [B, N, hidden_size]
        shallow_tokens = self.shallow_patch_embed(shallow_features)  # [B, hidden_size, H/patch, W/patch]
        shallow_tokens = shallow_tokens.flatten(2).transpose(1, 2)   # [B, N, hidden_size]

        deep_tokens = self.deep_patch_embed(deep_features)          # [B, hidden_size, H/patch, W/patch]
        deep_tokens = deep_tokens.flatten(2).transpose(1, 2)        # [B, N, hidden_size]

        # 4. 添加位置编码
        shallow_tokens = shallow_tokens + self.shallow_pos_embed
        deep_tokens = deep_tokens + self.deep_pos_embed

        return shallow_tokens, deep_tokens


class AdaptiveGateInjection(nn.Module):
    """
    自适应门控注入模块
    实现多头门控机制，控制条件信息注入强度
    """

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size必须能被num_heads整除"

        # 多头门控网络
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.head_dim, 4 * self.head_dim),
                nn.SiLU(inplace=True),
                nn.Linear(4 * self.head_dim, self.head_dim),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])

        # 融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

        # 门控值的缩放因子（可学习，初始为0）
        self.gate_scale = nn.Parameter(torch.zeros(1))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重 - 零中心初始化"""
        for gate_net in self.gate_nets:
            # 第一层线性层
            nn.init.kaiming_normal_(gate_net[0].weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(gate_net[0].bias)

            # 第二层线性层
            nn.init.kaiming_normal_(gate_net[2].weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(gate_net[2].bias)  # 零初始化bias，使初始门控值接近0.5

    def forward(self, main_feat, cond_feat, num_patch_tokens=None):
        """
        自适应门控融合

        Args:
            main_feat: 主特征 [B, L, C]（JiT隐藏状态，可能包含in-context tokens）
            cond_feat: 条件特征 [B, N, C]（条件token，应该只与patch tokens对齐）
            num_patch_tokens: patch tokens的数量（可选，如果不提供则假设cond_feat长度就是patch tokens数量）

        Returns:
            注入条件后的特征 [B, L, C]
        """
        B, L, C = main_feat.shape
        B_cond, N, C_cond = cond_feat.shape

        assert C == C_cond, f"特征维度不匹配: main_feat {C} != cond_feat {C_cond}"

        # 如果提供了num_patch_tokens，使用它；否则假设cond_feat长度就是patch tokens数量
        if num_patch_tokens is None:
            num_patch_tokens = N

        # 检查序列长度关系
        if L == num_patch_tokens:
            # 情况1: main_feat只包含patch tokens，没有in-context tokens
            # 直接对全部token进行注入
            gated_cond_feat = self._compute_gated_cond(main_feat, cond_feat)
            output = main_feat + self.gate_scale * gated_cond_feat
            return output

        elif L > num_patch_tokens:
            # 情况2: main_feat包含in-context tokens + patch tokens
            # 计算in-context tokens的数量
            num_in_context = L - num_patch_tokens

            # 分割特征
            in_context_feat = main_feat[:, :num_in_context, :]  # [B, num_in_context, C]
            patch_feat = main_feat[:, num_in_context:, :]  # [B, num_patch_tokens, C]

            # 验证条件特征与patch tokens对齐
            assert patch_feat.shape[1] == cond_feat.shape[1], \
                f"条件特征序列长度({cond_feat.shape[1]})应与patch tokens数量({patch_feat.shape[1]})一致"

            # 只对patch tokens进行条件注入
            gated_cond_feat = self._compute_gated_cond(patch_feat, cond_feat)
            assert patch_feat.shape[2] == gated_cond_feat.shape[2], \
                f"条件特征序列隐藏维度({gated_cond_feat.shape[2]})应与patch tokens隐藏维度({patch_feat.shape[2]})一致"
            injected_patch = patch_feat + self.gate_scale * gated_cond_feat

            # 重新拼接：in-context tokens不变 + 注入后的patch tokens
            output = torch.cat([in_context_feat, injected_patch], dim=1)
            return output

        else:
            # 情况3: main_feat序列长度小于patch tokens数量（不应该发生）
            raise ValueError(
                f"主特征序列长度({L})小于patch tokens数量({num_patch_tokens})。"
                f"这可能意味着in-context tokens处理有问题。"
            )

    def _compute_gated_cond(self, main_feat, cond_feat):
        """计算门控条件特征"""
        B, N, C = main_feat.shape

        # 1. 分割为多头
        main_head = main_feat.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        cond_head = cond_feat.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]

        # 2. 多头门控计算
        all_gates = []
        for i in range(self.num_heads):
            # 获取当前头的特征
            main_head_i = main_head[:, i, :, :]  # [B, N, head_dim]
            cond_head_i = cond_head[:, i, :, :]  # [B, N, head_dim]

            # 拼接特征
            combined = torch.cat([main_head_i, cond_head_i], dim=-1)  # [B, N, 2*head_dim]

            # 通过门控网络
            gate_i = self.gate_nets[i](combined)  # [B, N, head_dim]
            all_gates.append(gate_i)

        # 3. 加权融合多头门控
        gates = torch.stack(all_gates, dim=1)  # [B, num_heads, N, head_dim]

        # 应用融合权重（softmax归一化）
        weights = F.softmax(self.fusion_weights, dim=0)  # [num_heads]

        # 加权求和
        weighted_gates = torch.zeros_like(gates[:, 0, :, :])  # [B, N, head_dim]
        for i in range(self.num_heads):
            weighted_gates = weighted_gates + weights[i] * gates[:, i, :, :]

        # 4. 扩展加权门控到完整维度
        # 将 [B, N, head_dim] 扩展为 [B, N, num_heads * head_dim]
        weighted_gates_expanded = weighted_gates.unsqueeze(2).repeat(1, 1, self.num_heads,
                                                                     1)  # [B, N, num_heads, head_dim]
        weighted_gates_expanded = weighted_gates_expanded.view(B, N, C)  # [B, N, C]

        # 5. 应用门控到条件特征
        gated_cond = weighted_gates_expanded * cond_feat
        gated_cond = self.dropout(gated_cond)

        return gated_cond


class ConditionJiT(nn.Module):
    """
    完整条件JiT模型：实现分层选择性注入
    支持多尺度条件编码和自适应门控注入
    """

    def __init__(self, pretrained_model_name='JiT-B/16', img_size=256, num_classes=1000, *, cfa_mode):
        super().__init__()
        adapter = get_cfa_adapter(cfa_mode)
        self.cfa_mode = adapter.mode
        self.debug_checks = adapter.debug_checks

        # 加载预训练JiT模型
        self.jit_model = JiT_models[pretrained_model_name](
            input_size=img_size,
            in_channels=3,
            num_classes=num_classes
        )

        # 获取模型配置
        self.hidden_size = self.jit_model.hidden_size
        self.num_patches = self.jit_model.x_embedder.num_patches
        self.in_context_len = self.jit_model.in_context_len
        self.in_context_start = self.jit_model.in_context_start
        self.num_blocks = len(self.jit_model.blocks)
        self.patch_size = self.jit_model.patch_size

        print(f"加载预训练模型: {pretrained_model_name}")
        print(f"hidden_size: {self.hidden_size}")
        print(f"num_patches: {self.num_patches}")
        print(f"in_context_len: {self.in_context_len}")
        print(f"in_context_start: {self.in_context_start}")
        print(f"num_blocks: {self.num_blocks}")
        print(f"patch_size: {self.patch_size}")

        # 1. CFA pattern重构模块
        self.cfa_reconstructor = CFAPatternReconstructorModule(cfa_mode)

        # 2. 多尺度条件编码器
        self.cond_encoder = MultiScaleConditionEncoder(
            in_channels=adapter.encoder_channels,
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            img_size=img_size
        )

        # 3. 定义分层注入策略
        # 按照设计文档选择关键层进行注入
        self.injection_layers = nn.ModuleDict()
        self.injection_types = {}  # 记录每层的注入类型

        # 计算可用的注入层（从in_context_start+1开始，避免RoPE问题）
        available_start = self.in_context_start + 1

        # 获取注入层配置
        self.injection_indices, self.injection_types = self._get_injection_config(
            available_start, self.num_blocks
        )

        print(f"注入层索引: {self.injection_indices}")

        # 为每个注入层创建自适应门控注入模块
        for layer_idx in self.injection_indices:
            layer_str = str(layer_idx)
            inject_type = self.injection_types[layer_str]
            self.injection_layers[layer_str] = AdaptiveGateInjection(
                hidden_size=self.hidden_size,
                num_heads=4  # 使用4个头
            )

        # 混合注入的权重（用于中层混合注入）
        if any(t == 'mixed' for t in self.injection_types.values()):
            self.mix_weight = nn.Parameter(torch.tensor(0.5))  # 初始权重0.5

        # 4. 门控正则化统计
        self.gate_stats = {'mean': 0.0, 'std': 1.0}

    def _get_injection_config(self, available_start, num_blocks):
        """
        根据模型配置生成注入层索引和类型

        Args:
            available_start: 可用的起始层索引（in_context_start + 1）
            num_blocks: Transformer块总数

        Returns:
            tuple: (injection_indices, injection_types)
        """
        injection_indices = []
        injection_types = {}

        available_layers = num_blocks - available_start

        if available_layers <= 0:
            print(f"警告：没有可用注入层（available_start={available_start}, num_blocks={num_blocks}）")
            return injection_indices, injection_types

        # 根据不同模型深度采用不同的注入策略
        if num_blocks == 12:  # JiT-B系列 (depth=12)
            # 原策略：5,7,9,11，调整为从available_start开始
            base_indices = [5, 7, 9, 11]
            injection_indices = [
                idx for idx in base_indices
                if idx >= available_start and idx < num_blocks
            ]

        elif num_blocks == 24:  # JiT-L系列 (depth=24)
            # 使用相对比例策略：1/4, 1/2, 3/4, 5/6
            ratios = [0.25, 0.5, 0.75, 5 / 6]
            injection_indices = []
            for ratio in ratios:
                idx = available_start + int(ratio * (num_blocks - available_start - 1))
                idx = max(available_start, min(idx, num_blocks - 1))
                if idx not in injection_indices:
                    injection_indices.append(idx)

        elif num_blocks == 32:  # JiT-H系列 (depth=32)
            # 使用更密集的注入策略：1/6, 1/3, 1/2, 2/3, 5/6 (最多5个)
            ratios = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]
            injection_indices = []
            for ratio in ratios:
                idx = available_start + int(ratio * (num_blocks - available_start - 1))
                idx = max(available_start, min(idx, num_blocks - 1))
                if idx not in injection_indices and len(injection_indices) < 5:
                    injection_indices.append(idx)
        else:
            # 通用策略：均匀分布
            step = max(1, available_layers // 4)
            injection_indices = [
                available_start + i * step
                for i in range(min(4, available_layers))
            ]

        # 确保注入点不超过4个（除了JiT-H可以用5个）
        if num_blocks != 32 and len(injection_indices) > 4:
            injection_indices = injection_indices[:4]

        # 如果没有找到合适的注入点，使用备用策略
        if not injection_indices:
            print("警告：没有找到合适的注入点，使用备用策略")
            step = max(1, available_layers // 4)
            injection_indices = [
                available_start + i * step
                for i in range(min(4, available_layers))
            ]

        # 为每个注入层分配类型
        total_injections = len(injection_indices)

        for i, layer_idx in enumerate(injection_indices):
            # 根据注入点位置决定类型
            if total_injections == 4:
                # 4个注入点：浅层、混合、深层、深层
                if i == 0:
                    inject_type = 'shallow'
                elif i == 1:
                    inject_type = 'mixed'
                else:
                    inject_type = 'deep'
            elif total_injections == 5:
                # 5个注入点（JiT-H）：浅层、浅层、混合、深层、深层
                if i < 2:
                    inject_type = 'shallow'
                elif i == 2:
                    inject_type = 'mixed'
                else:
                    inject_type = 'deep'
            else:
                # 其他情况：根据位置比例分配
                position_ratio = i / (total_injections - 1) if total_injections > 1 else 0
                if position_ratio < 0.33:
                    inject_type = 'shallow'
                elif position_ratio < 0.67:
                    inject_type = 'mixed'
                else:
                    inject_type = 'deep'

            injection_types[str(layer_idx)] = inject_type

        return injection_indices, injection_types

    def freeze_pretrained(self):
        """冻结预训练模型的所有参数"""
        for param in self.jit_model.parameters():
            param.requires_grad = False
        print("预训练JiT模型已冻结")

    def unfreeze_pretrained_layers(self, num_layers=8):
        """
        解冻预训练模型的最后几层（第二阶段训练），并确保条件编码器和注入层保持可训练

        Args:
            num_layers: 要解冻的Transformer块数量（从最后开始）
        """
        self.freeze_pretrained()
        total_blocks = len(self.jit_model.blocks)

        # 1. 解冻最后num_layers个Transformer块
        start_idx = max(0, total_blocks - num_layers)
        for i in range(start_idx, total_blocks):
            for param in self.jit_model.blocks[i].parameters():
                param.requires_grad = True

        # 2. 解冻最终层
        for param in self.jit_model.final_layer.parameters():
            param.requires_grad = True

        # 3. 确保条件编码器参数可训练
        for param in self.cond_encoder.parameters():
            param.requires_grad = True

        # 4. 确保注入层参数可训练
        for param in self.injection_layers.parameters():
            param.requires_grad = True

        # 6. 确保混合注入权重可训练（如果存在）
        if hasattr(self, 'mix_weight'):
            self.mix_weight.requires_grad = True

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_percentage = 100 * trainable_params / total_params

        print(f"已解冻最后{num_layers}个Transformer块和最终层")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"可训练比例: {trainable_percentage:.2f}%")

        # 打印各模块可训练状态
        print("\n各模块可训练状态:")
        for name, module in self.named_children():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module_total > 0:
                module_percentage = 100 * module_trainable / module_total
                trainable_status = "可训练" if module_trainable > 0 else "冻结"
                print(
                    f"  {name}: {module_trainable:,}/{module_total:,} ({module_percentage:.1f}%) - {trainable_status}")

    def get_gate_regularization_loss(self):
        """
        计算门控正则化损失
        鼓励门控值在0.5附近，避免极端依赖

        Returns:
            门控正则化损失
        """
        reg_loss = 0.0

        for name, module in self.injection_layers.items():
            # 收集门控网络中的sigmoid输出
            for gate_net in module.gate_nets:
                # 获取最后一层（sigmoid）前的线性层权重
                linear_layer = gate_net[-2]  # 倒数第二层是线性层

                # L1正则化：鼓励门控网络输出接近0.5
                # 通过正则化线性层的权重，间接影响门控值
                reg_loss += torch.abs(linear_layer.weight).mean()
                reg_loss += torch.abs(linear_layer.bias - 0.5).mean()  # bias接近0.5

        # 门控缩放因子的正则化（鼓励接近1）
        for name, module in self.injection_layers.items():
            # 方法1：加上.mean()转换为标量
            reg_loss += torch.abs(module.gate_scale - 1.0).mean()

            # 或者方法2：直接使用.item()转换为Python标量
            # reg_loss += torch.abs(module.gate_scale - 1.0).item()

        return reg_loss

    # def forward(self, x, conditions, t, y):
    #     return self.jit_model(x, t, y)

    def forward(self, x, conditions, t, y):
        """
        前向传播

        Args:
            x: 噪声图像 [B, 3, H, W]
            conditions: [B, C, H, W]，Sparse C=4，Bayer C=3
            t: 时间步 [B,]
            y: 类别标签 [B,]

        Returns:
            去噪后的图像 [B, 3, H, W]
        """
        # 1. CFA pattern重构
        # Sparse: 4→5 通道；Bayer: 3→4 通道
        dense_conditions = self.cfa_reconstructor(conditions)
        if self.debug_checks:
            assert not torch.isnan(dense_conditions).any(), f"CFA重构后出现NaN: {torch.isnan(dense_conditions).sum()}"

        # 2. 多尺度条件编码
        shallow_cond, deep_cond = self.cond_encoder(dense_conditions)
        if self.debug_checks:
            assert not torch.isnan(shallow_cond).any(), f"浅层条件出现NaN: {torch.isnan(shallow_cond).sum()}"
        if self.debug_checks:
            assert not torch.isnan(deep_cond).any(), f"深层条件出现NaN: {torch.isnan(deep_cond).sum()}"

        # 3. 时间步和类别嵌入
        t_emb = self.jit_model.t_embedder(t)
        y_emb = self.jit_model.y_embedder(y)
        c = t_emb + y_emb
        if self.debug_checks:
            assert not torch.isnan(t_emb).any(), f"时间嵌入出现NaN: {torch.isnan(t_emb).sum()}"
        if self.debug_checks:
            assert not torch.isnan(y_emb).any(), f"类别嵌入出现NaN: {torch.isnan(y_emb).sum()}"
        if self.debug_checks:
            assert not torch.isnan(c).any(), f"条件嵌入出现NaN: {torch.isnan(c).sum()}"

        # 4. 提取patch嵌入
        x_patches = self.jit_model.x_embedder(x)  # [B, num_patches, hidden_size]
        if self.debug_checks:
            assert not torch.isnan(x_patches).any(), f"Patch嵌入出现NaN: {torch.isnan(x_patches).sum()}"
        if self.debug_checks:
            assert not torch.isinf(x_patches).any(), f"Patch嵌入出现inf: {torch.isinf(x_patches).sum()}"
        x_patches += self.jit_model.pos_embed
        if self.debug_checks:
            assert not torch.isnan(x_patches).any(), f"添加位置编码后出现NaN: {torch.isnan(x_patches).sum()}"
        if self.debug_checks:
            assert not torch.isinf(x_patches).any(), f"添加位置编码后出现inf: {torch.isinf(x_patches).sum()}"

        # 5. 准备in-context tokens（如果需要）
        if self.in_context_len > 0:
            in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
            in_context_tokens += self.jit_model.in_context_posemb
            if self.debug_checks:
                assert not torch.isnan(
                    in_context_tokens).any(), f"in-context tokens出现NaN: {torch.isnan(in_context_tokens).sum()}"

        if self.debug_checks:
            assert not torch.isnan(x).any(), "输入x包含NaN"
        if self.debug_checks:
            assert not torch.isinf(x).any(), "输入x包含Inf"
        # 6. 通过Transformer块，在选定层注入条件
        for i, block in enumerate(self.jit_model.blocks):
            # 在指定层添加in-context tokens
            if i == self.in_context_start and self.in_context_len > 0:
                if self.debug_checks:
                    assert not torch.isnan(x_patches).any(), f"拼接in-context前出现NaN: {torch.isnan(x_patches).any()}"
                if self.debug_checks:
                    assert not torch.isinf(x_patches).any(), f"拼接in-context前出现Inf: {torch.isinf(x_patches).any()}"
                x_patches = torch.cat([in_context_tokens, x_patches], dim=1)
                if self.debug_checks:
                    assert not torch.isnan(x_patches).any(), f"拼接in-context后出现NaN: {torch.isnan(x_patches).any()}"
                if self.debug_checks:
                    assert not torch.isinf(x_patches).any(), f"拼接in-context后出现Inf: {torch.isinf(x_patches).any()}"

            # 条件注入（如果需要）
            if str(i) in self.injection_layers:
                inject_type = self.injection_types[str(i)]

                # 选择条件token
                if inject_type == 'shallow':
                    cond_feat = shallow_cond
                elif inject_type == 'deep':
                    cond_feat = deep_cond
                elif inject_type == 'mixed':
                    # 混合浅层和深层条件
                    mix = torch.sigmoid(self.mix_weight)  # 限制在0-1之间
                    cond_feat = mix * shallow_cond + (1 - mix) * deep_cond
                else:
                    raise ValueError(f"未知的注入类型: {inject_type}")

                # 获取注入模块
                injection_module = self.injection_layers[str(i)]

                # 计算当前序列的patch tokens数量
                B, L, C = x_patches.shape
                if L > self.num_patches:
                    # 有in-context tokens
                    num_patch_tokens = self.num_patches
                else:
                    # 没有in-context tokens
                    num_patch_tokens = self.num_patches  # 注意：这里应该是num_patches，不是L

                if self.debug_checks:
                    assert not torch.isnan(x_patches).any(), f"Transformer块{i}条件注入前含NaN"
                if self.debug_checks:
                    assert not torch.isnan(cond_feat).any(), f"Transformer块{i}条件注入前含NaN"
                if self.debug_checks:
                    assert not torch.isinf(cond_feat).any(), f"Transformer块{i}条件注入前含Inf"
                if self.debug_checks:
                    assert not torch.isinf(x_patches).any(), f"Transformer块{i}条件注入前含Inf"
                # 自适应门控注入（传入num_patch_tokens参数）
                x_patches = injection_module(x_patches, cond_feat, num_patch_tokens=num_patch_tokens)
                if self.debug_checks:
                    assert not torch.isnan(x_patches).any(), f"Transformer块{i}条件注入后含NaN"
                if self.debug_checks:
                    assert not torch.isinf(x_patches).any(), f"Transformer块{i}条件注入后含Inf"

            # 选择正确的RoPE
            if i < self.in_context_start:
                rope = self.jit_model.feat_rope
            else:
                rope = self.jit_model.feat_rope_incontext

            # Transformer块前向传播
            if self.debug_checks:
                assert not torch.isnan(x_patches).any(), f"Transformer块{i}前出现NaN torch.isnan(x_patches).any()={torch.isnan(x_patches).any()}"
            if self.debug_checks:
                assert not torch.isinf(x_patches).any(), f"Transformer块{i}前出现inf"
            if self.debug_checks:
                assert not torch.isnan(c).any(), f"Transformer块{i}前出现NaN torch.isnan(c).any()={torch.isnan(c).any()}"
            x_patches = block(x_patches, c, feat_rope=rope)
            if self.debug_checks:
                assert not torch.isnan(x_patches).any(), f"Transformer块{i}后出现NaN"
            if self.debug_checks:
                assert not torch.isinf(x_patches).any(), f"Transformer块{i}后出现inf"

        # 7. 移除in-context tokens（如果有）
        if self.in_context_len > 0:
            x_patches = x_patches[:, self.in_context_len:]
            if self.debug_checks:
                assert not torch.isnan(x_patches).any(), f"移除in-context后出现NaN"

        # 8. 最终层
        x_out = self.jit_model.final_layer(x_patches, c)
        if self.debug_checks:
            assert not torch.isnan(x_out).any(), f"最终层输出出现NaN"
        output = self.jit_model.unpatchify(x_out, self.jit_model.patch_size)
        if self.debug_checks:
            assert not torch.isnan(output).any(), f"最终输出出现NaN"

        return output


def load_pretrained_weights(model, checkpoint_path):
    """
    加载预训练JiT权重

    Args:
        model: ConditionJiT模型
        checkpoint_path: 预训练权重路径

    Returns:
        加载权重后的模型
    """
    print(f"正在加载预训练权重: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 提取状态字典
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 过滤出JiT模型的权重
        jit_state_dict = {}
        for k, v in state_dict.items():
            # 跳过条件编码器和注入层的权重
            if any(keyword in k for keyword in [
                'cfa_reconstructor', 'cond_encoder', 'injection_layers',
                'injection_types', 'mix_weight'
            ]):
                continue

            # 处理不同的键名前缀
            if k.startswith('net.'):
                new_key = k.replace('net.', '')
                jit_state_dict[new_key] = v
            elif k.startswith('jit_model.'):
                new_key = k.replace('jit_model.', '')
                jit_state_dict[new_key] = v
            elif k.startswith('module.'):  # 分布式训练
                new_key = k.replace('module.', '')
                jit_state_dict[new_key] = v
            elif k in model.jit_model.state_dict():
                jit_state_dict[k] = v

        # 加载权重
        missing, unexpected = model.jit_model.load_state_dict(jit_state_dict, strict=False)

        if missing or unexpected:
            raise ValueError(f'Incompatible JiT weights: missing={missing}, unexpected={unexpected}')
        print(f"成功加载预训练权重")
        print(f"缺失的键: {len(missing)}")
        if missing:
            print(f"前10个缺失的键: {missing[:10]}")
        print(f"意外的键: {len(unexpected)}")
        if unexpected:
            print(f"前10个意外的键: {unexpected[:10]}")

        # 更新门控统计信息
        model.gate_stats = {'mean': 0.5, 'std': 0.1}  # 初始统计

    except Exception as e:
        print(f"加载预训练权重失败: {e}")
        raise RuntimeError("Pretrained loading failed; refusing random initialization") from e

    return model


def create_condition_model(model_name='JiT-B/16', img_size=256, num_classes=1000, pretrained_path=None, *, cfa_mode):
    """
    创建条件模型并加载预训练权重

    Args:
        model_name: JiT模型名称
        img_size: 图像大小
        num_classes: 类别数
        pretrained_path: JiT 主干预训练权重路径（不是完整训练断点）
        cfa_mode: 必须显式指定 'sparse' 或 'bayer'

    Returns:
        ConditionJiT模型
    """
    print(f"创建条件模型: {model_name}")
    print(f"图像大小: {img_size}")
    print(f"类别数: {num_classes}")

    model = ConditionJiT(
        pretrained_model_name=model_name,
        img_size=img_size,
        num_classes=num_classes,
        cfa_mode=cfa_mode
    )

    if pretrained_path:
        model = load_pretrained_weights(model, pretrained_path)

    return model
