"""
Sparse/Bayer 共享解马赛克去噪器
包装条件扩散模型，实现v-prediction训练和采样
集成门控正则化和约束投影机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemosaicDenoiser(nn.Module):
    """
    解马赛克去噪器
    使用v-prediction目标训练条件扩散模型
    集成门控正则化损失和约束投影机制

    主要修改点：
    1. CFA 条件由模型适配器选择，损失和采样算法共享
    2. 添加门控正则化损失
    3. 增强约束投影机制
    """

    def __init__(self, model, img_size=256, num_classes=1000,
                 noise_scale=0.5, t_eps=1e-3, P_mean=-1.2, P_std=1.2,
                 gate_reg_weight=0.01, constraint_weight=0.1, pattern_mask = None):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.num_classes = num_classes
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.P_mean = P_mean
        self.P_std = P_std

        # 损失权重参数
        self.gate_reg_weight = gate_reg_weight  # 门控正则化权重
        self.constraint_weight = constraint_weight  # 约束损失权重

        # 标签dropout概率（用于分类器自由引导）
        self.label_drop_prob = 0.1

        if pattern_mask is None:
            from training.cfa import get_cfa_adapter
            self.pattern_mask = get_cfa_adapter(model.cfa_mode).pattern_mask()
        else:
            self.pattern_mask = pattern_mask
        # 保持普通 tensor 属性，不增加持久化 state_dict 键
        self._init_rgb_mask(img_size)

    def _init_rgb_mask(self, img_size):
        """初始化时生成完整的RGB mask"""
        # 提取RGB掩码（前3个通道）
        rgb_pattern = self.pattern_mask[:3]  # [3, tile_h, tile_w]

        # 计算需要在H和W方向上重复的次数
        H, W = img_size, img_size
        tile_h, tile_w = rgb_pattern.shape[-2:]
        H_repeat = (H + tile_h - 1) // tile_h
        W_repeat = (W + tile_w - 1) // tile_w

        # 重复模式到整个图像
        rgb_mask = rgb_pattern.unsqueeze(0)  # [1, 3, tile_h, tile_w]
        rgb_mask = rgb_mask.repeat(1, 1, H_repeat, W_repeat)
        rgb_mask = rgb_mask[:, :, :H, :W]  # 裁剪到精确尺寸 [1, 3, H, W]

        # 不在这里注册为buffer，而是在forward时根据设备创建
        self.rgb_mask_template = rgb_mask

        # 计算并存储已知像素总数（用于归一化）
        self.mask_sum_value = rgb_mask.sum().item()

        print(f"RGB mask模板已生成: {self.rgb_mask_template.shape}")
        print(f"已知像素总数: {self.mask_sum_value}")

    def drop_labels(self, labels):
        """
        随机drop标签，用于分类器自由引导

        Args:
            labels: 类别标签 [B,]

        Returns:
            随机drop后的标签，dropped的标签被替换为num_classes
        """
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n, device=None):
        """
        从logit-Normal分布采样时间步t

        Args:
            n: 采样数量
            device: 设备

        Returns:
            时间步t [n,]，值在(0,1)范围内
        """
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, conditions, labels, return_components=False):
        """
        前向传播计算v-prediction损失和门控正则化损失

        Args:
            x: 干净图像 [B, 3, H, W]
            conditions: 条件输入 [B, 4, H, W]（稀疏RGGBW马赛克，4通道）
            labels: 类别标签 [B,]
            return_components: 是否返回损失分量（用于调试）

        Returns:
            总损失，或（总损失，损失分量字典）
        """
        # === 输入验证 ===
        assert not torch.isnan(x).any(), f"输入x包含NaN: {torch.isnan(x).sum()}"
        assert not torch.isinf(x).any(), f"输入x包含Inf: {torch.isinf(x).sum()}"
        assert x.isfinite().all(), f"输入x包含非有限值"
        assert not torch.isnan(conditions).any(), f"输入conditions包含NaN: {torch.isnan(x).sum()}"
        assert not torch.isinf(conditions).any(), f"输入conditions包含Inf: {torch.isinf(x).sum()}"
        assert conditions.isfinite().all(), f"输入conditions包含非有限值"
        # === 检查模型参数 ===
        model_params_nan = any(torch.isnan(p).any() for p in self.model.parameters() if p.requires_grad)
        assert not model_params_nan, "模型参数包含NaN"

        # 随机drop标签（用于分类器自由引导）
        labels_dropped = self.drop_labels(labels) if self.training else labels

        # 采样时间步
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))

        # 采样噪声
        e = torch.randn_like(x) * self.noise_scale

        # 构造噪声输入（与预训练模型保持一致）
        z = t * x + (1 - t) * e
        # === 噪声输入验证 ===
        assert not torch.isnan(z).any(), f"噪声输入z包含NaN"
        assert not torch.isinf(z).any(), f"噪声输入z包含Inf"

        # 计算真实速度v（使用预训练模型公式）
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        # === 真实速度验证 ===
        assert not torch.isnan(v).any(), f"真实速度v包含NaN"
        assert not torch.isinf(v).any(), f"真实速度v包含Inf"

        # 模型预测
        # 注意：conditions现在是4通道稀疏RGGBW马赛克
        # 在model_condition.py中会先重构为5通道稠密表示
        x_pred = self.model(z, conditions, t.flatten(), labels_dropped)
        # === 模型预测验证 ===
        assert not torch.isnan(x_pred).any(), (f"模型预测x_pred包含NaN,torch.isnan(x_pred).any()={torch.isnan(x_pred).any()},"
                                               f"\ntorch.isnan(z).any()={torch.isnan(z).any()},"
                                               f"\ntorch.isnan(conditions).any()={torch.isnan(conditions).any()},"
                                               f"\ntorch.isnan(t).any()={torch.isnan(t).any()},"
                                               f"\nlabels_dropped = {labels_dropped}"
                                               f"\nDEBUG: x_pred NaN统计 - 总数: {torch.isnan(x_pred).sum()}"
                                               f"\nDEBUG: x_pred inf统计 - 总数: {torch.isinf(x_pred).sum()}")
        assert not torch.isinf(x_pred).any(), f"模型预测x_pred包含Inf"

        # 计算预测速度v_pred（使用预训练模型公式）
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        # === 预测速度验证 ===
        assert not torch.isnan(v_pred).any(), f"预测速度v_pred包含NaN"
        assert not torch.isinf(v_pred).any(), f"预测速度v_pred包含Inf"

        # v-prediction损失（主要训练目标）
        vpred_loss = F.mse_loss(v_pred, v, reduction='mean')
        # === 损失验证 ===
        assert not torch.isnan(vpred_loss), f"v-prediction损失为NaN: {vpred_loss}"
        assert not torch.isinf(vpred_loss), f"v-prediction损失为Inf: {vpred_loss}"
        assert vpred_loss >= 0, f"v-prediction损失为负: {vpred_loss}"

        # 1. 约束损失（已知像素一致性）
        # 注意：这里需要从稀疏conditions中提取已知像素掩码
        # 假设conditions的通道顺序为：[R, G, B, W]
        # RGB已知像素在conditions的前3个通道中，值不为0的位置就是已知像素
        constraint_loss = self._calculate_constraint_loss(x, x_pred, conditions)

        # 2. 门控正则化损失
        gate_reg_loss = self.model.get_gate_regularization_loss()

        # 总损失
        total_loss = vpred_loss + self.constraint_weight * constraint_loss + self.gate_reg_weight * gate_reg_loss

        if return_components:
            # 返回损失分量，便于调试和分析
            loss_components = {
                'total': total_loss.item(),
                'vpred': vpred_loss.item(),
                'constraint': constraint_loss.item(),
                'gate_reg': gate_reg_loss.item(),
                'constraint_weighted': (self.constraint_weight * constraint_loss).item(),
                'gate_reg_weighted': (self.gate_reg_weight * gate_reg_loss).item()
            }
            return total_loss, loss_components
        else:
            return total_loss

    def _calculate_constraint_loss(self, x_clean, x_pred, conditions):
        """
        计算已知像素约束损失（使用预先生成的RGB mask）

        Args:
            x_clean: 干净图像 [B, 3, H, W]
            x_pred: 模型预测图像 [B, 3, H, W]
            conditions: CFA 条件输入（该损失沿用原实现，以 clean/pred 和 RGB mask 计算）

        Returns:
            约束损失
        """
        B = x_clean.shape[0]
        device = x_clean.device

        # 动态生成掩码并移动到正确设备
        rgb_mask = self.rgb_mask_template.to(device).repeat(B, 1, 1, 1)  # [B, 3, H, W]
        mask_sum = torch.tensor(self.mask_sum_value, device=device)

        if mask_sum > 0:
            # 计算已知像素的MSE损失
            diff = (x_pred - x_clean) * rgb_mask
            constraint_loss = (diff ** 2).sum() / mask_sum.clamp_min(1e-5)
        else:
            # 没有已知像素，约束损失为0
            constraint_loss = torch.tensor(0.0, device=x_clean.device)

        return constraint_loss

    @torch.no_grad()
    def generate(self, conditions, labels, num_steps=25, method="euler",
                 cfg_scale=3.0, apply_constraint=True, constraint_strength=1.0):
        """
        生成解马赛克图像

        Args:
            conditions: 条件输入 [B, 4, H, W]（稀疏RGGBW马赛克）
            labels: 类别标签 [B,]
            num_steps: 采样步数
            method: 采样方法 ("euler" 或 "heun")
            cfg_scale: 分类器自由引导尺度
            apply_constraint: 是否应用约束投影
            constraint_strength: 约束强度（0.0-1.0），1.0表示完全约束

        Returns:
            生成的RGB图像 [B, 3, H, W]
        """
        device = conditions.device
        bsz = conditions.shape[0]

        # 初始化噪声
        z = self.noise_scale * torch.randn(
            bsz, 3, self.img_size, self.img_size, device=device
        )

        # 时间步调度（从0到1）
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        # 采样循环
        for i in range(num_steps):
            t = timesteps[i].expand(bsz, 1, 1, 1)
            t_next = timesteps[i + 1].expand(bsz, 1, 1, 1)

            if method == "euler":
                z = self._euler_step(z, conditions, t, t_next, labels, cfg_scale)
            elif method == "heun":
                z = self._heun_step(z, conditions, t, t_next, labels, cfg_scale)
            else:
                raise ValueError(f"未知的采样方法: {method}")

            # 约束投影：将已知像素投影回条件值
            if apply_constraint:
                z = self._constraint_projection(z, conditions, strength=constraint_strength)

        return z

    @torch.no_grad()
    def generate_unconstrained(self, conditions, labels, num_steps=25,
                               method="euler", cfg_scale=3.0):
        """
        生成解马赛克图像（不进行约束投影）
        用于比较约束投影的效果

        Args:
            conditions: 条件输入 [B, 4, H, W]
            labels: 类别标签 [B,]
            num_steps: 采样步数
            method: 采样方法 ("euler" 或 "heun")
            cfg_scale: 分类器自由引导尺度

        Returns:
            生成的RGB图像 [B, 3, H, W]
        """
        # 调用generate方法，但关闭约束投影
        return self.generate(
            conditions=conditions,
            labels=labels,
            num_steps=num_steps,
            method=method,
            cfg_scale=cfg_scale,
            apply_constraint=False,
            constraint_strength=0.0
        )

    @torch.no_grad()
    def _constraint_projection(self, z, conditions, strength=1.0):
        """
        约束投影（使用预先生成的RGB mask）
        """
        if strength <= 0:
            return z

        B = z.shape[0]
        device = z.device

        # 扩展mask到批次维度
        rgb_mask = self.rgb_mask_template.to(device).repeat(B, 1, 1, 1)  # [B, 3, H, W]
        mask_sum = torch.tensor(self.mask_sum_value, device=device)

        if mask_sum == 0:
            return z  # 没有已知像素，无需投影

        # conditions中的RGB已知值
        rgb_known = conditions[:, :3, :, :]  # [B, 3, H, W]

        # 投影（只在mask位置替换）
        z_projected = z * (1 - strength * rgb_mask) + rgb_known * (strength * rgb_mask)

        return z_projected

    @torch.no_grad()
    def _forward_sample(self, z, conditions, t, labels, cfg_scale):
        """
        单步前向采样，支持分类器自由引导

        Args:
            z: 当前噪声图像 [B, 3, H, W]
            conditions: 条件输入 [B, 4, H, W]
            t: 当前时间步 [B, 1, 1, 1]
            labels: 类别标签 [B,]
            cfg_scale: 分类器自由引导尺度

        Returns:
            预测的速度场 [B, 3, H, W]
        """
        # 条件预测
        x_cond = self.model(z, conditions, t.flatten(), labels)
        v_cond = (x_cond - z) / (1 - t).clamp_min(self.t_eps)

        if cfg_scale <= 0:
            # 不使用CFG
            return v_cond

        # 无条件预测（使用dropped标签）
        labels_uncond = torch.full_like(labels, self.num_classes)
        x_uncond = self.model(z, conditions, t.flatten(), labels_uncond)
        v_uncond = (x_uncond - z) / (1 - t).clamp_min(self.t_eps)

        # 分类器自由引导
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, conditions, t, t_next, labels, cfg_scale):
        """
        Euler采样步（一阶方法）

        Args:
            z: 当前噪声图像 [B, 3, H, W]
            conditions: 条件输入 [B, 4, H, W]
            t: 当前时间步 [B, 1, 1, 1]
            t_next: 下一个时间步 [B, 1, 1, 1]
            labels: 类别标签 [B,]
            cfg_scale: 分类器自由引导尺度

        Returns:
            下一个时间步的噪声图像
        """
        v_pred = self._forward_sample(z, conditions, t, labels, cfg_scale)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, conditions, t, t_next, labels, cfg_scale):
        """
        Heun采样步（二阶方法，更精确但更慢）

        Args:
            z: 当前噪声图像 [B, 3, H, W]
            conditions: 条件输入 [B, 4, H, W]
            t: 当前时间步 [B, 1, 1, 1]
            t_next: 下一个时间步 [B, 1, 1, 1]
            labels: 类别标签 [B,]
            cfg_scale: 分类器自由引导尺度

        Returns:
            下一个时间步的噪声图像
        """
        # 第一步：Euler预测
        v_pred_t = self._forward_sample(z, conditions, t, labels, cfg_scale)
        z_next_euler = z + (t_next - t) * v_pred_t

        # 第二步：在预测点上计算速度
        v_pred_t_next = self._forward_sample(z_next_euler, conditions, t_next, labels, cfg_scale)

        # 第三步：平均速度
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)

        # 第四步：使用平均速度更新
        z_next = z + (t_next - t) * v_pred

        return z_next

    def get_gate_statistics(self):
        """
        获取门控机制统计信息

        Returns:
            包含门控统计信息的字典
        """
        if hasattr(self.model, 'gate_stats'):
            return self.model.gate_stats
        else:
            return {'mean': 0.0, 'std': 1.0}

    def set_gate_regularization_weight(self, weight):
        """
        设置门控正则化权重

        Args:
            weight: 新的门控正则化权重
        """
        self.gate_reg_weight = weight
        print(f"门控正则化权重设置为: {weight}")

    def set_constraint_weight(self, weight):
        """
        设置约束损失权重

        Args:
            weight: 新的约束损失权重
        """
        self.constraint_weight = weight
        print(f"约束损失权重设置为: {weight}")

    def analyze_gate_values(self):
        """
        分析门控值分布（用于调试和监控）

        Returns:
            包含门控值统计信息的字典
        """
        gate_stats = {}

        # 收集所有注入层的门控信息
        for name, module in self.model.injection_layers.items():
            layer_stats = {
                'gate_scale': module.gate_scale.item(),
                'fusion_weights': module.fusion_weights.detach().cpu().numpy().tolist()
            }

            # 收集门控网络的统计信息
            gate_means = []
            for i, gate_net in enumerate(module.gate_nets):
                # 获取sigmoid层的输出统计（如果有的话）
                if hasattr(gate_net, 'forward'):
                    # 这里可以添加更详细的门控值统计
                    gate_means.append(0.5)  # 默认值，实际需要运行时数据

            layer_stats['gate_means'] = gate_means
            gate_stats[f'layer_{name}'] = layer_stats

        return gate_stats
