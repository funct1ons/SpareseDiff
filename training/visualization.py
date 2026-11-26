"""Shared visualization; mode-specific layout and projection choices stay explicit."""
import os
import torch
from training.cfa import get_cfa_adapter


def generate_model_comparison(
        original_denoiser,
        demosaic_denoiser,
        output_dir,
        device,
        num_classes=1000,
        img_size=256,
        num_samples=4
):
    """
    生成对比图像：确保两个模型使用相同的初始化噪声
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置随机种子，确保可重复性
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # 生成相同的噪声，用于两个模型
    noise = torch.randn(
        num_samples, 3, img_size, img_size,
        device=device
    ) * demosaic_denoiser.noise_scale

    # 使用无类别标签
    labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)

    # 创建零条件输入（用于DemosaicDenoiser）
    channels = get_cfa_adapter(demosaic_denoiser.model.cfa_mode).condition_channels
    conditions = torch.zeros((num_samples, channels, img_size, img_size), device=device)

    with torch.no_grad():
        print("使用原始预训练模型生成图像...")

        # 由于原始denoiser的generate不接受外部噪声，我们需要临时修改它的生成方式
        # 或者创建一个辅助函数来确保噪声一致
        original_images = generate_with_specified_noise(
            original_denoiser,
            noise,
            labels,
            num_steps=30,
            method="euler"
        )
        original_images = (original_images + 1) / 2  # 反归一化

        print("使用DemosaicDenoiser生成图像...")
        # DemosaicDenoiser也需要使用相同的噪声
        # 我们可以创建一个临时函数或直接修改调用方式
        demosaic_images = generate_demosaic_with_specified_noise(
            demosaic_denoiser,
            noise,
            conditions,
            labels,
            num_steps=30,
            method="euler",
            cfg_scale=0.0
        )
        demosaic_images = (demosaic_images + 1) / 2  # 反归一化

    from torchvision.utils import make_grid, save_image

    # 创建对比网格
    all_images = []
    for i in range(num_samples):
        all_images.append(original_images[i])
    for i in range(num_samples):
        all_images.append(demosaic_images[i])

    # 创建网格：2行num_samples列
    grid = make_grid(
        torch.stack(all_images),
        nrow=num_samples,  # 每行num_samples列
        padding=10,
        pad_value=1.0
    )

    # 保存图像
    save_path = os.path.join(output_dir, "model_comparison_table.png")
    save_image(grid, save_path)

    print(f"对比图像已保存到: {save_path}")
    return save_path


def generate_with_specified_noise(denoiser, noise, labels, num_steps=25, method="euler"):
    """
    使用指定的噪声为原始Denoiser生成图像
    """
    device = labels.device
    bsz = labels.size(0)

    # 保存原始的步数和方法
    original_steps = denoiser.steps
    original_method = denoiser.method

    # 临时设置步数和方法（与DemosaicDenoiser一致）
    denoiser.steps = num_steps
    denoiser.method = method

    # 手动设置初始噪声
    z = noise.clone()

    # 时间步调度
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    # 复制原始denoiser的生成逻辑
    for i in range(num_steps):
        t = timesteps[i].view(1, *([1] * (z.ndim - 1)))
        t_next = timesteps[i + 1].view(1, *([1] * (z.ndim - 1)))

        if method == "euler":
            z = denoiser._euler_step(z, t, t_next, labels)
        elif method == "heun":
            z = denoiser._heun_step(z, t, t_next, labels)

    # 恢复原始设置
    denoiser.steps = original_steps
    denoiser.method = original_method

    return z


def generate_demosaic_with_specified_noise(
        demosaic_denoiser,
        noise,
        conditions,
        labels,
        num_steps=25,
        method="euler",
        cfg_scale=1.0
):
    """
    使用指定的噪声为DemosaicDenoiser生成图像（无约束投影）
    """
    device = conditions.device
    bsz = conditions.shape[0]

    # 手动设置初始噪声
    z = noise.clone()

    # 时间步调度
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    # 采样循环 - 不进行约束投影
    for i in range(num_steps):
        t = timesteps[i].expand(bsz, 1, 1, 1)
        t_next = timesteps[i + 1].expand(bsz, 1, 1, 1)

        if method == "euler":
            z = demosaic_denoiser._euler_step(z, conditions, t, t_next, labels, cfg_scale)
        elif method == "heun":
            z = demosaic_denoiser._heun_step(z, conditions, t, t_next, labels, cfg_scale)

    return z


def generate_visualization(denoiser, batch, epoch, output_dir, device, adapter):
    """
    生成可视化结果：原始图像、马赛克图像、解马赛克结果对比
    根据光照水平使用不同的约束强度
    """
    denoiser.eval()

    # 获取数据
    clean_images = batch["image"].to(device)
    conditions, cfa_images, cfa_w_images = adapter.visualization_inputs(batch, device)

    # 获取光照水平标签
    if "light_levels" in batch:
        light_levels = batch["light_levels"].cpu().numpy()
    else:
        light_levels = [1.0] * len(clean_images)

    # 判断是否为正常光照（light_level=1.0）
    # 注意：由于浮点数精度，使用容差判断
    normal_light_threshold = 0.01
    is_normal_light = [abs(level - 1.0) < normal_light_threshold for level in light_levels]

    # 使用中性标签
    labels = torch.full((clean_images.shape[0],), fill_value=denoiser.num_classes,
                        dtype=torch.long, device=device)

    # 初始化结果张量
    generated = torch.zeros_like(clean_images)

    # 找出正常光照和低光照的索引
    normal_indices = [i for i, is_normal in enumerate(is_normal_light) if is_normal]
    lowlight_indices = [i for i, is_normal in enumerate(is_normal_light) if not is_normal]

    print(f"可视化样本: {len(clean_images)}个 (正常光照: {len(normal_indices)}个, 低光照: {len(lowlight_indices)}个)")

    # 分别处理正常光照和低光照样本
    with torch.no_grad():
        # 1. 正常光照样本：使用强约束 (constraint_strength=1.0)
        if normal_indices:
            print(f"  正常光照样本索引: {normal_indices}")
            normal_conditions = conditions[normal_indices]
            normal_labels = labels[normal_indices]

            normal_generated = denoiser.generate(
                conditions=normal_conditions,
                labels=normal_labels,
                num_steps=25,
                method="euler",
                cfg_scale=0.0,
                apply_constraint=True,
                constraint_strength=1.0  # 正常光照使用强约束
            )

            generated[normal_indices] = normal_generated

        # 2. 低光照样本：使用弱约束或无约束 (constraint_strength=0.0)
        if lowlight_indices:
            print(f"  低光照样本索引: {lowlight_indices}")
            lowlight_conditions = conditions[lowlight_indices]
            lowlight_labels = labels[lowlight_indices]

            lowlight_generated = denoiser.generate(
                conditions=lowlight_conditions,
                labels=lowlight_labels,
                num_steps=25,
                method="euler",
                cfg_scale=0.0,
                apply_constraint=adapter.lowlight_apply_constraint,
                constraint_strength=0.0  # 低光照使用弱约束或无约束
            )

            generated[lowlight_indices] = lowlight_generated

    # 反归一化：[-1, 1] -> [0, 1]
    clean_images = (clean_images + 1) / 2
    generated = (generated + 1) / 2
    cfa_display = (cfa_images + 1) / 2
    if cfa_w_images is not None:
        cfa_w_display = (cfa_w_images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) + 1) / 2

    # 创建可视化网格（使用torchvision，无需matplotlib）
    from torchvision.utils import make_grid, save_image

    # 收集所有要显示的图像
    all_images = []

    for i in range(len(clean_images)):
        # 获取光照信息
        light_level = light_levels[i]
        if abs(light_level - 1.0) < normal_light_threshold:
            light_label = "Normal Light"
            constraint_label = "Constraint=1.0"
        else:
            light_label = f"Low Light (1/{int(1 / light_level)})"
            constraint_label = "Constraint=0.0"

        # 收集图像：原始RGB、马赛克RGB、W通道、生成RGB
        all_images.append(clean_images[i])
        all_images.append(cfa_display[i])
        if cfa_w_images is not None:
            all_images.append(cfa_w_display[i])
        all_images.append(generated[i])

    # 创建网格
    # 每行显示一个样本的4个视图
    num_samples = len(clean_images)
    grid = make_grid(
        torch.stack(all_images),
        nrow=4 if cfa_w_images is not None else 3,
        padding=10,
        pad_value=1.0
    )

    # 保存图像
    save_path = os.path.join(output_dir, f'visualization_epoch_{epoch:03d}.png')
    save_image(grid, save_path)

    # print(f"可视化结果已保存到: {save_path}")


    return grid


def log_visualization_to_tensorboard(tracker, grid, epoch):
    """
    将可视化结果记录到TensorBoard
    """
    # 确保网格在[0, 1]范围
    grid = torch.clamp(grid, 0, 1)

    # 记录图像
    tracker.add_images(
        "validation/demosaic_results",
        grid.unsqueeze(0),  # 添加批次维度
        global_step=epoch
    )
