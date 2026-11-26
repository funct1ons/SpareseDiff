"""Single shared training driver; CFA data construction stays in adapters."""
import os

# Keep the archived-code-derived startup behavior before importing torch/models.
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'

import json
import math
from pathlib import Path
from functools import partial

import datasets
import diffusers
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils import is_accelerate_version

from training.distributed import (
    execution_metadata, main_action, process_group_kwargs, require_finite,
    resolve_checkpoint, restore_rng_strict, save_checkpoint, seed_worker, validate_execution,
    training_configuration, validate_training_configuration,
)
from training.cfa import get_cfa_adapter
from training.data import SimulateLowLight, make_dataset
from training.models.model_condition import create_condition_model
from training.models.denoiser_demosaic import DemosaicDenoiser
from training.visualization import (
    generate_model_comparison, generate_visualization, log_visualization_to_tensorboard,
)

logger = get_logger(__name__, log_level='INFO')
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(False)


def count_trainable_parameters(model):
    """统计可训练参数数量和总参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_trainable_layers(model, prefix=""):
    """打印每个层的可训练状态"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{prefix}✓ TRAINABLE: {name} ({param.numel()} parameters)")
        else:
            print(f"{prefix}✗ FROZEN: {name} ({param.numel()} parameters)")


def verify_model_parameters(model):
    """验证训练阶段参数设置是否正确"""
    # 统计参数
    total_params, trainable_params = count_trainable_parameters(model)
    trainable_percentage = 100 * trainable_params / total_params

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"可训练参数比例: {trainable_percentage:.2f}%")

    # 按模块统计
    print("\n按模块统计:")
    for module_name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if module_total > 0:
            module_percentage = 100 * module_trainable / module_total
            print(f"  {module_name}: {module_trainable:,}/{module_total:,} ({module_percentage:.1f}%)")

        print(f"警告：可训练参数比例({trainable_percentage:.2f}%)可能过低")

    return total_params, trainable_params


def run_training(args):
    set_seed(args.seed)
    adapter = get_cfa_adapter(args.cfa_mode)

    # 初始化Accelerator
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[process_group_kwargs()],
    )

    validate_execution(accelerator, args)
    runtime = execution_metadata(accelerator)
    training_config = training_configuration(vars(args))
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Read-only preflight: reject both source and destination mismatches before
    # creating a new output/config/log file, initializing models, or loading state.
    def prepare_output():
        config_path = Path(args.output_dir) / 'training_config.json'
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding='utf-8'))
            validate_training_configuration(previous, training_config)
            if 'checkpoint_schema' not in previous:
                if accelerator.num_processes != 1 or previous['mixed_precision'] != args.mixed_precision:
                    raise ValueError('Legacy resume requires same precision and world_size=1')
            elif any(previous.get(k) != v for k, v in runtime.items() if k != 'checkpoint_schema') or previous['checkpoint_schema'] not in (1, 2):
                raise ValueError('Output execution world/precision/scheduler compatibility mismatch')
            if not args.resume_from_checkpoint:
                raise ValueError('Existing run: use a new output directory or explicitly resume')
        if args.resume_from_checkpoint:
            resolve_checkpoint(args.output_dir, args.resume_from_checkpoint,
                               dict(runtime, training_config=training_config))

    main_action(accelerator, prepare_output)

    weight_dtype = torch.float32

    # 加载数据集
    logger.info(f"Loading dataset from {args.cache_dir}")
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.smoke_test:
        dataset = make_dataset(args.resolution, args.cfa_mode)
    elif args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            None,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    if args.max_train_samples:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    # 数据预处理
    # 初始化低光照模拟器
    lowlight_simulator = SimulateLowLight(
        lowlight_levels=[1 / 30, 1 / 40, 1 / 50, 1 / 80, 1 / 100, 1 / 150],
        noise_std=args.lowlight_noise_std,
        apply_prob=args.lowlight_prob if args.light_mode == "mixed" else 0.0
    )

    # 定义基础变换
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def preprocess_fn(examples):
        return adapter.preprocess(
            examples, resolution=args.resolution,
            lowlight_simulator=lowlight_simulator, base_transform=base_transform
        )

    required_columns = list(adapter.required_columns)
    if any(k not in dataset.column_names for k in required_columns):
        raise ValueError(f'Dataset must contain {required_columns}; got {dataset.column_names}')
    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(preprocess_fn)

    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.dataloader_num_workers,
        worker_init_fn=partial(seed_worker, rank=accelerator.process_index),
        # pin_memory=True
    )

    # 初始化模型
    logger.info(f"Initializing model: {args.model_name}")

    # 正常初始化
    model = create_condition_model(
        model_name=args.model_name,
        img_size=args.resolution,
        num_classes=args.num_classes,
        pretrained_path=args.pretrained_checkpoint,
        cfa_mode=args.cfa_mode
    )

    # 确保所有参数都冻结
    for param in model.parameters():
        param.requires_grad = False

    # 确保条件编码器和注入层保持可训练
    condition_trained = False
    for name, param in model.named_parameters():
        if 'cond_encoder' in name or 'injection_layers' in name:
            param.requires_grad = True
            condition_trained = True

    if not condition_trained:
        logger.warning("未找到条件编码器或注入层参数，请检查模型结构")

    # 解冻预训练模型的最后几层
    model.unfreeze_pretrained_layers(num_layers=args.num_layers)

    logger.info(f"解冻最后 {args.num_layers} 层进行微调")

    adapter.after_model_init(model)

    # ========== 验证1：设置后的参数状态 ==========
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("初始参数设置验证")
        print("=" * 60)
        total_params, trainable_params = count_trainable_parameters(model)
        print(f"设置后 - 总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

        # 详细打印
        print_trainable_layers(model, prefix="[设置后] ")

    # 创建去噪器
    denoiser = DemosaicDenoiser(
        model=model,
        img_size=args.resolution,
        num_classes=args.num_classes,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps
    )

    smoke_before = {}
    if args.smoke_test:
        for prefix in ['cond_encoder.shallow_patch_embed.0.conv.0.weight', 'injection_layers.5.gate_scale', 'jit_model.blocks.11.attn.qkv.weight', 'jit_model.x_embedder.proj1.weight']:
            smoke_before[prefix] = dict(model.named_parameters())[prefix].detach().cpu().clone()
    observed_levels = set()
    level_values = [1.0, 1/30, 1/40, 1/50, 1/80, 1/100, 1/150]
    level_counts = torch.zeros(len(level_values), dtype=torch.long, device=accelerator.device)

    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, denoiser.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # Sharding changes the real epoch length. Build the scheduler only AFTER prepare.
    denoiser, optimizer, train_dataloader = accelerator.prepare(
        denoiser, optimizer, train_dataloader
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    if max_train_steps == 0:
        raise ValueError('Empty prepared training loader')
    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=max_train_steps,
    )
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)
    runtime.update(scheduler_total_steps=max_train_steps,
                   updates_per_epoch=num_update_steps_per_epoch)
    # Model initialization was identical; stochastic augmentation/diffusion is rank-specific.
    set_seed(args.seed, device_specific=True)

    # ========== 验证2：Accelerator准备后的参数状态 ==========
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Accelerator准备后参数验证")
        print("=" * 60)
        # 获取原始模型（unwrap_model确保我们检查的是实际模型）
        unwrapped_denoiser = accelerator.unwrap_model(denoiser)
        verify_model_parameters(unwrapped_denoiser)
        print_trainable_layers(unwrapped_denoiser, prefix="[优化器内] ")
        # 验证优化器参数
        optimizer_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        print(f"优化器管理的参数数量: {optimizer_params:,}")

        # 检查是否有不应该训练的参数被优化
        for group in optimizer.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    print("警告：优化器包含了不需要梯度的参数！")

    # 在训练循环开始前，准备一个固定的batch用于可视化
    if accelerator.is_main_process and args.enable_visualization:
        # 定义我们想要可视化的光照类型
        target_light_types = [1.0] if args.light_mode == "normal" else [1.0, 1/30, 1/50, 1/80, 1/150]

        # 保留原数量：Sparse 每档2个，Bayer 每档6个
        samples_per_type = adapter.samples_per_light
        selected_samples = {lt: [] for lt in target_light_types}

        # 遍历数据集，按光照类型收集样本
        for idx, sample in enumerate(dataset):
            light_level = sample.get('light_level', 1.0)

            # 四舍五入到最接近的目标光照类型（因为浮点数可能有精度问题）
            closest_type = min(target_light_types, key=lambda x: abs(x - light_level))

            # 如果这个类型还没有收集足够的样本，并且差值很小（确保匹配）
            if (len(selected_samples[closest_type]) < samples_per_type and
                    abs(closest_type - light_level) < 0.01):
                selected_samples[closest_type].append(sample)

            # 检查是否所有类型都收集够了
            if all(len(samples) >= samples_per_type for samples in selected_samples.values()):
                break

        # 将所有选中的样本合并
        all_selected = []
        for light_type, samples in selected_samples.items():
            print(f"光照类型 {light_type}: 收集到 {len(samples)} 个样本")
            all_selected.extend(samples)

        if not all_selected:
            # 如果没收集到足够的样本，回退到原来的方法
            print("警告：未能收集到足够的样本，使用备用方法")
            vis_batch = None
            for batch in train_dataloader:
                vis_batch = batch
                break

            if vis_batch is not None:
                # Preserve the old fallback: light_levels is not added here.
                vis_batch = {key: vis_batch[key].to(accelerator.device) for key in adapter.batch_columns}
        else:
            # 将选中的样本转换为批次格式
            vis_batch = {key: torch.stack([sample[key] for sample in all_selected])
                         for key in adapter.batch_columns}
            vis_batch['light_levels'] = torch.tensor([sample.get('light_level', 1.0) for sample in all_selected])

            # 移动到指定设备
            vis_batch = {key: value.to(accelerator.device) for key, value in vis_batch.items()}

            print(f"已创建包含 {len(all_selected)} 个样本的可视化批次")
            print(f"光照类型分布: {vis_batch['light_levels'].cpu().tolist()}")


    # 训练统计
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Training Configuration *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # 训练循环
    global_step = 0
    first_epoch = 0
    last_saved_step = None
    checkpoint_cursor = {}

    # 恢复检查点
    if args.resume_from_checkpoint:
        def check_resume():
            resolve_checkpoint(args.output_dir, args.resume_from_checkpoint, dict(runtime, training_config=training_config))
        main_action(accelerator, check_resume)
        path = resolve_checkpoint(args.output_dir, args.resume_from_checkpoint, dict(runtime, training_config=training_config))
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(str(path))
        restore_rng_strict(accelerator, path)
        global_step = int(path.name.split('-')[1])
        if args.max_train_steps and global_step >= args.max_train_steps:
            raise ValueError('Checkpoint has already reached max_train_steps')
        if global_step >= max_train_steps:
            raise ValueError('Checkpoint has already reached configured num_epochs')
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps
        marker = path / 'complete.json'
        if marker.is_file():
            cursor = json.loads(marker.read_text(encoding='utf-8'))
            first_epoch = cursor.get('next_epoch', first_epoch)
            resume_step = cursor.get('next_microstep', resume_step)

    def record_execution():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        config_path = Path(args.output_dir) / 'training_config.json'
        config = dict(vars(args), **runtime)
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    # Write/upgrade output metadata only after all resume checks and state load.
    main_action(accelerator, record_execution)
    checkpoint_metadata = dict(runtime, training_config=training_config)
    if accelerator.is_main_process:
        accelerator.init_trackers('train_demosaic')

    # ========== 添加模型对比 ==========
    if accelerator.is_main_process and args.initial_comparison and not args.resume_from_checkpoint:
        logger.info("首次启动，进行模型生成对比...")

        # 1. 创建原始预训练Denoiser
        from training.models.denoiser import Denoiser

        # 创建原始Denoiser的参数
        class OriginalArgs:
            model = args.model_name
            img_size = args.resolution
            class_num = args.num_classes
            attn_dropout = 0.0
            proj_dropout = 0.0
            label_drop_prob = 0.1
            P_mean = -1.2
            P_std = 1.2
            t_eps = args.t_eps
            noise_scale = args.noise_scale
            ema_decay1 = 0.9999
            ema_decay2 = 0.9999
            sampling_method = 'euler'
            num_sampling_steps = 30
            cfg = 0.0
            interval_min = 0.0
            interval_max = 1.0

        original_args = OriginalArgs()
        original_denoiser = Denoiser(original_args)

        # 加载预训练权重
        pretrained_dict = torch.load(args.pretrained_checkpoint, map_location='cpu')
        if 'model' in pretrained_dict:
            original_denoiser.load_state_dict(pretrained_dict['model'])
        else:
            original_denoiser.load_state_dict(pretrained_dict)

        original_denoiser.to(accelerator.device)
        original_denoiser.eval()

        # 2. 获取DemosaicDenoiser（去掉accelerate包装）
        unwrapped_demosaic = accelerator.unwrap_model(denoiser)

        # 3. 生成对比图像
        comparison_dir = os.path.join(args.output_dir, "initial_comparison")
        generate_model_comparison(
            original_denoiser=original_denoiser,
            demosaic_denoiser=unwrapped_demosaic,
            output_dir=comparison_dir,
            device=accelerator.device,
            num_classes=args.num_classes,
            img_size=args.resolution,
            num_samples=10
        )

        logger.info(f"模型对比完成，图像保存在: {comparison_dir}")
    # ========== 模型对比结束 ==========


    # 训练
    for epoch in range(first_epoch, args.num_epochs):
        denoiser.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            for value in batch['light_level']:
                value = float(value)
                observed_levels.add(value)
                level_counts[min(range(len(level_values)), key=lambda i: abs(level_values[i] - value))] += 1
            # 获取数据
            clean_images = batch["image"].to(weight_dtype)  # 原始RGB图像
            conditions = adapter.make_conditions(batch, dtype=weight_dtype)

            # 无类别标签
            labels = torch.full((clean_images.shape[0],), fill_value=args.num_classes, dtype=torch.long)
            labels = labels.to(clean_images.device)


            with accelerator.accumulate(denoiser):
                # 前向传播计算损失
                loss = denoiser(clean_images, conditions, labels)

                require_finite(accelerator, loss, 'training loss')

                # 反向传播
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(denoiser.parameters(), 1.0)
                    require_finite(accelerator, grad_norm, 'gradient norm')

                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad()

            # 更新进度
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                progress_bar.update(1)
                global_step += 1
                checkpoint_cursor = {'next_epoch': epoch, 'next_microstep': step + 1}
                if step + 1 == len(train_dataloader):
                    checkpoint_cursor = {'next_epoch': epoch + 1, 'next_microstep': 0}

                if not args.smoke_test and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(accelerator, args.output_dir, f'checkpoint-{global_step}',
                                    dict(checkpoint_metadata, **checkpoint_cursor), args.checkpoints_total_limit)
                    last_saved_step = global_step

            global_loss = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            logs = {"loss": global_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if args.max_train_steps and global_step >= args.max_train_steps:
                break
        progress_bar.close()
        accelerator.wait_for_everyone()

        # All ranks participate in epoch snapshots too (including per-rank RNG).
        if not args.smoke_test and (epoch % 5 == 0 or epoch == args.num_epochs - 1):
            save_checkpoint(accelerator, args.output_dir, f'epoch-{epoch}',
                            checkpoint_metadata, args.checkpoints_total_limit)

        if accelerator.is_main_process:
            if args.enable_visualization and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
                # 创建临时目录保存可视化结果
                vis_dir = os.path.join(args.output_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)

                # 获取原始模型（去掉accelerate包装）
                unwrapped_denoiser = accelerator.unwrap_model(denoiser)
                unwrapped_denoiser.eval()

                # 生成可视化结果
                grid = generate_visualization(
                    denoiser=unwrapped_denoiser,
                    batch=vis_batch,
                    epoch=epoch,
                    output_dir=vis_dir,
                    device=accelerator.device,
                    adapter=adapter,
                )

                # 记录到TensorBoard
                if is_accelerate_version(">=", "0.17.0.dev0"):
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                else:
                    tracker = accelerator.get_tracker("tensorboard")
                log_visualization_to_tensorboard(tracker, grid, epoch)

        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    if not args.smoke_test and (last_saved_step != global_step or args.enable_visualization):
        save_checkpoint(accelerator, args.output_dir, f'checkpoint-{global_step}',
                        dict(checkpoint_metadata, **checkpoint_cursor), args.checkpoints_total_limit)
    smoke_updates = {}
    if args.smoke_test:
        trained = accelerator.unwrap_model(denoiser).model
        for name, before in smoke_before.items():
            after = dict(trained.named_parameters())[name].detach().cpu()
            smoke_updates[name] = not torch.equal(before, after)
        if not all(v for k, v in smoke_updates.items() if not k.startswith('jit_model.x_embedder')):
            raise AssertionError(f'Trainable weights did not update: {smoke_updates}')
        if smoke_updates['jit_model.x_embedder.proj1.weight']:
            raise AssertionError('Frozen backbone weights changed')
    # Reduce a fixed histogram: collectives stay aligned even when ranks see different levels.
    level_counts = accelerator.reduce(level_counts, reduction='sum').cpu().tolist()
    observed_levels = {v for v, count in zip(level_values, level_counts) if count}
    def write_result():
        (Path(args.output_dir) / 'training_result.json').write_text(json.dumps({
            'global_step': global_step, 'loss': logs['loss'],
            'cfa_mode': args.cfa_mode, 'light_mode': args.light_mode,
            'smoke_test': args.smoke_test, 'observed_light_levels': sorted(observed_levels),
            'parameter_updates': smoke_updates, **runtime,
            'scheduler_last_epoch': lr_scheduler.state_dict()['last_epoch'],
            'device': str(accelerator.device),
            'light_level_counts': {str(v): count for v, count in zip(level_values, level_counts)},
        }, indent=2), encoding='utf-8')
    main_action(accelerator, write_result)
    accelerator.end_training()
    logger.info("Training completed!")
