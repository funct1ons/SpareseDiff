"""Training CLI: recipe defaults are applied by launch.py, not this parser."""
import argparse
from pathlib import Path
import os


def resolve_mixed_precision(default, explicit_argv):
    """Accelerate's environment wins over implicit recipe/parser defaults only."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default=None)
    explicit, _ = parser.parse_known_args(explicit_argv)
    environment = os.environ.get('ACCELERATE_MIXED_PRECISION')
    if environment is not None:
        if environment not in ('no', 'fp16', 'bf16'):
            parser.error('Unsupported ACCELERATE_MIXED_PRECISION')
        if explicit.mixed_precision is not None and explicit.mixed_precision != environment:
            parser.error('Explicit CLI mixed_precision conflicts with Accelerate mixed precision')
        return environment
    return explicit.mixed_precision or default


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sparse/Bayer 解马赛克扩散模型训练")

    # 数据参数
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="训练数据目录（当dataset_name为None时使用）"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="输入图像分辨率"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="训练批次大小"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="数据加载工作线程数"
    )

    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demosaic-model",
        help="输出目录"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="训练轮数"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=1000,
        help="学习率预热步数"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help="保存检查点的步数间隔"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )

    # 模型参数
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="预训练模型检查点路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="JiT-B/16",
        choices=["JiT-B/16", "JiT-B/32", "JiT-L/16", "JiT-L/32"],
        help="JiT模型类型"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="类别数"
    )

    # 训练阶段
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="解冻的jit层数"
    )
    # 扩散参数
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="噪声尺度"
    )
    parser.add_argument(
        "--t_eps",
        type=float,
        default=1e-2,
        help="时间步最小值"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=5,
        help="每多少个epoch保存一次可视化结果"
    )

    parser.add_argument(
        "--num_visualization_samples",
        type=int,
        default=4,
        help="可视化显示的样本数量"
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="最大检查点数量"
    )
    parser.add_argument('--light_mode', choices=['normal', 'mixed'], required=True)
    parser.add_argument('--lowlight_prob', type=float, default=0.6)
    parser.add_argument('--lowlight_noise_std', type=float, default=0.004)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_train_steps', type=int, default=0)
    parser.add_argument('--max_train_samples', type=int, default=0)
    parser.add_argument('--smoke_test', action='store_true', help='Synthetic data/random weights; no checkpoints')
    parser.add_argument('--enable_visualization', action='store_true')
    parser.add_argument('--initial_comparison', action='store_true')
    parser.add_argument("--cfa_mode", required=True, choices=["sparse", "bayer"])
    args = parser.parse_args(argv)
    if args.model_name != 'JiT-B/16':
        parser.error('These four experiments use JiT-B/16 only')
    if args.resolution < 32 or args.resolution % 16:
        parser.error('resolution must be >=32 and divisible by16')
    if not 0 <= args.lowlight_prob <= 1 or args.lowlight_noise_std < 0:
        parser.error('Invalid low-light settings')
    if min(args.num_epochs, args.train_batch_size, args.gradient_accumulation_steps, args.checkpointing_steps, args.save_images_epochs) <= 0:
        parser.error('Epochs, batch size, accumulation and save intervals must be positive')
    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit < 1:
        parser.error('checkpoints_total_limit must be positive')
    if args.max_train_steps < 0 or args.max_train_samples < 0:
        parser.error('max_train_steps cannot be negative')
    if int(__import__('os').environ.get('WORLD_SIZE', '1')) > 1 and (args.enable_visualization or args.initial_comparison):
        parser.error('DDP does not support visualization or initial comparison')
    if args.smoke_test:
        if args.resume_from_checkpoint or args.pretrained_checkpoint:
            parser.error('smoke_test uses random weights, not pretrained/resume checkpoints')
        args.max_train_steps = args.max_train_steps or 4
        args.enable_visualization = False
        args.initial_comparison = False
    elif not args.pretrained_checkpoint or not Path(args.pretrained_checkpoint).is_file():
        parser.error('A valid pretrained_checkpoint is required outside smoke_test')
    if not args.smoke_test and not (args.dataset_name or args.train_data_dir):
        parser.error('Provide dataset_name or train_data_dir')


    return args
