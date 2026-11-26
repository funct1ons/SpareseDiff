"""Small DDP/checkpoint safety helpers; no model or loss changes."""
import json
import os
import re
import random
import shutil
from datetime import timedelta
from pathlib import Path

# Process-local NCCL settings: a rank missing its FIRST communicator must not
# hang in blocking ncclCommInitRank outside the process-group watchdog.
os.environ.setdefault('TORCH_NCCL_USE_COMM_NONBLOCKING', '1')
os.environ.setdefault('TORCH_NCCL_NONBLOCKING_TIMEOUT', '120')

import numpy as np
import torch
from accelerate.utils import DistributedType, InitProcessGroupKwargs, broadcast_object_list, load
from accelerate.utils.constants import SCALER_NAME

SCHEDULER_SEMANTICS = 'prepared-loader-successful-update-v1'
COMPLETION = 'complete.json'
CHECKPOINT_SCHEMA = 2
# Stop/save/output knobs are deliberately excluded: they do not define the
# optimizer schedule or training inputs and may change when continuing a run.
TRAINING_CONFIG_KEYS = (
    'cfa_mode', 'light_mode', 'lowlight_prob', 'lowlight_noise_std', 'resolution',
    'num_layers', 'model_name', 'seed', 'train_batch_size',
    'gradient_accumulation_steps', 'max_train_samples', 'dataset_name',
    'train_data_dir', 'learning_rate', 'lr_warmup_steps', 'lr_scheduler',
    'num_epochs', 'noise_scale', 't_eps', 'num_classes',
    'dataloader_num_workers', 'mixed_precision',
)


def training_configuration(config):
    """Fail closed, including nullable data keys; missing is not equal to None."""
    missing = [key for key in TRAINING_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f'Missing required training configuration: {missing}')
    return {key: config[key] for key in TRAINING_CONFIG_KEYS}


def validate_training_configuration(previous, current):
    previous = training_configuration(previous)
    current = training_configuration(current)
    for key in TRAINING_CONFIG_KEYS:
        if previous[key] != current[key]:
            raise ValueError(f'Training configuration mismatch: {key}: {previous[key]} != {current[key]}')


def process_group_kwargs():
    # Bound a missing/crashed rank's collectives. Elastic launch propagates failures.
    return InitProcessGroupKwargs(timeout=timedelta(seconds=120))


def bound_rendezvous_store():
    """NCCL's first communicator uses the store timeout, not the PG timeout.

    Torch's native launch may leave that store at 600s; bound it after group
    initialization as well. This small compatibility hook is covered on the
    existing server Torch/Accelerate versions (no custom address/port/store).
    """
    if torch.distributed.is_initialized():
        from torch.distributed.distributed_c10d import _get_default_store
        _get_default_store().set_timeout(timedelta(seconds=120))


def validate_execution(accelerator, args):
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU):
        raise ValueError('Only single-process or single-machine CUDA DDP is supported')
    if accelerator.num_processes > 1:
        if accelerator.device.type != 'cuda':
            raise ValueError('DDP requires CUDA')
        if int(os.environ.get('LOCAL_WORLD_SIZE', '0')) != accelerator.num_processes:
            raise ValueError('Only single-machine CUDA DDP is supported')
        if accelerator.num_processes > torch.cuda.device_count():
            raise ValueError('Insufficient visible CUDA devices')
        if args.enable_visualization or args.initial_comparison:
            raise ValueError('DDP does not support visualization or initial comparison')
    bound_rendezvous_store()
    if args.mixed_precision != accelerator.mixed_precision:
        raise ValueError('Requested and effective Accelerate precision differ')


def execution_metadata(accelerator):
    return {'checkpoint_schema': CHECKPOINT_SCHEMA, 'world_size': accelerator.num_processes,
            'effective_precision': accelerator.mixed_precision,
            'scheduler_semantics': SCHEDULER_SEMANTICS,
            'uses_grad_scaler': accelerator.scaler is not None}


def main_action(accelerator, action):
    """Propagate rank-zero filesystem/validation failures before peers continue."""
    result = [None]
    if accelerator.is_main_process:
        try:
            action()
        except Exception as exc:
            result[0] = f'{type(exc).__name__}: {exc}'
    if accelerator.num_processes > 1:
        broadcast_object_list(result)
    if result[0] is not None:
        raise RuntimeError(result[0])


def require_finite(accelerator, value, label):
    bad = (~torch.isfinite(value)).any().to(device=accelerator.device, dtype=torch.int32)
    if accelerator.reduce(bad, reduction='sum').item():
        raise FloatingPointError(f'Non-finite {label} on at least one rank')


def restore_rng_strict(accelerator, path):
    """Accelerate 1.7 catches RNG load errors silently; full resume must not."""
    states = load(Path(path) / f'random_states_{accelerator.process_index}.pkl', map_location='cpu')
    random.setstate(states['random_state'])
    np.random.set_state(states['numpy_random_seed'])
    torch.set_rng_state(states['torch_manual_seed'])
    if accelerator.device.type == 'cuda':
        torch.cuda.set_rng_state_all(states['torch_cuda_manual_seed'])
    if 'step' in states:
        accelerator.step = states['step']


def seed_worker(worker_id, *, rank):
    """Keep shared shuffle indices but independent augmentation streams in workers."""
    seed = (torch.initial_seed() + rank * 1000003) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def numbered_paths(root, prefix):
    return sorted((p for p in Path(root).iterdir()
                   if p.is_dir() and re.fullmatch(rf'{prefix}-\d+', p.name)),
                  key=lambda p: int(p.name.split('-')[1]))


def verify_checkpoint(path, expected, *, legacy=False):
    path = Path(path)
    marker = path / COMPLETION
    source_config = path.parent / 'training_config.json'
    metadata = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else None
    # New manifests are self-contained. Older manifests/no-marker states require
    # the ORIGINAL run config alongside the source, never the destination config.
    if 'training_config' in expected:
        if metadata is not None and metadata.get('checkpoint_schema') == CHECKPOINT_SCHEMA:
            validate_training_configuration(metadata.get('training_config', {}), expected['training_config'])
            if source_config.is_file():
                validate_training_configuration(json.loads(source_config.read_text(encoding='utf-8')),
                                                metadata['training_config'])
        else:
            if not source_config.is_file():
                raise ValueError(f'Missing source training configuration: {source_config}')
            source = json.loads(source_config.read_text(encoding='utf-8'))
            validate_training_configuration(source, expected['training_config'])
            if metadata is None:
                legacy = 'checkpoint_schema' not in source
            if metadata is None and legacy and (expected['world_size'] != 1 or
                                     source['mixed_precision'] != expected['effective_precision']):
                raise ValueError('Legacy resume requires same precision and world_size=1')
    if not marker.is_file():
        if not legacy or expected['world_size'] != 1:
            raise ValueError(f'Incomplete checkpoint (missing completion marker): {path}')
        # Legacy single-process states had no manifest. Require their full core state.
        required = ['optimizer.bin', 'scheduler.bin', 'random_states_0.pkl']
        if not (path / 'model.safetensors').is_file() and not (path / 'pytorch_model.bin').is_file():
            raise ValueError('Incomplete legacy model state')
        if any(not (path / name).is_file() for name in required):
            raise ValueError('Incomplete legacy optimizer/scheduler/RNG state')
        return
    for key, value in expected.items():
        if key == 'training_config':
            continue  # Checked strictly above (including missing/null keys).
        # Schema-1 checkpoints predate the bound config; strict source config
        # verification above is the only migration route used by the engine.
        if key == 'checkpoint_schema' and metadata.get(key) == 1 and value == CHECKPOINT_SCHEMA:
            continue
        if metadata.get(key) != value:
            raise ValueError(f'Checkpoint compatibility mismatch: {key}: {metadata.get(key)} != {value}')
    files = metadata.get('files', {})
    required = ['optimizer.bin', 'scheduler.bin'] + [f'random_states_{r}.pkl' for r in range(expected['world_size'])]
    if metadata.get('uses_grad_scaler'):
        required.append(SCALER_NAME)
    if 'model.safetensors' not in files and 'pytorch_model.bin' not in files:
        raise ValueError('Incomplete checkpoint model manifest')
    if any(name not in files for name in required):
        raise ValueError('Incomplete checkpoint state manifest')
    for name, size in files.items():
        file = path / name
        if not file.is_file() or file.stat().st_size != size or size <= 0:
            raise ValueError(f'Incomplete checkpoint file: {file}')


def resolve_checkpoint(root, requested, expected, *, legacy=False):
    if requested != 'latest':
        path = Path(requested)
        if not path.is_absolute():
            path = Path(root) / path
        if not re.fullmatch(r'checkpoint-\d+', path.name):
            raise ValueError('Resume requires checkpoint-<step>, not epoch snapshots')
        verify_checkpoint(path, expected, legacy=legacy)
        return path
    candidates = numbered_paths(root, 'checkpoint')
    # Incomplete directories from interrupted saves must never become latest.
    for path in reversed(candidates):
        try:
            verify_checkpoint(path, expected, legacy=legacy)
        except (ValueError, OSError):
            continue
        return path
    raise FileNotFoundError('No complete compatible checkpoint found for latest')


def save_checkpoint(accelerator, root, name, metadata, limit):
    path = Path(root) / name
    staging = Path(root) / f'.{name}.incomplete'
    backup = Path(root) / f'.{name}.previous'

    def begin():
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
    main_action(accelerator, begin)
    accelerator.wait_for_everyone()
    # ALL ranks must call save_state: Accelerator writes rank-specific RNG files.
    # Keep an existing completed checkpoint intact throughout a potentially failed save.
    accelerator.save_state(str(staging))
    accelerator.wait_for_everyone()

    def complete_and_retain():
        files = {str(p.relative_to(staging)): p.stat().st_size for p in staging.rglob('*') if p.is_file()}
        marker = dict(metadata, files=files)
        (staging / COMPLETION).write_text(json.dumps(marker, indent=2), encoding='utf-8')
        verify_checkpoint(staging, metadata)
        if backup.exists():
            shutil.rmtree(backup)
        if path.exists():
            path.rename(backup)
        try:
            staging.rename(path)
        except Exception:
            if backup.exists():
                backup.rename(path)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        if limit is not None:
            candidates = numbered_paths(root, name.split('-')[0])
            # Retain only after a new full state exists; never delete the last good state first.
            complete = []
            for candidate in candidates:
                if (candidate / COMPLETION).is_file():
                    complete.append(candidate)
                elif metadata['world_size'] == 1:
                    # Include full old single-process states after metadata migration.
                    try:
                        verify_checkpoint(candidate, {'world_size': 1}, legacy=True)
                    except (ValueError, OSError):
                        continue
                    complete.append(candidate)
            for old in complete[:-limit]:
                shutil.rmtree(old)
    main_action(accelerator, complete_and_retain)
    accelerator.wait_for_everyone()
