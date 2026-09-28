"""Train BatchTopK heterogeneous crosscoders on cached activations.

Defaults reproduce the paper's crosscoders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from contextlib import suppress
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from uuid import uuid4

import torch

from dictionary_learning import (
    BatchTopKHeterogeneousCrossCoder,
    HeterogeneousActivationBatchDataset,
    HeterogeneousActivationCacheTuple,
)
from dictionary_learning.trainers import BatchTopKHeterogeneousCrossCoderTrainer
from dictionary_learning.training import run_validation, trainSAE

COMBINATIONS = {
    "base-opd-teacher": ("base", "opd", "teacher"),
    "base-teacher": ("base", "teacher"),
    "opd-teacher": ("opd", "teacher"),
    "base-opd": ("base", "opd"),
}
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pt$")
CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_SEMANTICS = (
    "checkpoint_N contains state after N completed optimizer updates; "
    "the next trainer update uses zero-based global step N"
)
TRAINER_STATE_FIELDS = (
    "num_tokens_since_fired",
    "steps_since_active",
    "effective_l0",
    "running_deads",
    "pre_norm_auxk_loss",
    "k_current_value",
)
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class Prefetcher:
    """Read batches ahead on a background thread so storage latency overlaps with compute."""

    _DONE = object()

    def __init__(self, iterable, depth: int = 4):
        self.iterable = iterable
        self.depth = max(1, int(depth))

    def __len__(self):
        return len(self.iterable)

    @property
    def config(self):
        return self.iterable.config

    def __iter__(self):
        queue: Queue = Queue(maxsize=self.depth)
        stop = Event()

        def offer(item) -> bool:
            # Poll so that the thread exits once the trainer stops consuming.
            while not stop.is_set():
                try:
                    queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def produce():
            try:
                for item in self.iterable:
                    if not offer(item):
                        return
            except Exception as error:
                offer(error)
            finally:
                offer(self._DONE)

        worker = Thread(target=produce, daemon=True)
        worker.start()
        try:
            while True:
                item = queue.get()
                if item is self._DONE:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            with suppress(Empty):
                while True:
                    queue.get_nowait()
            worker.join(timeout=5)


class _ResumedStream:
    """Skip the first ``start`` batches of a stream without reading their activations."""

    def __init__(self, dataset: HeterogeneousActivationBatchDataset, start: int):
        self.dataset = dataset
        self.start = int(start)

    def __len__(self):
        return max(0, _effective_stream_steps(self.dataset) - self.start)

    @property
    def config(self):
        return {**self.dataset.config, "resume_start_step": self.start}

    def _iter_group(self, group, generator, skipped):
        dataset = self.dataset
        per_shard = max(1, dataset.batch_size // len(group))
        cursors = [0] * len(group)
        counts = [dataset._row_count(shard_idx) for shard_idx in group]
        while True:
            slices = []
            for position, shard_idx in enumerate(group):
                stop = min(cursors[position] + per_shard, counts[position])
                if stop > cursors[position]:
                    slices.append((shard_idx, cursors[position], stop))
                    cursors[position] = stop
            if not slices:
                return skipped
            rows = sum(stop - start for _, start, stop in slices)
            if dataset.drop_last and rows < dataset.batch_size:
                return skipped
            order = None
            if dataset.shuffle and len(slices) > 1:
                # Draw the permutation even for skipped batches to keep the generator in sync.
                order = torch.randperm(rows, generator=generator)
            if skipped:
                skipped -= 1
                continue
            blocks = [dataset._read_block(shard_idx, start, stop) for shard_idx, start, stop in slices]
            batch = tuple(
                torch.cat([block[source] for block in blocks], dim=0)
                for source in range(len(dataset.caches.activation_caches))
            )
            if order is not None:
                batch = tuple(source[order] for source in batch)
            yield batch

    def __iter__(self):
        dataset = self.dataset
        generator = torch.Generator().manual_seed(dataset.seed + dataset._epoch)
        dataset._epoch += 1
        shard_order = list(dataset.shard_indices)
        if dataset.shuffle:
            permutation = torch.randperm(len(shard_order), generator=generator).tolist()
            shard_order = [shard_order[idx] for idx in permutation]

        skipped = self.start
        if dataset.interleave_groups > 1:
            width = dataset.interleave_groups
            for start in range(0, len(shard_order), width):
                iterator = self._iter_group(shard_order[start : start + width], generator, skipped)
                while True:
                    try:
                        yield next(iterator)
                        skipped = 0
                    except StopIteration as done:
                        skipped = done.value or 0
                        break
            return

        for shard_idx in shard_order:
            row_count = dataset._row_count(shard_idx)
            row_indices = torch.randperm(row_count, generator=generator) if dataset.shuffle else torch.arange(row_count)
            for start in range(0, row_count, dataset.batch_size):
                stop = min(start + dataset.batch_size, row_count)
                if dataset.drop_last and stop - start < dataset.batch_size:
                    continue
                if skipped:
                    skipped -= 1
                    continue
                selection = row_indices[start:stop].numpy()
                yield tuple(cache.shards[shard_idx][selection] for cache in dataset.caches.activation_caches)


def _resume_data(data, completed_steps: int):
    if completed_steps == 0:
        return data
    if isinstance(data, Prefetcher):
        return Prefetcher(_resume_data(data.iterable, completed_steps), data.depth)
    return _ResumedStream(data, completed_steps)


def _effective_stream_steps(dataset: HeterogeneousActivationBatchDataset, *, epoch: int | None = None) -> int:
    """Count the batches one epoch yields without reading activations."""

    if epoch is None:
        epoch = dataset._epoch
    generator = torch.Generator().manual_seed(dataset.seed + epoch)
    shard_order = list(dataset.shard_indices)
    if dataset.shuffle:
        permutation = torch.randperm(len(shard_order), generator=generator).tolist()
        shard_order = [shard_order[index] for index in permutation]

    if dataset.interleave_groups <= 1:
        return sum(
            dataset._row_count(shard_idx) // dataset.batch_size
            if dataset.drop_last
            else (dataset._row_count(shard_idx) + dataset.batch_size - 1) // dataset.batch_size
            for shard_idx in shard_order
        )

    yielded = 0
    width = dataset.interleave_groups
    for start in range(0, len(shard_order), width):
        group = shard_order[start : start + width]
        per_shard = max(1, dataset.batch_size // len(group))
        remaining = [dataset._row_count(shard_idx) for shard_idx in group]
        while True:
            rows = sum(min(per_shard, count) for count in remaining)
            if rows == 0 or (dataset.drop_last and rows < dataset.batch_size):
                break
            yielded += 1
            remaining = [max(0, count - per_shard) for count in remaining]
    return yielded


def _build_loaders(args, caches):
    """Hold out shards spread evenly over the stream and read the rest in groups of interleaved shards."""

    shard_count = len(caches.activation_caches[0].shards)
    if args.validation_shards <= 0 or args.validation_shards >= shard_count:
        raise ValueError("validation_shards must be positive and smaller than the shard count")
    validation_shards = sorted(
        set(torch.linspace(0, shard_count - 1, args.validation_shards).round().long().tolist())
    )
    train_shards = [i for i in range(shard_count) if i not in set(validation_shards)]
    train_data = HeterogeneousActivationBatchDataset(
        caches,
        batch_size=args.batch_size,
        shard_indices=train_shards,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
        interleave_groups=args.interleave_groups,
    )
    validation_data = HeterogeneousActivationBatchDataset(
        caches,
        batch_size=args.validation_batch_size,
        shard_indices=validation_shards,
        shuffle=False,
        seed=args.seed,
        drop_last=False,
        interleave_groups=1,
    )
    schedule_steps = len(train_data)
    effective_steps = _effective_stream_steps(train_data, epoch=0)
    if args.max_steps is not None:
        schedule_steps = min(schedule_steps, args.max_steps)
        effective_steps = min(effective_steps, args.max_steps)
    if args.prefetch > 0:
        train_data = Prefetcher(train_data, args.prefetch)
    return train_data, validation_data, effective_steps, {
        "loader": "stream",
        "schedule_steps": schedule_steps,
        "effective_steps": effective_steps,
        "interleave_groups": args.interleave_groups,
        "prefetch": args.prefetch,
        "validation_shards": validation_shards,
    }


def _checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            step = _checkpoint_step(path)
            if step is not None and path.is_file():
                candidates.append((step, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def _load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError("checkpoint does not contain a recognizable model state")


def _copy_trainer_value(current, saved, device):
    if torch.is_tensor(current):
        current.copy_(torch.as_tensor(saved, device=current.device))
        return current
    if torch.is_tensor(saved):
        return saved.to(device)
    return saved


def _capture_trainer_state(trainer) -> dict:
    state = {}
    for name in TRAINER_STATE_FIELDS:
        if not hasattr(trainer, name):
            continue
        value = getattr(trainer, name)
        if torch.is_tensor(value):
            value = value.detach().clone()
        elif not (isinstance(value, (bool, int, float, str)) or value is None):
            continue
        state[name] = value
    return state


def _move_optimizer_state(optimizer, device) -> None:
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if torch.is_tensor(value):
                parameter_state[key] = value.to(device)


def _reconstruct_scheduler(scheduler, completed_steps: int) -> None:
    if completed_steps <= 0:
        return
    if hasattr(scheduler, "lr_lambdas"):
        scheduler.last_epoch = completed_steps
        scheduler._step_count = completed_steps + 1
        lrs = [base_lr * fn(completed_steps) for base_lr, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
        for group, lr in zip(scheduler.optimizer.param_groups, lrs):
            group["lr"] = lr
        scheduler._last_lr = lrs
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler.step(completed_steps)


def _restore_training_checkpoint(trainer, path: Path) -> tuple[int, dict]:
    checkpoint = _load_checkpoint(path)
    trainer.ae.load_state_dict(_model_state(checkpoint))
    completed_steps = int(checkpoint.get("completed_steps", _checkpoint_step(path)))
    restored = {
        "checkpoint": str(path),
        "completed_steps": completed_steps,
        "optimizer": False,
        "scheduler": False,
        "trainer_state": [],
        "rng": False,
    }
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        trainer.optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state(trainer.optimizer, trainer.device)
        restored["optimizer"] = True
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        trainer.scheduler.load_state_dict(scheduler_state)
        restored["scheduler"] = True
    else:
        _reconstruct_scheduler(trainer.scheduler, completed_steps)
        restored["scheduler"] = "reconstructed_from_step"
    for name, saved in checkpoint.get("trainer_state", {}).items():
        if hasattr(trainer, name):
            setattr(trainer, name, _copy_trainer_value(getattr(trainer, name), saved, trainer.device))
            restored["trainer_state"].append(name)
    rng_state = checkpoint.get("rng_state", {})
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"].cpu())
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        restored["rng"] = True
    return completed_steps, restored


def _save_new(value, path: Path) -> None:
    """Write a torch file atomically, refusing to replace an existing one."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_training_checkpoint(trainer, path: Path, completed_steps: int) -> None:
    _save_new(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_semantics": CHECKPOINT_SEMANTICS,
            "completed_steps": int(completed_steps),
            "model_state_dict": trainer.ae.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "trainer_state": _capture_trainer_state(trainer),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


class _StepOffsetTrainer:
    """Shift trainSAE's local step count to the global step of a resumed run."""

    def __init__(self, trainer, offset: int):
        self._trainer = trainer
        self._offset = int(offset)

    def __getattr__(self, name):
        return getattr(self._trainer, name)

    def update(self, step, activations):
        return self._trainer.update(self._offset + step, activations)

    def loss(self, *args, step=None, **kwargs):
        if step is not None:
            step += self._offset
        return self._trainer.loss(*args, step=step, **kwargs)


class _TrainerFactory:
    def __init__(self, trainer_class, checkpoint: Path | None, offset: int):
        self.trainer_class = trainer_class
        self.checkpoint = checkpoint
        self.offset = int(offset)
        self.trainer = None
        self.restore_info = None

    def __call__(self, **kwargs):
        self.trainer = self.trainer_class(**kwargs)
        if self.checkpoint is not None:
            restored_step, self.restore_info = _restore_training_checkpoint(self.trainer, self.checkpoint)
            if restored_step != self.offset:
                raise ValueError(f"checkpoint restored step {restored_step}, expected {self.offset}")
        return _StepOffsetTrainer(self.trainer, self.offset)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _validate_existing_run(existing: dict, current: dict) -> None:
    identity_fields = (
        "combination",
        "sources",
        "source_layers",
        "source_models",
        "activation_dims",
        "token_stream_hash",
        "steps",
        "batch_size",
        "training_dtype",
        "loader",
    )
    mismatches = {
        field: {"existing": existing.get(field), "requested": current.get(field)}
        for field in identity_fields
        if existing.get(field) != current.get(field)
    }
    if mismatches:
        raise ValueError("resume configuration does not match the original run: " + json.dumps(mismatches, sort_keys=True))


def _resume_checkpoint(args, output_dir: Path) -> Path | None:
    if args.resume_from is not None:
        checkpoint = Path(args.resume_from).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        latest = _latest_checkpoint(output_dir)
        if latest is not None and _checkpoint_step(latest) > _checkpoint_step(checkpoint):
            raise ValueError(f"{output_dir} already has newer checkpoint {latest.name}")
        return checkpoint
    if args.resume_latest:
        checkpoint = _latest_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"no checkpoint found in {output_dir}")
        return checkpoint
    return None


def train_one(args, combination: str) -> None:
    sources = COMBINATIONS[combination]
    caches = HeterogeneousActivationCacheTuple(
        *(str(Path(args.cache_root) / source) for source in sources),
        require_token_hash=True,
    )
    train_data, validation_data, target_updates, loader_config = _build_loaders(args, caches)
    schedule_steps = int(loader_config["schedule_steps"])
    run_name = f"{combination}-batchtopk-seed{args.seed}"
    output_dir = Path(args.output_root) / run_name
    source_layers = tuple(0 if layer is None else int(layer) for layer in caches.source_layers)
    trainer_config = {
        "trainer": BatchTopKHeterogeneousCrossCoderTrainer,
        "dict_class": BatchTopKHeterogeneousCrossCoder,
        # The learning-rate schedule spans the nominal step count even if interleaved
        # groups end a few batches early.
        "steps": schedule_steps,
        "k": args.k,
        "lr": args.lr,
        "auxk_alpha": args.auxk_alpha,
        "warmup_steps": min(args.warmup_steps, max(schedule_steps - 1, 0)),
        "decay_start": int(schedule_steps * args.decay_fraction) if args.decay_fraction is not None else None,
        "threshold_start_step": args.threshold_start_step,
        "dict_class_kwargs": {"norm_init_scale": 1.0, "init_with_transpose": True},
        "activation_dims": caches.activation_dims,
        "dict_size": args.dict_size,
        "source_names": sources,
        "source_layers": source_layers,
        "seed": args.seed,
        "device": args.device,
        "wandb_name": run_name,
        "activation_mean": caches.mean,
        "activation_std": caches.std,
        "target_rms": args.target_rms,
    }
    run_config = {
        "combination": combination,
        "sources": list(sources),
        "source_layers": list(source_layers),
        "source_models": list(caches.source_models),
        "activation_dims": list(caches.activation_dims),
        "token_stream_hash": caches.token_stream_hash,
        "steps": schedule_steps,
        "schedule_steps": schedule_steps,
        "effective_steps": target_updates,
        "target_updates": target_updates,
        "batch_size": args.batch_size,
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "seed": args.seed,
        "training_dtype": args.training_dtype,
        "checkpoint_semantics": CHECKPOINT_SEMANTICS,
        **loader_config,
    }

    resume_requested = args.resume_latest or args.resume_from is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        if not resume_requested:
            raise FileExistsError(f"run already exists at {output_dir}; pass --resume-latest or --resume-from")
        existing = json.loads(run_config_path.read_text())
        _validate_existing_run(existing, run_config)
        run_config = {**existing, **run_config}
    elif any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to train in non-empty run directory: {output_dir}")

    checkpoint = _resume_checkpoint(args, output_dir)
    completed_steps = 0
    if checkpoint is not None:
        checkpoint_data = _load_checkpoint(checkpoint)
        completed_steps = int(checkpoint_data.get("completed_steps", _checkpoint_step(checkpoint)))
        if completed_steps > target_updates:
            raise ValueError(f"checkpoint has {completed_steps} updates, beyond the target {target_updates}")
        if completed_steps == target_updates:
            final_model = output_dir / "model_final.pt"
            if not final_model.exists():
                _save_new(_model_state(checkpoint_data), final_model)
            run_config.update({"status": "completed", "completed_steps": target_updates})
            _write_json(run_config_path, run_config)
            print(json.dumps({"run": run_name, "status": "already_completed"}, indent=2))
            return

    train_data = _resume_data(train_data, completed_steps)
    factory = _TrainerFactory(trainer_config["trainer"], checkpoint, completed_steps)
    trainer_config["trainer"] = factory
    run_config.update(
        {
            "status": "running",
            "completed_steps": completed_steps,
            "target_updates": target_updates,
            "resume_checkpoint": str(checkpoint) if checkpoint is not None else None,
        }
    )
    _write_json(run_config_path, run_config)
    print(json.dumps({"run": run_name, **run_config, "remaining_steps": target_updates - completed_steps}, indent=2))

    last_checkpoint_step = completed_steps

    def after_step(trainer, local_step):
        nonlocal last_checkpoint_step
        global_completed = completed_steps + local_step + 1
        if args.save_every and global_completed % args.save_every == 0:
            _save_training_checkpoint(trainer._trainer, output_dir / f"checkpoint_{global_completed}.pt", global_completed)
            last_checkpoint_step = global_completed
        if args.validate_every and global_completed % args.validate_every == 0:
            logs = run_validation(trainer._trainer, validation_data, step=global_completed, dtype=DTYPES[args.training_dtype])
            _save_new(logs, output_dir / f"eval_logs_{global_completed}.pt")

    try:
        trainSAE(
            data=train_data,
            trainer_config=trainer_config,
            use_wandb=not args.disable_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_group=Path(args.output_root).name or args.wandb_project,
            steps=target_updates - completed_steps,
            # Checkpoints are written by after_step with global step numbers.
            save_steps=None,
            save_dir=str(output_dir),
            log_steps=args.log_every,
            validate_every_n_steps=None,
            validation_data=validation_data,
            dtype=DTYPES[args.training_dtype],
            end_of_step_logging_fn=after_step,
        )
        final_checkpoint = output_dir / f"checkpoint_{target_updates}.pt"
        if last_checkpoint_step != target_updates:
            _save_training_checkpoint(factory.trainer, final_checkpoint, target_updates)
        run_config.update(
            {
                "status": "completed",
                "completed_steps": target_updates,
                "last_checkpoint": str(final_checkpoint),
                "resume_restore": factory.restore_info,
            }
        )
        _write_json(run_config_path, run_config)
    except Exception as error:
        run_config.update(
            {
                "status": "failed",
                "completed_steps": last_checkpoint_step,
                "last_error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(run_config_path, run_config)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, help="directory with one activation cache per slot: base/, opd/, teacher/")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--combinations", nargs="+", choices=tuple(COMBINATIONS), default=["base-opd-teacher"])
    parser.add_argument("--dict-size", type=int, default=32768)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--lr", type=float, help="defaults to 2e-4 / sqrt(dict_size / 16384)")
    parser.add_argument("--auxk-alpha", type=float, default=1 / 32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--validation-batch-size", type=int, default=8192)
    parser.add_argument("--validation-shards", type=int, default=4)
    parser.add_argument("--interleave-groups", type=int, default=16, help="shards read at once, so each batch spans distant parts of the stream")
    parser.add_argument("--prefetch", type=int, default=4, help="batches read ahead on a background thread; 0 disables")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--threshold-start-step", type=int, default=1000)
    parser.add_argument("--decay-fraction", type=float, help="start of linear LR decay as a fraction of training; no decay by default")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--validate-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-latest", action="store_true")
    resume.add_argument("--resume-from")
    parser.add_argument("--target-rms", type=float, default=1.0)
    parser.add_argument("--training-dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-project", default="opd-crosscoder")
    args = parser.parse_args()
    if args.resume_from is not None and len(args.combinations) != 1:
        parser.error("--resume-from requires exactly one combination")
    return args


def main():
    args = parse_args()
    for combination in args.combinations:
        train_one(args, combination)


if __name__ == "__main__":
    main()
