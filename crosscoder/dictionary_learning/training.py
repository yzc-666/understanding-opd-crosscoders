"""
Training dictionaries
"""

import json
import os
from collections import defaultdict
import torch as th
from tqdm import tqdm
from warnings import warn
import wandb
from typing import List, Optional

from .trainers.batch_top_k import BatchTopKTrainer
from .trainers.crosscoder import CrossCoderTrainer, BatchTopKCrossCoderTrainer


def _is_activation_sequence(value):
    return isinstance(value, (tuple, list))


def _move_activation_batch(value, device, dtype):
    if _is_activation_sequence(value):
        return tuple(item.to(device=device, dtype=dtype) for item in value)
    return value.to(device=device, dtype=dtype)


def _activation_batch_size(value):
    return value[0].shape[0] if _is_activation_sequence(value) else value.shape[0]


def get_stats(
    trainer,
    act: th.Tensor,
    deads_sum: bool = True,
    step: int = None,
    use_threshold: bool = True,
):
    with th.no_grad():
        act, act_hat, f, losslog = trainer.loss(
            act, step=step, logging=True, return_deads=True, use_threshold=use_threshold
        )

    # L0
    l0 = (f != 0).float().detach().cpu().sum(dim=-1).mean().item()

    out = {
        "l0": l0,
        **{f"{k}": v for k, v in losslog.items() if k != "deads"},
    }
    if "deads" in losslog and losslog["deads"] is not None:
        total_feats = losslog["deads"].shape[0]
        out["frac_deads"] = (
            losslog["deads"].sum().item() / total_feats
            if deads_sum
            else losslog["deads"]
        )

    # fraction of variance explained
    if _is_activation_sequence(act):
        total_variance_per_source = []
        residual_variance_per_source = []
        for source_idx, (source, reconstruction) in enumerate(zip(act, act_hat)):
            total = th.var(source, dim=0).sum()
            residual = th.var(source - reconstruction, dim=0).sum()
            total_variance_per_source.append(total)
            residual_variance_per_source.append(residual)
            out[f"cl{source_idx}_frac_variance_explained"] = 1 - residual / total
        total_variance = sum(total_variance_per_source)
        residual_variance = sum(residual_variance_per_source)
        frac_variance_explained = 1 - residual_variance / total_variance
    elif act.dim() == 2:
        # act.shape: [batch, d_model]
        # fraction of variance explained
        total_variance = th.var(act, dim=0).sum()
        residual_variance = th.var(act - act_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
    else:
        # act.shape: [batch, layer, d_model]
        total_variance_per_layer = []
        residual_variance_per_layer = []

        for l in range(act_hat.shape[1]):
            total_variance_per_layer.append(th.var(act[:, l, :], dim=0).cpu().sum())
            residual_variance_per_layer.append(
                th.var(act[:, l, :] - act_hat[:, l, :], dim=0).cpu().sum()
            )
            out[f"cl{l}_frac_variance_explained"] = (
                1 - residual_variance_per_layer[l] / total_variance_per_layer[l]
            )
        total_variance = sum(total_variance_per_layer)
        residual_variance = sum(residual_variance_per_layer)
        frac_variance_explained = 1 - residual_variance / total_variance

    out["frac_variance_explained"] = frac_variance_explained.item()
    return out


def get_model(trainer):
    if hasattr(trainer, "ae"):
        model = trainer.ae
    else:
        model = trainer.model
    if hasattr(model, "_orig_mod"):  # Check if model is compiled
        model = model._orig_mod
    return model


def log_stats(
    trainer,
    step: int,
    act: th.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    stage: str = "train",
    use_threshold: bool = True,
    epoch_idx_per_step: Optional[List[int]] = None,
    num_tokens: int = None,
):
    with th.no_grad():
        log = {}
        if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
            if _is_activation_sequence(act):
                raise ValueError(
                    "activations_split_by_head is not supported for heterogeneous batches"
                )
            act = act[..., 0, :]
        if not transcoder:
            stats = get_stats(trainer, act, step=step, use_threshold=use_threshold)
            log.update({f"{stage}/{k}": v for k, v in stats.items()})
        else:  # transcoder
            x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)
            # L0
            l0 = (f != 0).float().sum(dim=-1).mean().item()
            log[f"{stage}/l0"] = l0

        # log parameters from training
        log["step"] = step
        trainer_log = trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            log[f"{stage}/{name}"] = value

        if epoch_idx_per_step is not None:
            log["epoch"] = epoch_idx_per_step[step]
        if num_tokens is not None:
            log["num_tokens"] = num_tokens
        wandb.log(log, step=step)


@th.no_grad()
def run_validation(
    trainer,
    validation_data,
    step: int = None,
    dtype: th.dtype = th.float32,
    epoch_idx_per_step: Optional[List[int]] = None,
):
    l0 = []
    frac_variance_explained = []
    frac_variance_explained_per_feature = []
    deads = []
    if getattr(trainer, "is_crosscoder_trainer", False) or isinstance(
        trainer, (CrossCoderTrainer, BatchTopKCrossCoderTrainer)
    ):
        frac_variance_explained_per_layer = defaultdict(list)
    for val_step, act in enumerate(tqdm(validation_data, total=len(validation_data))):
        act = _move_activation_batch(act, trainer.device, dtype)
        stats = get_stats(trainer, act, deads_sum=False, step=step)
        l0.append(stats["l0"])
        if "frac_deads" in stats:
            deads.append(stats["frac_deads"])
        if "frac_variance_explained" in stats:
            frac_variance_explained.append(stats["frac_variance_explained"])
        if "frac_variance_explained_per_feature" in stats:
            frac_variance_explained_per_feature.append(
                stats["frac_variance_explained_per_feature"]
            )

        if getattr(trainer, "is_crosscoder_trainer", False) or isinstance(
            trainer, (CrossCoderTrainer, BatchTopKCrossCoderTrainer)
        ):
            source_count = len(act) if _is_activation_sequence(act) else act.shape[1]
            for l in range(source_count):
                if f"cl{l}_frac_variance_explained" in stats:
                    frac_variance_explained_per_layer[l].append(
                        stats[f"cl{l}_frac_variance_explained"]
                    )
    log = {}
    if getattr(trainer, "is_crosscoder_trainer", False) or isinstance(
        trainer, (CrossCoderTrainer, BatchTopKCrossCoderTrainer)
    ):
        if hasattr(trainer.ae, "decoder_norms"):
            dec_norms = trainer.ae.decoder_norms(p=2)
        else:
            dec_norms = trainer.ae.decoder.weight.norm(dim=-1)
        dec_norms_sum = dec_norms.sum(dim=0)
        for layer_idx in range(dec_norms.shape[0]):
            dec_norm_diff = 0.5 * (
                (2 * dec_norms[layer_idx] - dec_norms_sum)
                / th.maximum(dec_norms[layer_idx], dec_norms_sum - dec_norms[layer_idx])
                + 1
            )
            num_layer_specific_latents = (dec_norm_diff > 0.9).sum().item()
            log[f"val/num_specific_latents_l{layer_idx}"] = num_layer_specific_latents
    if len(deads) > 0:
        log["val/frac_deads"] = th.stack(deads).all(dim=0).float().mean().item()
    if len(l0) > 0:
        log["val/l0"] = th.tensor(l0).mean().item()
    if len(frac_variance_explained) > 0:
        log["val/frac_variance_explained"] = th.tensor(frac_variance_explained).mean()
    if len(frac_variance_explained_per_feature) > 0:
        frac_variance_explained_per_feature = th.stack(
            frac_variance_explained_per_feature
        ).cpu()  # [num_features]
        log["val/frac_variance_explained_per_feature"] = (
            frac_variance_explained_per_feature
        )
    if getattr(trainer, "is_crosscoder_trainer", False) or isinstance(
        trainer, (CrossCoderTrainer, BatchTopKCrossCoderTrainer)
    ):
        for l in frac_variance_explained_per_layer:
            log[f"val/cl{l}_frac_variance_explained"] = th.tensor(
                frac_variance_explained_per_layer[l]
            ).mean()
    if step is not None:
        log["step"] = step
        if epoch_idx_per_step is not None:
            log["epoch"] = epoch_idx_per_step[step]
    wandb.log(log, step=step)

    return log


def save_model(trainer, checkpoint_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model = get_model(trainer)
    th.save(model.state_dict(), os.path.join(save_dir, checkpoint_name))


def trainSAE(
    data,
    trainer_config,
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    wandb_group="",
    steps=None,
    save_steps=None,
    save_dir=None,
    log_steps=None,
    activations_split_by_head=False,
    validate_every_n_steps=None,
    validation_data=None,
    transcoder=False,
    run_cfg={},
    end_of_step_logging_fn=None,
    save_last_eval=True,
    start_of_training_eval=False,
    dtype=th.float32,
    run_wandb_finish=True,
    epoch_idx_per_step: Optional[List[int]] = None,
    return_last_eval_logs=False,
):
    """
    Train SAE using the given trainer

    Args:
        data: Training data iterator/dataloader
        trainer_config: Configuration dictionary for the trainer
        use_wandb: Whether to use Weights & Biases logging (default: False)
        wandb_entity: W&B entity name (default: "")
        wandb_project: W&B project name (default: "")
        wandb_group: W&B group name (default: "")
        steps: Maximum number of training steps (default: None)
        save_steps: Frequency of model checkpointing (default: None)
        save_dir: Directory to save checkpoints and config (default: None)
        log_steps: Frequency of logging statistics (default: None)
        activations_split_by_head: Whether activations are split by attention head (default: False)
        validate_every_n_steps: Frequency of validation evaluation (default: None)
        validation_data: Validation data iterator/dataloader (default: None)
        transcoder: Whether training a transcoder model (default: False)
        run_cfg: Additional run configuration (default: {})
        end_of_step_logging_fn: Custom logging function called at end of each step (default: None)
        save_last_eval: Whether to save evaluation results at end of training (default: True)
        start_of_training_eval: Whether to run evaluation before training starts (default: False)
        dtype: Training data type (default: torch.float32)
        run_wandb_finish: Whether to call wandb.finish() at end of training (default: True)
        epoch_idx_per_step: Optional mapping of training steps to epoch indices (default: None). Mainly used for logging when the dataset is pre-shuffled and contains multiple epochs.

    Returns:
        Trained model

    Raises:
        AssertionError: If validation_data is None but validate_every_n_steps is specified
    """
    assert not (
        validation_data is None and validate_every_n_steps is not None
    ), "Must provide validation data if validate_every_n_steps is not None"

    trainer_class = trainer_config["trainer"]
    del trainer_config["trainer"]
    trainer = trainer_class(**trainer_config)

    wandb_config = trainer.config | run_cfg
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config=wandb_config,
        name=wandb_config["wandb_name"],
        mode="disabled" if not use_wandb else "online",
        group=wandb_group,
    )

    trainer.model.to(dtype)

    # make save dir, export config
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # save config
        config = {"trainer": trainer.config}
        try:
            config["buffer"] = data.config
        except Exception as e:
            warn(f"Error saving config: {e}")
            pass
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    num_tokens = 0
    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break
        act = _move_activation_batch(act, trainer.device, dtype)
        num_tokens += _activation_batch_size(act)
        # logging
        if log_steps is not None and step % log_steps == 0 and step != 0:
            with th.no_grad():
                log_stats(
                    trainer,
                    step,
                    act,
                    activations_split_by_head,
                    transcoder,
                    use_threshold=False,
                    epoch_idx_per_step=epoch_idx_per_step,
                    num_tokens=num_tokens,
                )
                if (
                    getattr(trainer, "is_batch_topk_trainer", False)
                    or isinstance(trainer, BatchTopKCrossCoderTrainer)
                    or isinstance(trainer, BatchTopKTrainer)
                ):
                    log_stats(
                        trainer,
                        step,
                        act,
                        activations_split_by_head,
                        transcoder,
                        use_threshold=True,
                        stage="trainthres",
                        epoch_idx_per_step=epoch_idx_per_step,
                        num_tokens=num_tokens,
                    )

        # saving
        if save_steps is not None and step % save_steps == 0:
            print(f"Saving at step {step}")
            if save_dir is not None:
                save_model(trainer, f"checkpoint_{step}.pt", save_dir)

        # training
        trainer.update(step, act)

        if (
            validate_every_n_steps is not None
            and step % validate_every_n_steps == 0
            and (start_of_training_eval or step > 0)
        ):
            print(f"Validating at step {step}")
            logs = run_validation(
                trainer,
                validation_data,
                step=step,
                dtype=dtype,
                epoch_idx_per_step=epoch_idx_per_step,
            )
            try:
                os.makedirs(save_dir, exist_ok=True)
                th.save(logs, os.path.join(save_dir, f"eval_logs_{step}.pt"))
            except:
                pass

        if end_of_step_logging_fn is not None:
            end_of_step_logging_fn(trainer, step)
    try:
        last_eval_logs = run_validation(
            trainer,
            validation_data,
            step=step,
            dtype=dtype,
            epoch_idx_per_step=epoch_idx_per_step,
        )
        if save_last_eval:
            os.makedirs(save_dir, exist_ok=True)
            th.save(last_eval_logs, os.path.join(save_dir, f"last_eval_logs.pt"))
    except Exception as e:
        print(f"Error during final validation: {str(e)}")

    # save final SAE
    if save_dir is not None:
        save_model(trainer, f"model_final.pt", save_dir)

    if use_wandb and run_wandb_finish:
        wandb.finish()

    if return_last_eval_logs:
        return get_model(trainer), last_eval_logs
    else:
        return get_model(trainer)
