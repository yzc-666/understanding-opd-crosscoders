"""
Implements the CrossCoder training scheme.
"""

import torch as th
from ..trainers.trainer import SAETrainer
from ..dictionary import (
    CrossCoder,
    BatchTopKCrossCoder,
    HeterogeneousCrossCoder,
    BatchTopKHeterogeneousCrossCoder,
)
from collections import namedtuple
from typing import Optional, Sequence
from ..trainers.trainer import get_lr_schedule


class CrossCoderTrainer(SAETrainer):
    """
    Trainer for CrossCoder sparse autoencoders.

    This trainer implements the CrossCoder training methodology. It supports multi-layer crosscoders with
    configurable sparsity penalties, learning rate scheduling, and optional neuron resampling to handle dead neurons.

    Args:
        dict_class: The CrossCoder class to use (default: CrossCoder)
        num_layers: Number of layers in the crosscoder (default: 2)
        activation_dim: Dimension of input activations (default: 512)
        dict_size: Size of the dictionary/number of features (default: 64 * 512)
        lr: Learning rate for optimization (default: 1e-3)
        l1_penalty: L1 sparsity penalty coefficient (default: 1e-1)
        warmup_steps: Number of warmup steps for learning rate schedule (default: 1000)
        resample_steps: How often to resample dead neurons (default: None, no resampling)
        seed: Random seed for reproducibility (default: None)
        device: Device to run on (default: None, auto-detect)
        layer: Layer index in the language model (required)
        lm_name: Name of the language model (required)
        wandb_name: Name for wandb logging (default: "CrossCoderTrainer")
        submodule_name: Specific submodule name within the layer (default: None)
        compile: Whether to compile the model with torch.compile (default: False)
        dict_class_kwargs: Additional arguments for the dictionary class (default: {})
        pretrained_ae: Pre-trained autoencoder to use instead of initializing new one (default: None)
        use_mse_loss: Whether to use MSE loss instead of L2 loss for reconstruction (default: False)
        activation_mean: Optional activation mean (default: None)
        activation_std: Optional activation std (default: None)
        target_rms: Target RMS for input activation normalization.
    """

    def __init__(
        self,
        dict_class=CrossCoder,
        num_layers=2,
        activation_dim=512,
        dict_size=64 * 512,
        lr=1e-3,
        l1_penalty=1e-1,
        warmup_steps=1000,  # lr warmup period at start of training and after each resample
        resample_steps=None,  # how often to resample neurons
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="CrossCoderTrainer",
        submodule_name=None,
        compile=False,
        dict_class_kwargs={},
        pretrained_ae=None,
        use_mse_loss=False,
        activation_mean: Optional[th.Tensor] = None,
        activation_std: Optional[th.Tensor] = None,
        target_rms: float = 1.0,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.compile = compile
        self.use_mse_loss = use_mse_loss
        self.target_rms = target_rms
        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

        # initialize dictionary
        if pretrained_ae is None:
            self.ae = dict_class(
                activation_dim,
                dict_size,
                num_layers=num_layers,
                activation_mean=activation_mean,
                activation_std=activation_std,
                target_rms=target_rms,
                **dict_class_kwargs,
            )
        else:
            self.ae = pretrained_ae

        if compile:
            self.ae = th.compile(self.ae)

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.wandb_name = wandb_name

        if device is None:
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.ae.to(self.device)

        self.resample_steps = resample_steps

        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = th.zeros(self.ae.dict_size, dtype=int).to(
                self.device
            )
        else:
            self.steps_since_active = None

        self.optimizer = th.optim.Adam(self.ae.parameters(), lr=lr)
        if resample_steps is None:

            def warmup_fn(step):
                return min(step / warmup_steps, 1.0)

        else:

            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.0)

        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_fn
        )

    def resample_neurons(self, deads, activations):
        """
        Resample dead neurons by reinitializing their weights and resetting optimizer state.

        Args:
            deads: Boolean tensor indicating which neurons are dead
            activations: Input activations used for resampling
        """
        with th.no_grad():
            if deads.sum() == 0:
                return
            self.ae.resample_neurons(deads, activations)
            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()["state"]
            ## encoder weight
            state_dict[0]["exp_avg"][:, :, deads] = 0.0
            state_dict[0]["exp_avg_sq"][:, :, deads] = 0.0
            ## encoder bias
            state_dict[1]["exp_avg"][deads] = 0.0
            state_dict[1]["exp_avg_sq"][deads] = 0.0
            ## decoder weight
            state_dict[3]["exp_avg"][:, deads, :] = 0.0
            state_dict[3]["exp_avg_sq"][:, deads, :] = 0.0

    def loss(
        self,
        x,
        logging=False,
        return_deads=False,
        normalize_activations=True,
        inplace_normalize=True,
        **kwargs,
    ):
        """
        Compute the training loss including reconstruction and sparsity terms.

        Args:
            x: Input activations (batch_size, num_layers, activation_dim)
            logging: Whether to return detailed logging information
            return_deads: Whether to return dead neuron information
            normalize_activations: Whether to normalize activations before processing
            inplace_normalize: Whether to normalize activations in-place
            **kwargs: Additional keyword arguments

        Returns:
            If logging=False: Total loss tensor
            If logging=True: Named tuple with detailed loss breakdown and intermediate values
        """
        x = (
            self.ae.normalize_activations(x, inplace=inplace_normalize)
            if normalize_activations
            else x
        )
        x_hat, f = self.ae(x, output_features=True, normalize_activations=False)
        l2_loss = th.linalg.norm(x - x_hat, dim=-1).mean()
        mse_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        if self.use_mse_loss:
            recon_loss = mse_loss
        else:
            recon_loss = l2_loss
        l1_loss = f.norm(p=1, dim=-1).mean()
        deads = (f <= 1e-4).all(dim=0)
        if self.steps_since_active is not None:
            # update steps_since_active
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0

        loss = recon_loss + self.l1_penalty * l1_loss

        if not logging:
            return loss
        else:
            log_dict = {
                "l2_loss": l2_loss.item(),
                "mse_loss": mse_loss.item(),
                "sparsity_loss": l1_loss.item(),
                "loss": loss.item(),
                "deads": deads if return_deads else None,
            }
            for layer in range(x.shape[1]):
                log_dict[f"rms_norm_l{layer}"] = th.sqrt(
                    (x[:, layer, :].pow(2).sum(-1)).mean()
                ).item()
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                log_dict,
            )

    def update(self, step, activations):
        """
        Perform a single training step.

        Args:
            step: Current training step number
            activations: Input activations for this step (batch_size, num_layers, activation_dim)
        """
        activations = activations.to(self.device)
        activations = self.ae.normalize_activations(
            activations,
            inplace=True,  # Normalize inplace to avoid copying the activations during training
        )
        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step, normalize_activations=False)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(
                self.steps_since_active > self.resample_steps / 2, activations
            )

    @property
    def config(self):
        """
        Get the trainer configuration as a dictionary.

        Returns:
            Dictionary containing all trainer configuration parameters
        """
        return {
            "dict_class": (
                self.ae.__class__.__name__
                if not self.compile
                else self.ae._orig_mod.__class__.__name__
            ),
            "trainer_class": self.__class__.__name__,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "lr": self.lr,
            "l1_penalty": self.l1_penalty,
            "warmup_steps": self.warmup_steps,
            "resample_steps": self.resample_steps,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "use_mse_loss": self.use_mse_loss,
            "code_normalization": str(self.ae.code_normalization),
            "code_normalization_alpha_sae": self.ae.code_normalization_alpha_sae,
            "code_normalization_alpha_cc": self.ae.code_normalization_alpha_cc,
            "target_rms": self.target_rms,
        }


class BatchTopKCrossCoderTrainer(SAETrainer):
    """
    Trainer for BatchTopK CrossCoder sparse autoencoders.

    This trainer implements a BatchTopK sparsity constraint for CrossCoders, with adaptive thresholding, auxiliary loss for dead feature
    resurrection, and k-annealing capabilities.

    Args:
        steps: Total number of training steps
        activation_dim: Dimension of input activations
        dict_size: Size of the dictionary/number of features
        k: Target number of active features per example
        layer: Layer index in the language model
        lm_name: Name of the language model
        num_layers: Number of layers in the crosscoder (default: 2)
        k_max: Initial k value for annealing (default: None, uses k)
        k_annealing_steps: Number of steps to anneal k from k_max to k (default: 0)
        dict_class: The crosscoder class to use (default: BatchTopKCrossCoder)
        lr: Learning rate, auto-computed if None (default: None)
        auxk_alpha: Weight for auxiliary loss (default: 1/32)
        warmup_steps: Number of warmup steps (default: 1000)
        decay_start: When learning rate decay starts (default: None)
        threshold_beta: Beta parameter for threshold EMA (default: 0.999)
        threshold_start_step: When to start threshold updates (default: 1000)
        seed: Random seed for reproducibility (default: None)
        device: Device to run on (default: None, auto-detect)
        wandb_name: Name for wandb logging (default: "BatchTopKSAE")
        submodule_name: Specific submodule name within the layer (default: None)
        pretrained_ae: Pre-trained autoencoder (default: None)
        dict_class_kwargs: Additional arguments for the dictionary class (default: {})
        activation_mean: Optional activation mean (default: None)
        activation_std: Optional activation std (default: None)
        target_rms: Target RMS for input activation normalization.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,  # Target k value
        layer: int,
        lm_name: str,
        num_layers: int = 2,
        k_max: Optional[int] = None,  # Initial k value for annealing (defaults to k)
        k_annealing_steps: int = 0,  # Steps to anneal k from k_max to k
        dict_class: type = BatchTopKCrossCoder,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "BatchTopKSAE",
        submodule_name: Optional[str] = None,
        pretrained_ae: Optional[BatchTopKCrossCoder] = None,
        dict_class_kwargs: dict = {},
        activation_mean: Optional[th.Tensor] = None,
        activation_std: Optional[th.Tensor] = None,
        target_rms: float = 1.0,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps

        # Store k annealing parameters
        self.k_target = k
        self.k_initial = k_max if k_max is not None else k
        self.k_annealing_total_steps = k_annealing_steps

        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.target_rms = target_rms

        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

        # initialize dictionary
        if pretrained_ae is None:
            self.ae = dict_class(
                activation_dim,
                dict_size,
                num_layers,
                self.k_initial,
                activation_mean=activation_mean,
                activation_std=activation_std,
                target_rms=target_rms,
                **dict_class_kwargs,
            )
        else:
            self.ae = pretrained_ae

        if device is None:
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = th.zeros(dict_size, dtype=th.long, device=device)
        self.logging_parameters = [
            "effective_l0",
            "running_deads",
            "pre_norm_auxk_loss",
            "k_current_value",
        ]
        self.dict_class_kwargs = dict_class_kwargs
        self.effective_l0 = -1
        self.running_deads = -1
        self.pre_norm_auxk_loss = -1
        self.k_current_value = self.k_initial

        self.optimizer = th.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def get_auxiliary_loss(
        self,
        residual_BD: th.Tensor,
        post_relu_f: th.Tensor,
        post_relu_f_scaled: th.Tensor,
    ):
        """
        Compute auxiliary loss to resurrect dead features.

        This implements an auxk loss similar to TopK and BatchTopKSAE that attempts
        to make dead latents active again by computing reconstruction loss on the
        top-k dead features.

        Args:
            residual_BD: Residual tensor (batch_size, num_layers, model_dim)
            post_relu_f: Post-ReLU feature activations
            post_relu_f_scaled: Scaled post-ReLU feature activations

        Returns:
            Normalized auxiliary loss tensor
        """
        batch_size, num_layers, model_dim = residual_BD.size()
        # reshape to (batch_size, num_layers*model_dim)
        residual_BD = residual_BD.reshape(batch_size, -1)
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.running_deads = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents_scaled = th.where(
                dead_features[None], post_relu_f_scaled, -th.inf
            ).detach()

            # Top-k dead latents
            _, auxk_indices = auxk_latents_scaled.topk(k_aux, sorted=False)
            auxk_buffer_BF = th.zeros_like(post_relu_f)
            row_indices = (
                th.arange(post_relu_f.size(0), device=post_relu_f.device)
                .view(-1, 1)
                .expand(-1, auxk_indices.size(1))
            )
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=post_relu_f[row_indices, auxk_indices]
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF, add_bias=False)
            x_reconstruct_aux = x_reconstruct_aux.reshape(batch_size, -1)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return th.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f_scaled: th.Tensor):
        """
        Update the activation threshold using exponential moving average.

        Args:
            f_scaled: Scaled feature activations
        """
        if self.ae.decoupled_code:
            return self.update_decoupled_threshold(f_scaled)
        active = f_scaled[f_scaled > 0]

        if active.size(0) == 0:
            min_activation = 0.0
        else:
            min_activation = active.min().detach().to(dtype=th.float32)

        if self.ae.threshold < 0:
            self.ae.threshold = min_activation
        else:
            self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                (1 - self.threshold_beta) * min_activation
            )

    def update_decoupled_threshold(self, f_scaled: th.Tensor):
        """
        Update thresholds for decoupled code normalization (per-layer thresholds).

        Args:
            f_scaled: Scaled feature activations
        """
        min_activation_f = (
            f_scaled.clone().transpose(0, 1).reshape(self.ae.num_layers, -1)
        )
        min_activation_f[min_activation_f <= 0] = th.inf
        min_activations = min_activation_f.min(dim=-1).values
        min_activations[min_activations == th.inf] = 0.0
        min_activations = min_activations.detach().to(dtype=th.float32)
        for layer, threshold in enumerate(self.ae.threshold):
            if threshold < 0:
                self.ae.threshold[layer] = min_activations[layer]
            else:
                self.ae.threshold[layer] = (
                    self.threshold_beta * self.ae.threshold[layer]
                ) + ((1 - self.threshold_beta) * min_activations[layer])

    def loss(
        self,
        x,
        step=None,
        logging=False,
        use_threshold=False,
        normalize_activations=True,
        inplace_normalize=True,
        **kwargs,
    ):
        """
        Compute the training loss with reconstruction and auxiliary terms.

        Args:
            x: Input activations (batch_size, num_layers, activation_dim)
            step: Current training step (used for k-annealing)
            logging: Whether to return detailed logging information
            use_threshold: Whether to use thresholding during encoding
            normalize_activations: Whether to normalize activations before processing
            inplace_normalize: Whether to normalize activations in-place
            **kwargs: Additional keyword arguments

        Returns:
            If logging=False: Total loss tensor
            If logging=True: Named tuple with detailed loss breakdown and intermediate values
        """
        x = (
            self.ae.normalize_activations(x, inplace=inplace_normalize)
            if normalize_activations
            else x
        )
        if step is not None:
            # Update k for annealing if applicable
            if self.k_annealing_total_steps > 0 and self.k_initial != self.k_target:
                if step < self.k_annealing_total_steps:
                    progress = float(step) / self.k_annealing_total_steps
                    # Linear interpolation from k_initial down to k_target
                    current_k_float = (
                        self.k_initial - (self.k_initial - self.k_target) * progress
                    )
                    new_k_val = max(1, int(round(current_k_float)))
                    if self.ae.k.item() != new_k_val:
                        self.ae.k.fill_(new_k_val)
                else:  # Annealing finished, ensure k is set to k_target
                    if self.ae.k.item() != self.k_target:
                        self.ae.k.fill_(self.k_target)
            elif (
                self.k_annealing_total_steps == 0 and self.ae.k.item() != self.k_initial
            ):
                # If no annealing steps, k should be fixed at k_initial
                self.ae.k.fill_(self.k_initial)

        self.k_current_value = self.ae.k.item()

        f, f_scaled, active_indices_F, post_relu_f, post_relu_f_scaled = self.ae.encode(
            x,
            return_active=True,
            use_threshold=use_threshold,
            normalize_activations=False,
        )  # (batch_size, dict_size)
        # l0 = (f != 0).float().sum(dim=-1).mean().item()

        if step > self.threshold_start_step and not logging:
            self.update_threshold(f_scaled)

        x_hat = self.ae.decode(f, denormalize_activations=False)

        e = x - x_hat
        assert e.shape == x.shape

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = th.zeros_like(self.num_tokens_since_fired, dtype=th.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        mse_loss = e.pow(2).sum(dim=-1).mean()
        l2_loss = th.linalg.norm(e, dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_f, post_relu_f_scaled)
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "mse_loss": mse_loss.item(),
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "deads": ~did_fire,
                    "threshold": self.ae.threshold.tolist(),
                    "sparsity_weight": self.ae.get_code_normalization().mean().item(),
                },
            )

    def update(self, step, x):
        """
        Perform a single training step.

        Args:
            step: Current training step number
            x: Input activations for this step (batch_size, num_layers, activation_dim)

        Returns:
            Loss value as float
        """
        x = x.to(self.device)
        x = self.ae.normalize_activations(
            x, inplace=True
        )  # Normalize inplace to avoid copying the activations during training
        if step == 0:
            median = self.geometric_median(x)
            median = median.to(self.device)
            self.ae.decoder.bias.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step, normalize_activations=False)
        loss.backward()

        th.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        return loss.item()

    @property
    def config(self):
        """
        Get the trainer configuration as a dictionary.

        Returns:
            Dictionary containing all trainer configuration parameters
        """
        return {
            "trainer_class": "BatchTopKCrossCoderTrainer",
            "dict_class": "BatchTopKCrossCoder",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "k_target": self.k_target,
            "k_initial": self.k_initial,
            "k_annealing_steps": self.k_annealing_total_steps,
            "code_normalization": str(self.ae.code_normalization),
            "code_normalization_alpha_sae": self.ae.code_normalization_alpha_sae,
            "code_normalization_alpha_cc": self.ae.code_normalization_alpha_cc,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "dict_class_kwargs": {k: str(v) for k, v in self.dict_class_kwargs.items()},
            "target_rms": self.target_rms,
        }

    @staticmethod
    def geometric_median(points: th.Tensor, max_iter: int = 100, tol: float = 1e-5):
        """
        Compute the geometric median of a set of points.

        The geometric median is the point that minimizes the sum of distances to all points.
        This is used to initialize the decoder bias to a robust estimate of the data center.

        Args:
            points: Tensor of shape (num_points, num_layers, model_dim)
            max_iter: Maximum number of iterations for the iterative algorithm
            tol: Tolerance for convergence

        Returns:
            Geometric median tensor of shape (num_layers, model_dim)
        """
        # points.shape = (num_points, num_layers, model_dim)
        guess = points.mean(dim=0)
        prev = th.zeros_like(guess)
        weights = th.ones((len(points), points.shape[1]), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / th.norm(points - guess, dim=-1)  # (num_points, num_layers)
            weights /= weights.sum(dim=0, keepdim=True)  # (num_points, num_layers)
            guess = (weights.unsqueeze(-1) * points).sum(
                dim=0
            )  # (num_layers, model_dim)
            if th.all(th.norm(guess - prev, dim=-1) < tol):
                break

        return guess


def _validate_activation_tuple(
    x: Sequence[th.Tensor], activation_dims: Sequence[int]
) -> tuple[th.Tensor, ...]:
    if not isinstance(x, (tuple, list)):
        raise TypeError("heterogeneous CrossCoders require a tuple/list of tensors")
    if len(x) != len(activation_dims):
        raise ValueError("activation tuple length does not match activation_dims")
    values = tuple(x)
    batch_size = values[0].shape[0]
    for source_idx, (value, dim) in enumerate(zip(values, activation_dims)):
        if value.ndim != 2 or value.shape != (batch_size, dim):
            raise ValueError(
                f"source {source_idx} must have shape {(batch_size, dim)}, "
                f"got {tuple(value.shape)}"
            )
    return values


class HeterogeneousCrossCoderTrainer(SAETrainer):
    """L1 CrossCoder trainer with a separate hidden width per model slot."""

    is_crosscoder_trainer = True

    def __init__(
        self,
        activation_dims: Sequence[int],
        dict_size: int,
        lr: float = 1e-4,
        l1_penalty: float = 2.0,
        warmup_steps: int = 1000,
        resample_steps: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        source_names: Sequence[str] | None = None,
        source_layers: Sequence[int] | None = None,
        wandb_name: str = "HeterogeneousCrossCoder",
        dict_class=HeterogeneousCrossCoder,
        dict_class_kwargs: dict | None = None,
        pretrained_ae: HeterogeneousCrossCoder | None = None,
        use_mse_loss: bool = True,
        activation_mean: Sequence[th.Tensor] | None = None,
        activation_std: Sequence[th.Tensor] | None = None,
        target_rms: float = 1.0,
    ):
        super().__init__(seed)
        self.activation_dims = tuple(int(dim) for dim in activation_dims)
        self.source_names = tuple(
            source_names
            if source_names is not None
            else (f"source_{idx}" for idx in range(len(self.activation_dims)))
        )
        self.source_layers = tuple(
            int(layer)
            for layer in (
                source_layers
                if source_layers is not None
                else (0 for _ in self.activation_dims)
            )
        )
        if len(self.source_names) != len(self.activation_dims) or len(
            self.source_layers
        ) != len(self.activation_dims):
            raise ValueError("source metadata must match activation_dims")
        self.device = device or ("cuda" if th.cuda.is_available() else "cpu")
        self.lr = float(lr)
        self.l1_penalty = float(l1_penalty)
        self.warmup_steps = int(warmup_steps)
        self.resample_steps = resample_steps
        self.wandb_name = wandb_name
        self.use_mse_loss = use_mse_loss
        self.target_rms = float(target_rms)
        dict_class_kwargs = dict_class_kwargs or {}
        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
        self.ae = pretrained_ae or dict_class(
            self.activation_dims,
            dict_size,
            activation_mean=activation_mean,
            activation_std=activation_std,
            target_rms=target_rms,
            **dict_class_kwargs,
        )
        self.ae.to(self.device)
        self.optimizer = th.optim.Adam(self.ae.parameters(), lr=self.lr)

        def warmup_fn(step: int) -> float:
            if self.warmup_steps <= 0:
                return 1.0
            if self.resample_steps is None:
                return min(step / self.warmup_steps, 1.0)
            return min((step % self.resample_steps) / self.warmup_steps, 1.0)

        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_fn
        )
        self.steps_since_active = (
            th.zeros(dict_size, dtype=th.long, device=self.device)
            if resample_steps is not None
            else None
        )

    def loss(
        self,
        x: Sequence[th.Tensor],
        logging: bool = False,
        return_deads: bool = False,
        normalize_activations: bool = True,
        **kwargs,
    ):
        del kwargs
        x = _validate_activation_tuple(x, self.activation_dims)
        normalized = self.ae.normalize_activations(x) if normalize_activations else x
        x_hat, f = self.ae(
            normalized, output_features=True, normalize_activations=False
        )
        residuals = tuple(
            value - reconstruction
            for value, reconstruction in zip(normalized, x_hat)
        )
        l2_per_source = th.stack(
            [residual.norm(dim=-1).mean() for residual in residuals]
        )
        mse_per_source = th.stack(
            [residual.square().sum(dim=-1).mean() for residual in residuals]
        )
        recon_loss = (
            mse_per_source.mean() if self.use_mse_loss else l2_per_source.mean()
        )
        l1_loss = f.norm(p=1, dim=-1).mean()
        deads = (f <= 1e-4).all(dim=0)
        if self.steps_since_active is not None:
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        loss = recon_loss + self.l1_penalty * l1_loss
        if not logging:
            return loss
        losses = {
            "l2_loss": l2_per_source.mean().item(),
            "mse_loss": mse_per_source.mean().item(),
            "sparsity_loss": l1_loss.item(),
            "loss": loss.item(),
            "deads": deads if return_deads else None,
        }
        for source_idx, source_name in enumerate(self.source_names):
            losses[f"l2_{source_name}"] = l2_per_source[source_idx].item()
            losses[f"mse_{source_name}"] = mse_per_source[source_idx].item()
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            normalized, x_hat, f, losses
        )

    def update(self, step: int, activations: Sequence[th.Tensor]):
        activations = tuple(value.to(self.device) for value in activations)
        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if (
            self.resample_steps is not None
            and step > 0
            and step % self.resample_steps == 0
        ):
            self.ae.resample_neurons(
                self.steps_since_active > self.resample_steps // 2,
                self.ae.normalize_activations(activations),
            )
        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": type(self).__name__,
            "dict_class": type(self.ae).__name__,
            "activation_dims": list(self.activation_dims),
            "dict_size": self.ae.dict_size,
            "source_names": list(self.source_names),
            "source_layers": list(self.source_layers),
            "lr": self.lr,
            "l1_penalty": self.l1_penalty,
            "warmup_steps": self.warmup_steps,
            "resample_steps": self.resample_steps,
            "device": self.device,
            "wandb_name": self.wandb_name,
            "use_mse_loss": self.use_mse_loss,
            "code_normalization": str(self.ae.code_normalization),
            "target_rms": self.target_rms,
        }


class BatchTopKHeterogeneousCrossCoderTrainer(SAETrainer):
    """BatchTopK trainer for cross-model activation tuples with unequal widths."""

    is_crosscoder_trainer = True
    is_batch_topk_trainer = True

    def __init__(
        self,
        steps: int,
        activation_dims: Sequence[int],
        dict_size: int,
        k: int = 50,
        lr: float | None = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: int | None = None,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: int | None = None,
        device: str | None = None,
        source_names: Sequence[str] | None = None,
        source_layers: Sequence[int] | None = None,
        wandb_name: str = "BatchTopKHeterogeneousCrossCoder",
        dict_class=BatchTopKHeterogeneousCrossCoder,
        dict_class_kwargs: dict | None = None,
        pretrained_ae: BatchTopKHeterogeneousCrossCoder | None = None,
        activation_mean: Sequence[th.Tensor] | None = None,
        activation_std: Sequence[th.Tensor] | None = None,
        target_rms: float = 1.0,
    ):
        super().__init__(seed)
        self.steps = int(steps)
        self.activation_dims = tuple(int(dim) for dim in activation_dims)
        self.source_names = tuple(
            source_names
            if source_names is not None
            else (f"source_{idx}" for idx in range(len(self.activation_dims)))
        )
        self.source_layers = tuple(
            int(layer)
            for layer in (
                source_layers
                if source_layers is not None
                else (0 for _ in self.activation_dims)
            )
        )
        self.device = device or ("cuda" if th.cuda.is_available() else "cpu")
        self.wandb_name = wandb_name
        self.auxk_alpha = float(auxk_alpha)
        self.warmup_steps = int(warmup_steps)
        self.decay_start = decay_start
        self.threshold_beta = float(threshold_beta)
        self.threshold_start_step = int(threshold_start_step)
        self.target_rms = float(target_rms)
        dict_class_kwargs = dict_class_kwargs or {}
        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
        self.ae = pretrained_ae or dict_class(
            self.activation_dims,
            dict_size,
            k=k,
            activation_mean=activation_mean,
            activation_std=activation_std,
            target_rms=target_rms,
            **dict_class_kwargs,
        )
        self.ae.to(self.device)
        scale = dict_size / (2**14)
        self.lr = float(lr if lr is not None else 2e-4 / scale**0.5)
        self.optimizer = th.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=get_lr_schedule(
                self.steps, self.warmup_steps, decay_start=self.decay_start
            ),
        )
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = max(self.activation_dims) // 2
        self.num_tokens_since_fired = th.zeros(
            dict_size, dtype=th.long, device=self.device
        )
        self.effective_l0 = -1
        self.running_deads = -1
        self.pre_norm_auxk_loss = -1
        self.logging_parameters = [
            "effective_l0",
            "running_deads",
            "pre_norm_auxk_loss",
        ]

    @staticmethod
    def geometric_median(
        points: th.Tensor, max_iter: int = 100, tol: float = 1e-5
    ) -> th.Tensor:
        guess = points.mean(dim=0)
        for _ in range(max_iter):
            distances = th.norm(points - guess, dim=-1).clamp_min(1e-8)
            weights = distances.reciprocal()
            weights = weights / weights.sum()
            updated = (weights.unsqueeze(-1) * points).sum(dim=0)
            if th.norm(updated - guess) < tol:
                guess = updated
                break
            guess = updated
        return guess

    def get_auxiliary_loss(
        self,
        residuals: Sequence[th.Tensor],
        post_relu_f: th.Tensor,
        post_relu_f_scaled: th.Tensor,
    ) -> th.Tensor:
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.running_deads = int(dead_features.sum().item())
        if not dead_features.any():
            self.pre_norm_auxk_loss = -1
            return th.zeros((), device=post_relu_f.device, dtype=post_relu_f.dtype)
        k_aux = min(self.top_k_aux, int(dead_features.sum().item()))
        candidates = th.where(
            dead_features.unsqueeze(0), post_relu_f_scaled, -th.inf
        ).detach()
        aux_indices = candidates.topk(k_aux, dim=-1, sorted=False).indices
        row_indices = th.arange(
            post_relu_f.shape[0], device=post_relu_f.device
        ).unsqueeze(1)
        aux_f = th.zeros_like(post_relu_f)
        aux_f.scatter_(
            1, aux_indices, post_relu_f[row_indices, aux_indices]
        )
        reconstructions = self.ae.decode(
            aux_f, denormalize_activations=False, add_bias=False
        )
        source_losses = []
        for residual, reconstruction in zip(residuals, reconstructions):
            numerator = (residual.float() - reconstruction.float()).square().sum(
                dim=-1
            ).mean()
            centered = residual.float() - residual.float().mean(dim=0, keepdim=True)
            denominator = centered.square().sum(dim=-1).mean().clamp_min(1e-8)
            source_losses.append(numerator / denominator)
        aux_loss = th.stack(source_losses).mean().nan_to_num(0.0)
        self.pre_norm_auxk_loss = aux_loss.item()
        return aux_loss

    def update_threshold(self, f_scaled: th.Tensor):
        active = f_scaled[f_scaled > 0]
        minimum = (
            active.min().detach().float()
            if active.numel()
            else th.zeros((), device=f_scaled.device)
        )
        if self.ae.threshold < 0:
            self.ae.threshold.copy_(minimum)
        else:
            self.ae.threshold.mul_(self.threshold_beta).add_(
                minimum * (1 - self.threshold_beta)
            )

    def loss(
        self,
        x: Sequence[th.Tensor],
        step: int | None = None,
        logging: bool = False,
        use_threshold: bool = False,
        normalize_activations: bool = True,
        **kwargs,
    ):
        del kwargs
        x = _validate_activation_tuple(x, self.activation_dims)
        normalized = self.ae.normalize_activations(x) if normalize_activations else x
        f, f_scaled, active, post_relu_f, post_relu_scaled = self.ae.encode(
            normalized,
            return_active=True,
            use_threshold=use_threshold,
            normalize_activations=False,
        )
        if (
            step is not None
            and step > self.threshold_start_step
            and not logging
            and not use_threshold
        ):
            self.update_threshold(f_scaled)
        x_hat = self.ae.decode(f, denormalize_activations=False)
        residuals = tuple(
            value - reconstruction
            for value, reconstruction in zip(normalized, x_hat)
        )
        l2_per_source = th.stack(
            [residual.norm(dim=-1).mean() for residual in residuals]
        )
        mse_per_source = th.stack(
            [residual.square().sum(dim=-1).mean() for residual in residuals]
        )
        num_tokens = normalized[0].shape[0]
        self.num_tokens_since_fired += num_tokens
        self.num_tokens_since_fired[active] = 0
        self.effective_l0 = int(self.ae.k.item())
        auxk_loss = self.get_auxiliary_loss(
            tuple(residual.detach() for residual in residuals),
            post_relu_f,
            post_relu_scaled,
        )
        loss = l2_per_source.mean() + self.auxk_alpha * auxk_loss
        if not logging:
            return loss
        losses = {
            "l2_loss": l2_per_source.mean().item(),
            "mse_loss": mse_per_source.mean().item(),
            "auxk_loss": auxk_loss.item(),
            "loss": loss.item(),
            "deads": ~active,
            "threshold": self.ae.threshold.item(),
        }
        for source_idx, source_name in enumerate(self.source_names):
            losses[f"l2_{source_name}"] = l2_per_source[source_idx].item()
            losses[f"mse_{source_name}"] = mse_per_source[source_idx].item()
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            normalized, x_hat, f, losses
        )

    def update(self, step: int, activations: Sequence[th.Tensor]):
        activations = tuple(value.to(self.device) for value in activations)
        normalized = self.ae.normalize_activations(activations)
        if step == 0:
            for source_idx, value in enumerate(normalized):
                self.ae.decoder.projections[source_idx].bias.data.copy_(
                    self.geometric_median(value)
                )
        self.optimizer.zero_grad()
        loss = self.loss(
            normalized, step=step, normalize_activations=False, use_threshold=False
        )
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": type(self).__name__,
            "dict_class": type(self.ae).__name__,
            "activation_dims": list(self.activation_dims),
            "dict_size": self.ae.dict_size,
            "source_names": list(self.source_names),
            "source_layers": list(self.source_layers),
            "lr": self.lr,
            "steps": self.steps,
            "k": int(self.ae.k.item()),
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "device": self.device,
            "wandb_name": self.wandb_name,
            "code_normalization": str(self.ae.code_normalization),
            "target_rms": self.target_rms,
        }
