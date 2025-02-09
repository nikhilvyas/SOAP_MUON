"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import contextlib
import logging
from copy import deepcopy
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import numpy as np
import math

from distributed_soap.shampoo_types import (
    ACCUM_STEPS,
    BETA3,
    BETAS,
    DDPShampooConfig,
    DistributedConfig,
    DISTRIBUTOR,
    EPSILON,
    FILTERED_GRAD,
    FILTERED_GRAD_LIST,
    FSDPShampooConfig,
    FullyShardShampooConfig,
    HSDPShampooConfig,
    LR,
    MASKED_BLOCKED_GRADS,
    MASKED_BLOCKED_PARAMS,
    MASKED_FILTERED_GRAD_LIST,
    MAX_PRECONDITIONER_DIM,
    MAX_PRECOND_DIM_DIAG,
    PARAMS,
    PRECISION_CONFIG,
    PrecisionConfig,
    PRECONDITION_FREQUENCY,
    PRECONDITIONER_DTYPE,
    PREVIOUS_GRAD_SELECTOR,
    SHAMPOO_PRECONDITIONER_LIST,
    ShampooPT2CompileConfig,
    STEP,
    USE_BIAS_CORRECTION,
    USE_MERGE_DIMS,
    DECOUPLED_WEIGHT_DECAY,
)

from distributed_soap.utils.shampoo_checkpoint_utils import (
    extract_state_dict_content,
    flatten,
    unflatten,
    update_param_state_dict_object,
)
from distributed_soap.utils.shampoo_ddp_distributor import DDPDistributor
from distributed_soap.utils.shampoo_distributor import Distributor
from distributed_soap.utils.shampoo_fsdp_distributor import FSDPDistributor
from distributed_soap.utils.shampoo_fully_shard_distributor import (
    FullyShardDistributor,
)
from distributed_soap.utils.shampoo_hsdp_distributor import HSDPDistributor

from distributed_soap.utils.shampoo_preconditioner_list import (
    DequantizePreconditionersContext,
    ShampooPreconditionerList,
)
from distributed_soap.utils.shampoo_quantization import (
    DequantizeQuantizedTensorListContext,
    QuantizedTensor,
    QuantizedTensorList,
)
from distributed_soap.utils.shampoo_utils import compress_list
from torch.optim.optimizer import ParamsT

logger: logging.Logger = logging.getLogger(__name__)


class DistributedSOAP(torch.optim.Optimizer):
    """Implements distributed Shampoo algorithm.

    Developers:
        Hao-Jun Michael Shi (Meta Platforms, Inc.)
        Tsung-Hsien Lee
        Anna Cai (Meta Platforms, Inc.)
        Shintaro Iwasaki (Meta Platforms, Inc.)
        Ke Sang (Meta Platforms, Inc.)
        Wang Zhou (Meta Platforms, Inc.)

    with contributions and support from:

    Ganesh Ajjanagadde (Meta), Rohan Anil (Google), Adnan Aziz (Meta), Pavan Balaji (Meta), Shuo Chang (Meta), Weiwei Chu (Meta),
    Assaf Eisenman (Meta), Will Feng (Meta), Zhuobo Feng (Meta), Jose Gallego-Posada (Mila / Meta Platforms, Inc.), Avirup Ghosh (Meta),
    Yizi Gu (Meta), Vineet Gupta (Google), Yuchen Hao (Meta), Brian Hirsh (Meta), Yusuo Hu (Meta), Yuxi Hu (Meta), Minhui Huang (Meta),
    Guna Lakshminarayanan (Meta), Michael Lazos (Meta), Zhijing Li (Meta), Ming Liang (Meta), Wanchao Liang (Meta), Ying Liu
    (Meta), Wenguang Mao (Meta), Dheevatsa Mudigere (NVIDIA), Maxim Naumov (Meta), Jongsoo Park (Meta), Mike Rabbat (Meta),
    Kaushik Rangadurai (Meta), Dennis van der Staay (Meta), Fei Tian (Meta), Sanjay Vishwakarma (Meta), Xunnan (Shawn) Xu (Meta),
    Jiyan Yang (Meta), Chunxing Yin (Meta), and Iris Zhang (Meta).

    Details in: https://arxiv.org/pdf/2309.06497.pdf.

    Partly based on the work in:
    - https://arxiv.org/pdf/1802.09568.pdf
    - https://arxiv.org/pdf/2002.09018.pdf

    ------------
    Requirements
    ------------

    1. PyTorch >= 2.0
    2. Python >= 3.10
    3. CUDA 11.3, 11.4, 12.2+

    In order to support checkpointing, one must use torch.distributed.checkpoint and pass the named parameters into state_dict.
    Note that the standard checkpointing solution by PyTorch is not supported!

    --------
    Features
    --------

    1. Inner optimizer: tofill

    2. Blocking for Large-Dimensional Tensors: In order to scale Shampoo to large-dimensional tensors, we block the tensor
        and apply Shampoo to each block. For simplicity, suppose we have a linear layer/matrix parameter, W is a m x n matrix:

                [[w_11 w_12 ... w_1n]
                [w_21 w_22 ... w_2n]
            W =           :
                [w_m1 w_m2 ... w_mn]]

        Given a max_preconditioner_dim b > 0, blocks W and applies Shampoo to each block, i.e., if b divides both m, n, then:

                [[W_11 W_12 ... W_1k]
                 [W_21 W_22 ... W_2k]
            W =           :
                 [W_l1 W_l2 ... W_lk]]

        where l = m / b, k = n / b, and apply Shampoo to W_ij which is a b x b matrix. This can be viewed as further blocking
        each block of the Shampoo block-diagonal preconditioner.

        Computational cost = O(b^3)
        Memory cost = 4mn (including root inverse preconditioners)

    3. Distributed Memory and Computation: We support different distributed training setups through the distributed_config option,
        which specifies a configuration specific to that setting.

        - None: Performs serial single-GPU training. Replicates all computation and optimizer states across all
            devices.

        - DDPShampooConfig: Supports multi-GPU distributed data-parallel training via torch.distributed. Assigns optimizer states
            and computation for each block in a greedy fashion to different workers. Leverages DTensor in order to distribute the
            per-block optimizer states from Shampoo. An AllGather communication is performed in order to synchronize the parameter
            updates to applied to all parameter blocks.

            Distributed Training Specific Fields:
                - communication_dtype: We can specify the communication dtype used for the AllGather communication in order to
                    reduce communication overhead per-iteration.
                - num_trainers_per_group: Specifies the number of GPUs used per distributed group. This enables us to only
                    distribute computation across a subset of GPUs, and replicate the same computation across different distributed
                    groups. This is useful for performance by trading off communication costs vs. computational costs.
                - communicate_params: We offer the option to communicate the parameter updates or the updated parameters. Enabling
                    this option specifically communicates the updated parameters. Note that using a lower-precision
                    communication_dtype is more amenable to the case where this option is disabled (i.e., we are communicating the
                    parameter updates).

            Requirements:
                - torch.distributed must be initialized in advance.
                - Only supports homogeneous hardware architectures.

        - FSDPShampooConfig: Supports multi-GPU fully-sharded data-parallel training via torch.distributed. This option uses
            additional metadata in order to reconstruct valid tensor blocks of the original parameter from the flattened parameter
            representation.

            Distributed Training Specific Fields:
                - param_to_metadata: One must create a dictionary containing the metadata for each parameter in the FSDP model. This
                    includes the shape of the original parameter as well as the start and end indices of the tensor shard with
                    respect to the unsharded flattened parameter.

            Requirements:
                - torch.distributed must be initialized in advance.
                - One must enable the option use_orig_params = True in FSDP.

    4. PyTorch 2.0 Compile Support: Shampoo supports PyTorch 2.0's compilation feature to speed up model training. This is enabled by
        setting up the shampoo_pt2_compile_config arg for Shampoo PyTorch 2.0 compilation.

        - If shampoo_pt2_compile_config = None, ignores compilation, and Shampoo will run in eager mode.
            Shampoo PT2 eager mode means the optimizer runs on plain python code, there is no graphs and lower level kernels used
            to speed up the optimizer stage; and typically you would expect lower QPS for model training as a result.
            For more details regarding PT2 compilation: https://pytorch.org/get-started/pytorch-2.0/

        - If shampoo_pt2_compile_config is set to ShampooPT2CompileConfig class, Shampoo will run in PT2 mode. Shampoo PT2 mode typically gives
            on par numerics and model quality, plus higher QPS. But due to differences in lower level kernel implementation, model quality on par
            is not always guaranteed. If you see model quality gap, please switch back to Shampoo PT2 eager mode by setting
            shampoo_pt2_compile_config = None.

        Shampoo PT2 compilation can also be customized for the backend and options via ShampooPT2CompileConfig.
            ShampooPT2CompileConfig
                - pytorch_compile_backend: PT2 backend to use. All available backends in pytorch 2.0 is available for Shampoo. Typical backends to use
                    include 'inductor', 'aot_eager'. For more details: https://pytorch.org/docs/stable/torch.compiler.html
                - enable_shampoo_pt2_dynamic_shape: if true, PT2 will compile Shampoo data/tensors with `dynamic shape` mode. Default is False and use
                    `static` mode. `dynamic shape` means the tensor shapes can change from run to run, and PT2 will generate kernels not specialized to
                    particular tensor shape. Recommended to use `static` mode here for Shampoo.
                    More about dynamic shape: https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate. (Default: 1e-2)
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            (Default: (0.9, 1.0))
        beta3 (float): Coefficient used for computing running average of gradient only for the current iteration.
            This can be used to replicate a version of NAdam if set appropriately. For example, if beta1 = 0.9, then applying
            beta1 interpolation a second time is equivalent to setting beta3 = 0.9 * 0.9 = 0.81.
            If set to -1.0, will set equal to beta1. (Default: -1.0)
        epsilon (float): Term added to the denominator to improve numerical stability. (Default: 1e-12)
        decoupled_weight_decay (float): AdamW-style decoupled weight decay. (Default: 0.)
        max_preconditioner_dim (int): Maximum preconditioner dimension. (Default: 1024)
        precondition_frequency (int): Frequency for computing root inverse preconditioner. (Default: 1)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        use_merge_dims (bool): Merge dimensions if possible while respecting max_preconditioner_dim. (Default: True)
        use_pytorch_compile (Optional[bool]): Use PyTorch 2.0 compiler feature to speed up training. Deprecating, please use
            shampoo_pt2_compile_config instead; when this field is None, the use of PyTorch 2.0 compiler is decided by
            shampoo_pt2_compile_config. (Default: None)
        shampoo_pt2_compile_config (Optional[ShampooPT2CompileConfig]): Configuration for Shampoo PT2 compilation. If None,
            ignores compilation, and Shampoo will run in eager mode. (Default: None)
        distributed_config (Optional[DistributedConfig]): Configuration for applying Shampoo
            to different distributed training frameworks, such as distributed-data parallel (DDP) training.
            Based on the configuration, determines which version of Shampoo to use. (Default: None)
        preconditioner_dtype (Optional[torch.dtype]): **DEPRECATING** Data type for preconditioner. (Default: None)
        precision_config (PrecisionConfig): Data types for optimizer states. (Default: all fields torch.float)
        track_root_inv_residuals (bool): Track errors and residuals of root inverse. For debugging purposes.
            (Default: False)
        use_pf_warmup (bool): Flag for using preconditioner warmup. (Default: False)
        step_delay (int): If using preconditioner warmup, number of steps before moving to higher preconditioning. (Default: 5)

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        beta3: float = -1.0,
        epsilon: float = 1e-12,
        decoupled_weight_decay: float = 0.0,
        max_preconditioner_dim: int = 1000000,
        max_precond_dim_diag: int = 10000,
        precondition_frequency: int = 1,
        use_bias_correction: bool = True,
        use_merge_dims: bool = False,
        use_pytorch_compile: Optional[bool] = None,
        shampoo_pt2_compile_config: Optional[ShampooPT2CompileConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        preconditioner_dtype: Optional[torch.dtype] = None,
        precision_config: Optional[PrecisionConfig] = None,
        track_root_inv_residuals: bool = False,
        use_pf_warmup: bool = False,
        step_delay: int = 5,
    ) -> None:
        # Hyperparameter checks.
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be >= 0.0.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}. Must be in [0.0, 1.0)."
            )
        if not 0.0 < betas[1] <= 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}. Must be in (0.0, 1.0]."
            )
        if beta3 == -1.0:
            beta3 = betas[0]
        elif not 0.0 <= beta3 < 1.0:
            raise ValueError(
                f"Invalid beta3 parameter: {beta3}. Must be in [0.0, 1.0)."
            )
        if not epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Must be > 0.0.")
        if not decoupled_weight_decay >= 0.0:
            raise ValueError(
                f"Invalid decoupled_weight_decay value: {decoupled_weight_decay}. Must be >= 0.0."
            )
        if not max_preconditioner_dim >= 1:
            raise ValueError(
                f"Invalid max preconditioner dimension: {max_preconditioner_dim}. Must be >= 1."
            )
        if not precondition_frequency >= 1:
            raise ValueError(
                f"Invalid precondition frequency: {precondition_frequency}. Must be >= 1."
            )
        if track_root_inv_residuals:
            logger.setLevel(logging.DEBUG)


        # Deprecation warning for use_pytorch_compile
        if use_pytorch_compile is not None:
            if use_pytorch_compile and shampoo_pt2_compile_config is None:
                shampoo_pt2_compile_config = ShampooPT2CompileConfig()
                logger.warning(
                    "use_pytorch_compile is deprecating. Please use shampoo_pt2_compile_config instead."
                )
            elif use_pytorch_compile and shampoo_pt2_compile_config is not None:
                raise ValueError(
                    "Both use_pytorch_compile and shampoo_pt2_compile_config are provided. Please use only shampoo_pt2_compile_config as use_pytorch_compile is deprecating."
                )
            elif not use_pytorch_compile and shampoo_pt2_compile_config is not None:
                raise ValueError(
                    "use_pytorch_compile=False conflicts with non-None shampoo_pt2_compile_config arg. Please use only shampoo_pt2_compile_config as use_pytorch_compile is deprecating."
                )

        # Provide error for system Pytorch compile availability
        if shampoo_pt2_compile_config is not None and not torch.cuda.is_available():
            raise ValueError(
                "Backend does NOT support Pytorch 2.0 compile. Switch to use_pytorch_compile in (False, None) and shampoo_pt2_compile_config=None."
            )

        # Deprecation warning for preconditioner_dtype
        if preconditioner_dtype is not None:
            if precision_config is None:
                precision_config = PrecisionConfig(
                    factor_matrix_dtype=preconditioner_dtype,
                    eig_factor_matrix_dtype=preconditioner_dtype,
                )
                logger.warning(
                    "preconditioner_dtype is deprecated. Please use precision_config instead."
                )
            else:
                raise ValueError(
                    "Both preconditioner_dtype and precision_config are provided. Please use only precision_config as preconditioner_dtype is deprecated."
                )

        # Create default precision config if it is not provided.
        if precision_config is None:
            precision_config = PrecisionConfig()

        super().__init__(
            params,
            {
                LR: lr,
                BETAS: betas,
                BETA3: beta3,
                EPSILON: epsilon,
                DECOUPLED_WEIGHT_DECAY: decoupled_weight_decay,
                MAX_PRECONDITIONER_DIM: max_preconditioner_dim,
                MAX_PRECOND_DIM_DIAG: max_precond_dim_diag,
                PRECONDITION_FREQUENCY: precondition_frequency,
                USE_BIAS_CORRECTION: use_bias_correction,
                USE_MERGE_DIMS: use_merge_dims,
                PRECONDITIONER_DTYPE: preconditioner_dtype,
                PRECISION_CONFIG: precision_config,
            },
        )

        # Initialize non-group-related fields.
        self._distributed_config = distributed_config
        self._track_root_inv_residuals = track_root_inv_residuals
        self._shampoo_pt2_compile_config: Optional[ShampooPT2CompileConfig] = (
            shampoo_pt2_compile_config
        )
        self._step_delay = step_delay
        self._use_pf_warmup = use_pf_warmup

        # Initialize dictionary containing lists of .
        self._per_group_state_lists: List[Dict[str, Any]] = [
            {} for _ in self.param_groups
        ]

        # Block parameters and instantiate optimizer states.
        self._instantiate_distributor()
        self._instantiate_shampoo_preconditioner_list()
        self._instantiate_steps()
        self._instantiate_filtered_grads()
        self._instantiate_device()
        self._instantiate_per_group_step()

    @torch.no_grad()
    def _instantiate_distributor(self) -> None:
        if self._distributed_config is None:
            distributor = Distributor
        elif type(self._distributed_config) is DDPShampooConfig:
            distributor = partial(
                DDPDistributor, distributed_config=self._distributed_config
            )
        elif type(self._distributed_config) is FSDPShampooConfig:
            distributor = partial(
                FSDPDistributor, distributed_config=self._distributed_config
            )
        elif type(self._distributed_config) is FullyShardShampooConfig:
            distributor = FullyShardDistributor
        elif type(self._distributed_config) is HSDPShampooConfig:
            distributor = partial(
                HSDPDistributor,
                distributed_config=self._distributed_config,
            )
        else:
            raise NotImplementedError(f"{self._distributed_config=} not supported!")

        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Instantiate distributors for each group.
            state_lists[DISTRIBUTOR] = distributor(group)

            # If the number of trainers is more than the number of blocks,
            # some workers might not get any parameters which cause wasting resources because
            # those trainers are working on nothing.
            assert state_lists[
                DISTRIBUTOR
            ].local_blocked_params, f"Some workers have no parameters to work on. This mostly happens when the value of num_trainers_per_group field in {self._distributed_config=} is more than the number of local blocked params on a single device. Please check the num_trainers_per_group setting and consider reducing it."

            # Compile blocked parameters and block-to-parameter metadata into group lists.
            state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
                DISTRIBUTOR
            ].local_blocked_params
            # First PREVIOUS_GRAD_SELECTOR is set to None.
            state_lists[PREVIOUS_GRAD_SELECTOR] = None

    @torch.no_grad()
    def _instantiate_shampoo_preconditioner_list(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            state_lists[SHAMPOO_PRECONDITIONER_LIST] = ShampooPreconditionerList(
                block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                state=self.state,
                block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                beta2=group[BETAS][1],
                max_precond_dim_diag=group[MAX_PRECOND_DIM_DIAG],
                epsilon=group[EPSILON],
                use_bias_correction=group[USE_BIAS_CORRECTION],
                factor_matrix_dtype=group[PRECISION_CONFIG].factor_matrix_dtype,
                eig_factor_matrix_dtype=group[PRECISION_CONFIG].eig_factor_matrix_dtype,
                computation_dtype=(
                    group[PRECISION_CONFIG].computation_dtype
                    if group[PRECONDITIONER_DTYPE] is None
                    else group[PRECONDITIONER_DTYPE]
                ),
            )


    @torch.no_grad()
    def _instantiate_steps(self) -> None:
        for state_lists in self._per_group_state_lists:
            assert (
                len(state_lists[DISTRIBUTOR].global_block_info_list) > 0
            ), "There is no params in your param_group. Please check the instantiation of DistributedShampoo "
            'with param_group containing no params. For example, DistributedShampoo(params=[{"params": []}])'
            # NOTE: We instantiate a single step tensor on CPU for each group in order
            #       to track the number of steps taken by all parameters within the group.
            #       Instantiating on CPU avoids GPU synchronization.
            state_lists[STEP] = torch.tensor(0, dtype=torch.int64, device="cpu")
            state_lists[ACCUM_STEPS] = torch.tensor(0, dtype=torch.int64, device="cpu")
            state_lists[PRECONDITION_FREQUENCY] = torch.tensor(0, dtype=torch.int64, device="cpu")

            # In order to ensure that the step counter is checkpointed correctly, we store it
            # as a tensor (which is replicated across all devices) under the first parameter's state.
            block_info = state_lists[DISTRIBUTOR].global_block_info_list[0]
            self.state[block_info.param][STEP] = state_lists[STEP]
            self.state[block_info.param][ACCUM_STEPS] = state_lists[ACCUM_STEPS]
            self.state[block_info.param][PRECONDITION_FREQUENCY] = state_lists[PRECONDITION_FREQUENCY]

    @torch.no_grad()
    def _instantiate_filtered_grads(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[BETAS][0] == 0.0:
                continue

            # Construct global filtered gradient list.
            global_filtered_grad_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].global_blocked_params,
                state_lists[DISTRIBUTOR].global_block_info_list,
                strict=True,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[
                    block_info.param
                ], f"{block_index=} not found in {self.state[block_info.param]=}. "
                "Please check the initialization of self.state[block_info.param][block_index] "
                "within PreconditionerList, and check the initialization of BlockInfo within "
                "Distributor for the correctness of block_index."
                block_state = self.state[block_info.param][block_index]

                block_state[FILTERED_GRAD] = QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        shape=block.size(),
                        dtype=group[PRECISION_CONFIG].filtered_grad_dtype,
                        device=block.device,
                    ),
                    block_info,
                )
                global_filtered_grad_list.append(block_state[FILTERED_GRAD])

            # We compress the momentum list to only the locally-owned parameter states.
            state_lists[FILTERED_GRAD_LIST] = QuantizedTensorList(
                compress_list(
                    global_filtered_grad_list,
                    state_lists[DISTRIBUTOR].distributor_selector,
                ),
                group[PRECISION_CONFIG].filtered_grad_dtype,
                group[PRECISION_CONFIG].computation_dtype,
            )
            # Here, we set masked filtered grad list to filtered grad list because we assume
            # all parameters are active.
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[FILTERED_GRAD_LIST]

    @torch.no_grad()
    def _instantiate_device(self) -> None:
        # NOTE: Assume all parameter groups consistently exist on the same rank.
        self._device = self._per_group_state_lists[0][MASKED_BLOCKED_PARAMS][0].device

    @torch.no_grad()
    def _instantiate_per_group_step(self) -> None:
        # Use PT2 to compile the step function for each parameter group.
        self._per_group_step: Callable[
            [
                Dict[str, Any],
                torch.Tensor,
                torch.Tensor,
                float,
                float,
                float,
                bool,
                bool,
                bool,
                bool,
            ],
            None,
        ] = (
            torch.compile(
                self._per_group_step_impl,
                backend=self._shampoo_pt2_compile_config.pytorch_compile_backend,
                dynamic=self._shampoo_pt2_compile_config.enable_shampoo_pt2_dynamic_shape,
            )
            if self._shampoo_pt2_compile_config is not None
            else self._per_group_step_impl
        )
        if self._shampoo_pt2_compile_config is not None:
            logger.info(
                f"DistributedShampoo optimizer initialization is using {self._shampoo_pt2_compile_config.pytorch_compile_backend} backend and enable_shampoo_pt2_dynamic_shape={self._shampoo_pt2_compile_config.enable_shampoo_pt2_dynamic_shape}"
            )

    @staticmethod
    @torch.no_grad()
    def _mask_state_lists(state_lists: Dict[str, Any], group: Dict[str, Any]) -> None:
        if (
            state_lists[DISTRIBUTOR].local_grad_selector
            == state_lists[PREVIOUS_GRAD_SELECTOR]
        ):
            return

        if state_lists[STEP].item() >= 1:
            logger.warn(
                "PT2 will recompile because the gradient selction of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model."
            )
        # Updates masked state lists if previous block selector disagrees with current selector.
        # State list compression is necessary in order to avoid handling gradients with None.
        state_lists[PREVIOUS_GRAD_SELECTOR] = state_lists[
            DISTRIBUTOR
        ].local_grad_selector
        state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
            DISTRIBUTOR
        ].local_masked_blocked_params
        state_lists[SHAMPOO_PRECONDITIONER_LIST].compress_preconditioner_list(
            local_grad_selector=state_lists[DISTRIBUTOR].local_grad_selector,
        )
        if group[BETAS][0] != 0.0:
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[
                MASKED_FILTERED_GRAD_LIST
            ].compress(
                state_lists[DISTRIBUTOR].local_grad_selector,
            )

    @torch.no_grad()
    @torch.compiler.disable
    def _compute_eigenvectors(
        self, state_lists: Dict[str, Any], compute_eigenvectors: bool
    ) -> None:
        if compute_eigenvectors:
            state_lists[SHAMPOO_PRECONDITIONER_LIST].compute_eigenvectors()

    @torch.no_grad()
    @torch.compiler.disable
    def _precondition(
        self,
        state_lists: Dict[str, Any],
        masked_filtered_grad_list: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        # Precondition gradients.
        # Assumes that the step state is consistent across all parameters.

        masked_blocked_search_directions = state_lists[
            SHAMPOO_PRECONDITIONER_LIST
        ].precondition(
            masked_grad_list=masked_filtered_grad_list,
        )


        return masked_blocked_search_directions
    
    @torch.no_grad()
    @torch.compiler.disable
    def _precondition1(
        self,
        state_lists: Dict[str, Any],
        masked_filtered_grad_list: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        # Precondition gradients.
        # Assumes that the step state is consistent across all parameters.

        masked_blocked_search_directions = state_lists[
            SHAMPOO_PRECONDITIONER_LIST
        ].precondition1(
            masked_grad_list=masked_filtered_grad_list,
        )


        return masked_blocked_search_directions

    @torch.no_grad()
    def _update_preconditioners(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
    ) -> None:
        # Update Shampoo.
        state_lists[SHAMPOO_PRECONDITIONER_LIST].update_preconditioners(
            masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
            step=step,
        )
        
    @torch.no_grad()
    def _update_adam_preconditioners(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
    ) -> None:
        # Update Shampoo.
        state_lists[SHAMPOO_PRECONDITIONER_LIST].update_adam_preconditioners(
            masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
            step=step,
        )

    @torch.no_grad()
    def _update_adam_and_momentum(
        self,
        state_lists: Dict[str, Any],
        beta1: float,
        beta3: float,
        use_bias_correction: bool,
        step: torch.Tensor,
    ) -> None:
        if beta1 != 0.0:
            with DequantizeQuantizedTensorListContext(
                quantized_tensor_list = state_lists[MASKED_FILTERED_GRAD_LIST]
            ):
                
                masked_filtered_grad_list, state_list_update = state_lists[SHAMPOO_PRECONDITIONER_LIST].update_adam_and_momentum(
                    masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
                    masked_filtered_grad_list=state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
                    beta1 = beta1,
                    beta3 = beta3,
                    use_bias_correction = use_bias_correction,
                    step=step,
                )
                # print(beta3)
                # print(beta1)
                # print(state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value[0])
                # print(state_lists[MASKED_BLOCKED_GRADS][0])
                # print(state_list_update[0])

                # M' <- B1*M' + (1-B1)G'
                torch._foreach_lerp_(
                    state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
                    state_list_update,
                    weight=1 - beta1,
                )
                
                if math.isclose(beta1, beta3):
                    masked_filtered_grad_list = state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value


                # Apply bias correction if necessary.
                if use_bias_correction:
                    bias_correction1 = 1.0 - beta3 * beta1 ** (step - 1)
                    masked_filtered_grad_list = torch._foreach_div(
                        masked_filtered_grad_list,
                        bias_correction1,
                    )
                #print(state_lists[MASKED_BLOCKED_GRADS][0] == state_list_update[0])
                
        else:
            # Update Shampoo.
            state_lists[SHAMPOO_PRECONDITIONER_LIST].update_adam_preconditioners(
                masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
                step=step,
            )
            masked_filtered_grad_list = state_lists[MASKED_BLOCKED_GRADS]
        
        return masked_filtered_grad_list

    @torch.no_grad()
    def _project(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
    ) -> None:
        with DequantizeQuantizedTensorListContext(
            quantized_tensor_list=state_lists[MASKED_FILTERED_GRAD_LIST]
        ):
            # Update Shampoo.
            state_list_update = state_lists[SHAMPOO_PRECONDITIONER_LIST].project(
                masked_filtered_grad_list=state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
                step=step,
            )
            #Check state list different before and after
            #print(state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value[0])
            for i in range(len(state_list_update)):
                state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value[i].copy_(state_list_update[i])
            #print(state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value[0])


    @torch.no_grad()
    def _project_back(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
    ) -> None:
        with DequantizeQuantizedTensorListContext(
            quantized_tensor_list=state_lists[MASKED_FILTERED_GRAD_LIST]
        ):
            # Update Shampoo.
            state_list_update = state_lists[SHAMPOO_PRECONDITIONER_LIST].project_back(
                masked_filtered_grad_list=state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
                step=step,
            )

            #print(state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value[0])
            #Check state list different before and after
            for i in range(len(state_list_update)):
                state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value[i].copy_(state_list_update[i])
            #print(state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value[0])
        

    @torch.no_grad()
    def _compute_filtered_grad_list(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
        beta1: float,
        beta3: float,
        use_bias_correction: bool,
    ) -> Tuple[torch.Tensor, ...]:
        if beta1 != 0.0:
            with DequantizeQuantizedTensorListContext(
                quantized_tensor_list=state_lists[MASKED_FILTERED_GRAD_LIST]
            ):
                # Computes filtered gradient or EMA of the gradients with respect to beta3 if beta3 != beta1.
                masked_filtered_grad_list = (
                    torch._foreach_lerp(
                        state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
                        state_lists[MASKED_BLOCKED_GRADS],
                        weight=1 - beta3,
                    )
                    if beta3 != beta1
                    else state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value
                )

                # Update EMA of the gradients (with respect to beta1).
                # M <- B1*M + (1-B1)G
                torch._foreach_lerp_(
                    state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
                    state_lists[MASKED_BLOCKED_GRADS],
                    weight=1 - beta1,
                )

                # Apply bias correction if necessary.
                if use_bias_correction:
                    bias_correction1 = 1.0 - beta3 * beta1 ** (step - 1)
                    masked_filtered_grad_list = torch._foreach_div(
                        masked_filtered_grad_list,
                        bias_correction1,
                    )
        else:
            masked_filtered_grad_list = state_lists[MASKED_BLOCKED_GRADS]

        return masked_filtered_grad_list

    @torch.no_grad()
    def _apply_decoupled_weight_decay(
        self,
        state_lists: Dict[str, Any],
        masked_blocked_search_directions: Tuple[torch.Tensor, ...],
        decoupled_weight_decay: float,
    ) -> None:
        # Apply decoupled weight decay.
        if decoupled_weight_decay != 0.0:
            torch._foreach_add_(
                masked_blocked_search_directions,
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=decoupled_weight_decay,
            )


    @torch.no_grad()
    def _per_group_step_impl(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        beta3: float,
        decoupled_weight_decay: float,
        compute_eigenvectors: bool,
        use_bias_correction: bool,
    ) -> None:

        with DequantizePreconditionersContext(
            preconditioner_list=state_lists[SHAMPOO_PRECONDITIONER_LIST]
        ), contextlib.nullcontext():
            
            # For SOAP we flip the order of the computation of the filtered gradients and the preconditioning.
            if step == 1:
                #   L <- L + G * G^T
                #   R <- R + G^T * G
                self._update_preconditioners(
                    state_lists,
                    step,
                )

                # Compute matrix root inverse.
                #   L_inv <- L ** (-1/4)
                #   R_inv <- R ** (-1/4)
                #   (and similar)

                # update QL and QR
                if compute_eigenvectors:
                    self._project_back(state_lists, step)
                    self._compute_eigenvectors(state_lists, compute_eigenvectors)
                    self._project(state_lists, step)
                #self._compute_eigenvectors(state_lists, compute_eigenvectors)
                
                return
            # V
            # M'
            masked_filtered_grad_list = self._update_adam_and_momentum(
                state_lists,
                beta1,
                beta3,
                use_bias_correction,
                step,
            )
            
            # Precondition filtered gradients.
            # N'
            masked_filtered_grad_list = self._precondition(
                state_lists,
                masked_filtered_grad_list,
            )


            #G'
            #V
            # self._update_adam_preconditioners(
            #     state_lists,
            #     step,
            # )
            
            # # M and update M
            # masked_filtered_grad_list1 = self._compute_filtered_grad_list(
            #     state_lists,
            #     step,
            #     beta1,
            #     beta3,
            #     use_bias_correction,
            # )

            # # # Precondition filtered gradients.
            # # # M' <- Q^T_L M Q_R
            # # # N'
            # # N
            # masked_filtered_grad_list = self._precondition1(
            #     state_lists,
            #     masked_filtered_grad_list1,
            # )



            
            if step > 1:
                #   L <- L + G * G^T
                #   R <- R + G^T * G
                self._update_preconditioners(
                    state_lists,
                    step,
                )

                # Compute matrix root inverse.
                #   L_inv <- L ** (-1/4)
                #   R_inv <- R ** (-1/4)
                #   (and similar)

                if compute_eigenvectors:
                    self._project_back(state_lists, step)
                    self._compute_eigenvectors(state_lists, compute_eigenvectors)
                    self._project(state_lists, step)

                # update QL and QR
                # self._compute_eigenvectors(state_lists, compute_eigenvectors)
            
              

        # Incorporate decoupled weight decay into search direction if enabled.
        #   P <- P + weight_decay * W
        self._apply_decoupled_weight_decay(
            state_lists,
            masked_filtered_grad_list,
            decoupled_weight_decay,
        )

        # Updates parameters in distributed fashion.
        # If DDP, executes AllGather communication to ensure all parameters are updated after local updates.
        torch._foreach_mul_(masked_filtered_grad_list, -lr)
        
        state_lists[DISTRIBUTOR].update_params(
            masked_blocked_search_directions=masked_filtered_grad_list
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Construct blocked gradient list.
            state_lists[MASKED_BLOCKED_GRADS] = state_lists[
                DISTRIBUTOR
            ].merge_and_block_gradients()

            # Based on the current block selector, mask lists of parameters and optimizer states.
            DistributedSOAP._mask_state_lists(state_lists, group)

            # Check if gradient list is empty. If so, continue.
            if not state_lists[MASKED_BLOCKED_GRADS]:
                continue

            # Iterate group step counter and define Python scalar step.
            step = state_lists[STEP].add_(1)
            # NOTE: Wrap scalar of group[LR] into a 0D tensor to avoid PT2 recompilation;
            # Send 0D tensor to GPU in `non_blocking` to avoid QPS regression. Remove the gpu
            # tensor impl once PT2 supports cpu 0D tensor properly.
            lr = torch.tensor(group[LR], dtype=torch.float).to(
                self._device, non_blocking=True
            )
            beta1 = group[BETAS][0]
            beta3 = group[BETA3]
            decoupled_weight_decay = group[DECOUPLED_WEIGHT_DECAY]
            
            if state_lists[PRECONDITION_FREQUENCY] == 0:
                if self._use_pf_warmup:
                    state_lists[PRECONDITION_FREQUENCY] = np.minimum(10, group[PRECONDITION_FREQUENCY])
                else:
                    state_lists[PRECONDITION_FREQUENCY] = group[PRECONDITION_FREQUENCY]
                    
            if state_lists[PRECONDITION_FREQUENCY] < group[PRECONDITION_FREQUENCY]:
                if state_lists[STEP] > state_lists[ACCUM_STEPS] + state_lists[PRECONDITION_FREQUENCY]*self._step_delay:
                    state_lists[ACCUM_STEPS] = state_lists[STEP]
                    state_lists[PRECONDITION_FREQUENCY] = np.minimum(group[PRECONDITION_FREQUENCY], state_lists[PRECONDITION_FREQUENCY]*2)
            # Check compute root inverse or not for preconditioner
            compute_eigenvectors = (
                (step.item() % state_lists[PRECONDITION_FREQUENCY] == 0) or (step == 1)
            )
            use_bias_correction = group[USE_BIAS_CORRECTION]

            self._per_group_step(
                state_lists,
                step,
                lr,
                beta1,
                beta3,
                decoupled_weight_decay,
                compute_eigenvectors,
                use_bias_correction,
            )

        return loss

    def state_dict(self) -> None:
        raise NotImplementedError(
            "Distributed Shampoo does not support the standard state_dict() method for checkpointing!"
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError(
            "Distributed Shampoo does not support the standard load_state_dict() method for checkpointing!"
        )

    @staticmethod
    def _construct_param_group_key(
        group: Dict[str, Any], param_to_key: Dict[torch.Tensor, str]
    ) -> str:
        return "/".join(sorted(param_to_key[param] for param in group[PARAMS]))

    def distributed_state_dict(
        self,
        key_to_param: Iterator[Tuple[str, torch.Tensor]],
        save_param_groups: bool = True,
    ) -> Dict[str, Any]:
        """Distributed state dict simplified from TorchRec's KeyedOptimizer.
        Compatible with torch.distributed.checkpoint with DTensor.

        Returned state and param_groups will contain parameter keys
        instead of parameter indices in torch.Optimizer.
        This allows for advanced functionality like optimizer re-sharding to be implemented.

        Can also handle classes and supported data structures that follow the PyTorch stateful
        protocol.

        Args:
            key_to_param (Iterator[Tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
                maps a FQN to the parameters in the model.
            save_param_groups (bool): Flag for saving parameter groups. (Default: True)

        Returns:
            state_dict (Dict[str, Any]): Dictionary containing the optimizer state and potentially parameter
                groups.

        """

        # Create mapping from parameter to its name. Generate flattened state dictionary for state.
        param_to_key = {param: key for key, param in key_to_param}
        ret: Dict[str, Any] = {
            "state": {
                param_to_key[param]: flatten(extract_state_dict_content(param_state))
                for param, param_state in self.state.items()
            }
        }
        if not save_param_groups:
            return ret

        # Store parameter groups with unique parameter group identifier.
        # NOTE: The parameters are ignored since they are assumed to be checkpointed separately.
        ret["param_groups"] = {
            self._construct_param_group_key(group, param_to_key): {
                k: deepcopy(v) for k, v in group.items() if k != PARAMS
            }
            for group in self.param_groups
        }

        return ret

    def load_distributed_state_dict(
        self,
        state_dict: Mapping[str, Any],
        key_to_param: Iterator[Tuple[str, torch.Tensor]],
        save_param_groups: bool = True,
        enable_missing_key_check: bool = True,
    ) -> None:
        """Load state dict simplified from TorchRec's KeyedOptimizer.
        Compatible with torch.distributed.checkpoint.

        This implementation is much stricter than the one in torch.Optimizer:
        it requires implementations to fully initialize their state during first optimization iteration,
        and it prohibits loading an empty state into already initialized KeyedOptimizer and vise versa.

        Because of introduced strictness it allows us to:
            * do compatibility checks for state and param_groups, which improves usability
            * avoid state duplication by directly copying into state tensors, e.g.
              optimizer.step()  # make sure optimizer is initialized
              sd = optimizer.state_dict()
              load_checkpoint(sd)  # copy state directly into tensors, re-shard if needed
              optimizer.load_state_dict(sd)  # replace param_groups

        Args:
            state_dict (Dict[str, Any]): State dictionary to load containing the optimizer state and
                parameter groups.
            key_to_param (Iterator[Tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
                maps a FQN to the parameters in the model.
            save_param_groups (bool): Flag for saving parameter groups. (Default: True)
            enable_missing_key_check (bool): Flag for enabling missing key check. (Default: True)

        """

        # Create mapping from parameter to its name. Generate flattened state dictionary for state.
        state_to_load = state_dict["state"]
        key_to_param_mapping = dict(key_to_param)

        # Load state
        for param_key, param_state in state_to_load.items():
            # Check if parameter exists in current parameter state dict.
            if param_key not in key_to_param_mapping:
                if enable_missing_key_check:
                    raise KeyError(
                        f"Parameter key {param_key} not found in key_to_param mapping!"
                    )
                else:
                    logger.warning(
                        f"Parameter key {param_key} not found in key_to_param mapping!"
                    )
                    continue

            param = key_to_param_mapping[param_key]

            if param not in self.state:
                if enable_missing_key_check:
                    raise KeyError(f"Parameter {param} not found in state!")
                else:
                    logger.warning(f"Parameter {param} not found in state!")
                    continue

            # Update parameter state.
            update_param_state_dict_object(
                self.state[param],
                unflatten(param_state),
            )

        # Load param_groups.
        if save_param_groups:
            param_groups_to_load = state_dict["param_groups"]
            param_groups = self.param_groups

            if len(param_groups) != len(param_groups_to_load):
                raise ValueError(
                    f"Different param_groups count: {len(param_groups)} vs {len(param_groups_to_load)}"
                )
            param_to_key = {param: key for key, param in key_to_param_mapping.items()}

            # Loading the parameter group based on the unique parameter group key.
            for group in param_groups:
                param_group_key = self._construct_param_group_key(group, param_to_key)
                if param_group_key not in param_groups_to_load:
                    raise ValueError(
                        f"Param group {param_group_key} not found in param_groups_to_load!"
                    )
                param_group_to_load = param_groups_to_load[param_group_key]
                for key, value in param_group_to_load.items():
                    group[key] = deepcopy(value)