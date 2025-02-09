"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""



import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from itertools import chain
from operator import methodcaller
from typing import Any, DefaultDict, Sequence, Tuple, Union

import torch
from distributed_soap.utils.shampoo_block_info import BlockInfo
from distributed_soap.utils.shampoo_quantization import (
    QuantizedTensor,
    QuantizedTensorList,
)
from distributed_soap.utils.shampoo_utils import (
    compress_list,
    get_dtype_size,
    ParameterizeEnterExitContext,
)

from distributed_soap.utils.matrix_functions import (
    matrix_eigendecomposition,
)
from distributed_soap.utils.optimizer_modules import OptimizerModule
from torch import Tensor
from torch.autograd import profiler


logger: logging.Logger = logging.getLogger(__name__)

ADAGRAD = "adagrad"
SHAMPOO = "shampoo"


class PreconditionerList(ABC):
    """Preconditioner base class.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
    ) -> None:
        super().__init__()
        self._numel_list: Tuple[int, ...] = (0,) * len(block_list)
        self._dims_list: Tuple[torch.Size, ...] = tuple(
            block.size() for block in block_list
        )
        self._num_bytes_list: Tuple[int, ...] = (0,) * len(block_list)

    @abstractmethod
    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None: ...

    @abstractmethod
    def precondition(
        self, masked_grad_list: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]: ...

    @abstractmethod
    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None: ...

    @abstractmethod
    def dequantize_preconditioners(self) -> None: ...

    @abstractmethod
    def quantize_preconditioners(self) -> None: ...

    @property
    def numel_list(self) -> Tuple[int, ...]:
        return self._numel_list

    @property
    def dims_list(self) -> Tuple[torch.Size, ...]:
        return self._dims_list

    @property
    def num_bytes_list(self) -> Tuple[int, ...]:
        return self._num_bytes_list

    def numel(self) -> int:
        return sum(self._numel_list)

    def num_bytes(self) -> int:
        return sum(self._num_bytes_list)

from contextlib import contextmanager

@contextmanager
def set_matmul_precision(precision):
    # Store the current precision
    previous_precision = torch.get_float32_matmul_precision()
    
    # Set the new precision
    torch.set_float32_matmul_precision(precision)
    
    try:
        yield
    finally:
        # Restore the previous precision after the block
        torch.set_float32_matmul_precision(previous_precision)

@dataclass
class ShampooKroneckerFactorsState(OptimizerModule):
    """Shampoo Kronecker Factors (wrapped) for storing in the optimizer state."""

    factor_matrices: Tuple[Union[QuantizedTensor, None], ...]
    eig_factor_matrices: Tuple[Union[QuantizedTensor, None], ...]
    inner_optimizer_state: Tuple[Union[QuantizedTensor, None], ...]
    factor_matrix_indices: Tuple[str, ...]

    def __post_init__(self) -> None:
        super().__init__()
        assert (
            len(self.factor_matrices)
            == len(self.eig_factor_matrices)
            == len(self.factor_matrix_indices)
        )


@dataclass
class ShampooKroneckerFactorsList(OptimizerModule):
    """Shampoo Kronecker Factors (unwrapped) for operations during optimizer computation."""

    factor_matrices: QuantizedTensorList
    eig_factor_matrices: QuantizedTensorList
    inner_optimizer_state: QuantizedTensorList
    factor_matrix_indices: Tuple[str, ...]

    def __post_init__(self) -> None:
        super().__init__()
        assert (
            len(self.factor_matrices)
            == len(self.eig_factor_matrices)
            == len(self.factor_matrix_indices)
        )


class ShampooPreconditionerList(PreconditionerList):
    """Shampoo preconditioners for list of parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.
        state (DefaultDict[Tensor, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum.
            (Default: 1.0)
        epsilon (float): Epsilon term for regularizing Adam. (Default: 1e-12)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        factor_matrix_dtype (torch.dtype): Data type for storing Shampoo factor matrices. (Default: torch.float)
        eig_factor_matrix_dtype (torch.dtype): Data type for storing Shampoo inverse factor matrices. (Default: torch.float)
        computation_dtype (torch.dtype): Data type for computation (i.e., matrix inverse) is performed in. (Default: torch.float)

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        max_precond_dim_diag: int = 10000,
        epsilon: float = 1e-8,
        use_bias_correction: bool = True,
        factor_matrix_dtype: torch.dtype = torch.float,
        eig_factor_matrix_dtype: torch.dtype = torch.float,
        preconditioner_dtype: torch.dtype = torch.float,
        computation_dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._beta2 = beta2
        self._max_precond_dim_diag = max_precond_dim_diag
        self._epsilon = epsilon
        self._factor_matrix_dtype = factor_matrix_dtype
        self._eig_factor_matrix_dtype = eig_factor_matrix_dtype
        self._preconditioner_dtype = preconditioner_dtype
        self._computation_dtype = computation_dtype
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        # Instantiate (blocked) Kronecker factors and construct list of Kronecker factors.
        # NOTE: We need to instantiate the Kronecker factor states within the optimizer's state dictionary,
        # and do not explicitly store them as ShampooPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but ShampooPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        kronecker_factors_list = []
        for block, block_info, dims in zip(
            block_list, block_info_list, self._dims_list, strict=True
        ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            # Instantiate ShampooKroneckerFactors for this block.
            # The factor matrices are instantiated using the determined dtype.
            factor_matrices = tuple(
                QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        (dim, dim),
                        self._factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                ) if dim <= self._max_precond_dim_diag else QuantizedTensor( #Just a dummy tensor to satisy the code, needs to be updated.
                    block_info.allocate_zeros_tensor(
                        (1,1),
                        self._factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                )
                for dim in dims
            )
            # The inverse factor matrices are instantiated using the dtype of the block / gradient.
            eig_factor_matrices = tuple(
                QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        (dim, dim),
                        self._eig_factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                ) if dim <= self._max_precond_dim_diag else QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        (1,1),
                        self._eig_factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                )
                for dim in dims
            )

            inner_optimizer_state = tuple([QuantizedTensor(
                block_info.allocate_zeros_tensor(
                    dims,
                    torch.float, #Should probably be changed to preconditioner_dtype?
                    block_info.param.device,
                ),
                block_info,
            )])

            preconditioner_index = str(param_index) + "." + str(block_index)
            factor_matrix_indices = tuple(
                preconditioner_index + "." + str(k) for k in range(len(dims))
            )
            block_state[SHAMPOO] = ShampooKroneckerFactorsState(
                factor_matrices=factor_matrices,
                eig_factor_matrices=eig_factor_matrices,
                inner_optimizer_state=inner_optimizer_state,
                factor_matrix_indices=factor_matrix_indices,
            )
            kronecker_factors_list.append(
                ShampooKroneckerFactorsList(
                    factor_matrices=QuantizedTensorList(
                        factor_matrices,
                        self._factor_matrix_dtype,
                        self._computation_dtype,
                    ),
                    eig_factor_matrices=QuantizedTensorList(
                        eig_factor_matrices,
                        self._eig_factor_matrix_dtype,
                        self._computation_dtype,
                    ),
                    inner_optimizer_state=QuantizedTensorList(
                        inner_optimizer_state,
                        torch.float,
                        torch.float,
                    ),
                    factor_matrix_indices=factor_matrix_indices,
                )
            )

            logger.info(
                f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                f"({[(factor_matrix.quantized_values.shape, factor_matrix.quantized_values.dtype) for factor_matrix in block_state[SHAMPOO].factor_matrices]}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        # Initialize local lists.
        local_block_list = compress_list(block_list, distributor_selector)
        self._local_kronecker_factors_list: Tuple[ShampooKroneckerFactorsList, ...] = (
            compress_list(kronecker_factors_list, distributor_selector)
        )
        self._local_order_list: Tuple[int, ...] = tuple(
            block.dim() for block in local_block_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: Tuple[int, ...] = self._local_order_list
        self._masked_kronecker_factors_list: Tuple[ShampooKroneckerFactorsList, ...] = (
            self._local_kronecker_factors_list
        )

        # Construct lists of bytes and numels for logging purposes.
        # NOTE: These lists are constructed across all blocked parameters.
        self._dims_list: Tuple[torch.Size, ...] = compress_list(
            self._dims_list, distributor_selector
        )
        self._numel_list: Tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in dims) for dims in self._dims_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            numel
            * (get_dtype_size(self._factor_matrix_dtype) + get_dtype_size(block.dtype))
            // 2
            for numel, block in zip(self._numel_list, local_block_list, strict=True)
        )


    def update_preconditioners(
        self, masked_grad_list: Tuple[Tensor, ...], step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
            # We apply foreach operators onto the list of Kronecker factor matrices (as opposed to the
            # full list of gradients/optimizer states).
            # with set_matmul_precision('highest'):
            for grad, order, kronecker_factors in zip(
                masked_grad_list,
                self._masked_order_list,
                self._masked_kronecker_factors_list,
                strict=True,
            ):
                # Scale Kronecker factors as a list.
                if self._beta2 != 1.0:
                    torch._foreach_mul_(
                        kronecker_factors.factor_matrices.dequantized_value, self._beta2
                    )

                # Construct outer product list for updating Kronecker factors.
                
                outer_product_list = tuple(
                    torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(k), range(k + 1, order))]] * 2,
                    ) if grad.shape[k] <= self._max_precond_dim_diag else torch.tensor([[1.0]], dtype=grad.dtype, device=grad.device)
                    for k in range(order)
                )
                
                # Update Kronecker factors.
                torch._foreach_add_(
                    kronecker_factors.factor_matrices.dequantized_value,
                    outer_product_list,
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )
                
                
                

            # Update bias correction term based on step list.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step
                
    def update_adam_preconditioners(
        self, masked_grad_list: Tuple[Tensor, ...], step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_adam_preconditioners.__name__} ##"
        ):
            
            def update_adam_preconditioner_single(
                masked_grad: Tensor,
                eig_factor_matrices: Tuple[Tensor, ...],
                inner_optimizer_state: Tensor,
                bias_correction2: Tensor,
                epsilon: float,
            ):
                
                # G' <- Q^T_L G Q_R
                for eig_factor_matrix in eig_factor_matrices:
                    # print(eig_factor_matrix.shape)
                    if eig_factor_matrix.shape[0] > 1:
                        masked_grad = torch.tensordot(
                            masked_grad, eig_factor_matrix, [[0], [0]]
                        )
                    else:
                        masked_grad = torch.movedim(masked_grad, 0, -1)
                    
                # Update inner optimizer state
                # V <- B_2 V + (1 - B_2) G'^2
                inner_optimizer_state.mul_(self._beta2).add_(masked_grad.square(), alpha=(1.0 - self._beta2)) 

                
                return

            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.inner_optimizer_state.dequantize_()
                
            for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, strict=True
                ):
                update_adam_preconditioner_single(
                    masked_grad=masked_grad,
                    eig_factor_matrices=kronecker_factors.eig_factor_matrices.dequantized_value,
                    inner_optimizer_state=kronecker_factors.inner_optimizer_state.dequantized_value[0],
                    bias_correction2=self._bias_correction2,
                    epsilon=self._epsilon,
                )

            
            return
        
    def update_adam_and_momentum(
        self, masked_grad_list: Tuple[Tensor, ...], masked_filtered_grad_list: Tuple[Tensor, ...], beta1: float, beta3: float, use_bias_correction: bool, step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_adam_preconditioners.__name__} ##"
        ):
            
            def update_adam_and_momentum_single(
                masked_grad: Tensor,
                masked_filtered_grad: Tensor,
                eig_factor_matrices: Tuple[Tensor, ...],
                inner_optimizer_state: Tensor,
                beta1: float,
                beta3: float,
                use_bias_correction: bool,
            ):
                
                # G' <- Q^T_L G Q_R
                for eig_factor_matrix in eig_factor_matrices:
                    # print(eig_factor_matrix.shape)
                    if eig_factor_matrix.shape[0] > 1:
                        masked_grad = torch.tensordot(
                            masked_grad, eig_factor_matrix, [[0], [0]]
                        )
                    else:
                        masked_grad = torch.movedim(masked_grad, 0, -1)

                state_list_update = masked_grad
                    
                # Update inner optimizer state
                # V <- B_2 V + (1 - B_2) G'^2
                inner_optimizer_state.mul_(self._beta2).add_(masked_grad.square(), alpha=(1.0 - self._beta2)) 

                # Compute filtered gradient or EMA of gradients
                masked_filtered_grad = (
                    torch.lerp(
                        masked_filtered_grad,
                        masked_grad,
                        weight=1-beta3,
                    )
                    if beta3 != beta1
                    else masked_filtered_grad
                )

                # #M' <- B1*M' + (1-B1)G'
                # state_list_update = torch.lerp(
                #     masked_filtered_grad,
                #     masked_grad,
                #     weight=1-beta1,
                # )

                if use_bias_correction:
                    bias_correction1 = 1.0 - beta3 * beta1 **(step-1)
                    masked_filtered_grad = torch.div(
                        masked_filtered_grad,
                        bias_correction1
                    )




                return masked_filtered_grad, state_list_update

            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.inner_optimizer_state.dequantize_()
            
            masked_filtered_grads = []
            state_list_updates = []
            for masked_grad, masked_filtered_grad, kronecker_factors in zip(
                    masked_grad_list, masked_filtered_grad_list, self._masked_kronecker_factors_list, strict=True
                ):
                masked_filtered_grad, state_list_update = update_adam_and_momentum_single(
                    masked_grad=masked_grad,
                    masked_filtered_grad = masked_filtered_grad,
                    eig_factor_matrices=kronecker_factors.eig_factor_matrices.dequantized_value,
                    inner_optimizer_state=kronecker_factors.inner_optimizer_state.dequantized_value[0],
                    beta1=beta1,
                    beta3=beta3,
                    use_bias_correction=use_bias_correction,
                )
                masked_filtered_grads.append(masked_filtered_grad)
                state_list_updates.append(state_list_update)
            
            return tuple(masked_filtered_grads), tuple(state_list_updates)
        
    def project(
        self, masked_filtered_grad_list: Tuple[Tensor, ...], step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.project.__name__} ##"
        ):
            
            def project_single(
                masked_filtered_grad: Tensor,
                eig_factor_matrices: Tuple[Tensor, ...],
            ):
                
                # M' <- Q^T_L M Q_R
                for eig_factor_matrix in eig_factor_matrices:
                    # print(eig_factor_matrix.shape)
                    if eig_factor_matrix.shape[0] > 1:
                        masked_filtered_grad = torch.tensordot(
                            masked_filtered_grad, eig_factor_matrix, [[0], [0]]
                        )
                    else:
                        masked_filtered_grad = torch.movedim(masked_filtered_grad, 0, -1)
                    
                return masked_filtered_grad
            

            ret_val = tuple(
                project_single(
                    masked_filtered_grad=masked_filtered_grad,
                    eig_factor_matrices=kronecker_factors.eig_factor_matrices.dequantized_value,
                )
                for masked_filtered_grad, kronecker_factors in zip(
                    masked_filtered_grad_list, self._masked_kronecker_factors_list, strict=True
                )
            )
            
            return ret_val
        

    def project_back(
        self, masked_filtered_grad_list: Tuple[Tensor, ...], step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.project.__name__} ##"
        ):
            
            def project_back_single(
                masked_filtered_grad: Tensor,
                eig_factor_matrices: Tuple[Tensor, ...],
            ):
                
                # M' <- Q^T_L M Q_R
                for eig_factor_matrix in eig_factor_matrices:
                    if eig_factor_matrix.shape[0] > 1:
                        masked_filtered_grad = torch.tensordot(
                            masked_filtered_grad, eig_factor_matrix, [[0], [1]]
                        )
                    else:
                        masked_filtered_grad = torch.movedim(masked_filtered_grad, 0, -1)
                    
                return masked_filtered_grad

            
            ret_val = tuple(
                project_back_single(
                    masked_filtered_grad=masked_filtered_grad,
                    eig_factor_matrices=kronecker_factors.eig_factor_matrices.dequantized_value,
                )
                for masked_filtered_grad, kronecker_factors in zip(
                    masked_filtered_grad_list, self._masked_kronecker_factors_list, strict=True
                )
            )

            return ret_val

    def precondition1(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition1.__name__} ##"
        ):

            def precondition_masked_grad1(
                masked_grad: Tensor,
                eig_factor_matrices: Tuple[Tensor, ...],
                inner_optimizer_state: Tensor,
                bias_correction2: Tensor,
                epsilon: float,
            ) -> Tensor:
                for eig_factor_matrix in eig_factor_matrices:
                    # print(eig_factor_matrix.shape)
                    if eig_factor_matrix.shape[0] > 1:
                        masked_grad = torch.tensordot(
                            masked_grad, eig_factor_matrix, [[0], [0]]
                        )
                    else:
                        masked_grad = torch.movedim(masked_grad, 0, -1)
                
                # # N'
                denom = inner_optimizer_state.sqrt() + epsilon # tofill, new operation
                denom /= bias_correction2 ** 0.5
                masked_grad /= denom
                
                for eig_factor_matrix in eig_factor_matrices:
                    if eig_factor_matrix.shape[0] > 1:
                        masked_grad = torch.tensordot(
                            masked_grad, eig_factor_matrix, [[0], [1]]
                        )
                    else:
                        masked_grad = torch.movedim(masked_grad, 0, -1)
                
                
                return masked_grad

            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.inner_optimizer_state.dequantize_()
            
            ret_val = tuple(
                precondition_masked_grad1(
                    masked_grad=masked_grad,
                    eig_factor_matrices=kronecker_factors.eig_factor_matrices.dequantized_value,
                    inner_optimizer_state=kronecker_factors.inner_optimizer_state.dequantized_value[0],
                    bias_correction2=self._bias_correction2,
                    epsilon=self._epsilon,
                )
                for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, strict=True
                )
            )
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.inner_optimizer_state.quantize_()
            
            # exit()
            
            return ret_val
                

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):

            def precondition_masked_grad(
                masked_grad: Tensor,
                eig_factor_matrices: Tuple[Tensor, ...],
                inner_optimizer_state: Tensor,
                bias_correction2: Tensor,
                epsilon: float,
            ) -> Tensor:
                
                # N'
                denom = inner_optimizer_state.sqrt() + epsilon # tofill, new operation
                denom /= bias_correction2 ** 0.5
                masked_grad /= denom
                
                for eig_factor_matrix in eig_factor_matrices:
                    if eig_factor_matrix.shape[0] > 1:
                        masked_grad = torch.tensordot(
                            masked_grad, eig_factor_matrix, [[0], [1]]
                        )
                    else:
                        masked_grad = torch.movedim(masked_grad, 0, -1)
                
                
                return masked_grad

            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.inner_optimizer_state.dequantize_()
            
            ret_val = tuple(
                precondition_masked_grad(
                    masked_grad=masked_grad,
                    eig_factor_matrices=kronecker_factors.eig_factor_matrices.dequantized_value,
                    inner_optimizer_state=kronecker_factors.inner_optimizer_state.dequantized_value[0],
                    bias_correction2=self._bias_correction2,
                    epsilon=self._epsilon,
                )
                for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, strict=True
                )
            )
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.inner_optimizer_state.quantize_()
            
            # exit()
            
            return ret_val

    def compute_eigenvectors(self) -> None:
        # NOTE: This function currently only computes the matrix eigendecomposition based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compute_eigenvectors.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                ind = 0
                for (
                    factor_matrix,
                    eig_factor_matrix,
                    factor_matrix_index,
                ) in zip(
                    kronecker_factors.factor_matrices.dequantized_value,
                    kronecker_factors.eig_factor_matrices.dequantized_value,
                    kronecker_factors.factor_matrix_indices,
                    strict=True,
                ):
                    
                    if factor_matrix.shape[0] == 1:
                        ind += 1
                        continue
                    # Check for nan or inf values.
                    if torch.isnan(factor_matrix).any():
                        raise ValueError(
                            f"Encountered nan values in factor matrix {factor_matrix_index}! "
                            f"To mitigate, check if nan inputs are being passed into the network or nan gradients "
                            f"are being passed to the optimizer."
                            f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                            f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                            f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
                        )
                    if torch.isinf(factor_matrix).any():
                        raise ValueError(
                            f"Encountered inf values in factor matrix {factor_matrix_index}! "
                            f"In some cases, this may be due to divergence of the algorithm. "
                            f"To mitigate, try decreasing the learning rate"
                            f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                            f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                            f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
                        )

                    # Compute inverse preconditioner.
                    # If reuse_previous_eig_factor_matrix is True, will reuse previous matrix if matrix
                    # eigendecomposition fails.
                    with set_matmul_precision('highest'):
                        computed_eig_factor_matrix, sort_idx = matrix_eigendecomposition(
                            A=factor_matrix,
                            prev_eigenvectors=eig_factor_matrix,
                        )
                    
                    
                    computed_eig_factor_matrix = computed_eig_factor_matrix.to(dtype=eig_factor_matrix.dtype)

                    if sort_idx is not None:
                        kronecker_factors.inner_optimizer_state.dequantize_()
                        kronecker_factors.inner_optimizer_state.dequantized_value[0].copy_(kronecker_factors.inner_optimizer_state.dequantized_value[0].index_select(ind, sort_idx))
                        kronecker_factors.inner_optimizer_state.quantize_()
                    ind += 1
                    
                    # Check if we encounter NaN or inf values in computed inverse matrix.
                    if (
                        torch.isnan(computed_eig_factor_matrix).any()
                        or torch.isinf(computed_eig_factor_matrix).any()
                    ):
                        torch.set_printoptions(threshold=100_000)
                        raise ValueError(
                            f"Encountered nan or inf values in inverse factor matrix {factor_matrix_index}! "
                            f"To mitigate, check factor matrix before matrix eigendecomposition: "
                            f"{factor_matrix=}"
                        )

                    eig_factor_matrix.copy_(computed_eig_factor_matrix)

    def dequantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.dequantize_preconditioners.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.factor_matrices.dequantize_()
                kronecker_factors.eig_factor_matrices.dequantize_()

    def quantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.quantize_preconditioners.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.factor_matrices.quantize_()
                kronecker_factors.eig_factor_matrices.quantize_()

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_order_list = compress_list(
                self._local_order_list, local_grad_selector
            )
            self._masked_kronecker_factors_list: Tuple[
                ShampooKroneckerFactorsList, ...
            ] = compress_list(self._local_kronecker_factors_list, local_grad_selector)


class DequantizePreconditionersContext(ParameterizeEnterExitContext):
    """DequantizePreconditionersContext is used for automatically dequantize and then quantize the preconditioners used within this context.

    Args:
        preconditioner_list (PreconditionerList): Preconditioner list which contains the preconditioners to be dequantized and quantized.

    Examples:
        >>> with DequantizePreconditionersContext(preconditioner_list):
        >>>     # Do something with the preconditioners which are dequantized.
        >>> # After the context is exited, the preconditioners will be quantized.

    """

    def __init__(self, preconditioner_list: PreconditionerList) -> None:
        super().__init__(
            input_with_enter_exit_context=preconditioner_list,
            enter_method_caller=methodcaller("dequantize_preconditioners"),
            exit_method_caller=methodcaller("quantize_preconditioners"),
        )
