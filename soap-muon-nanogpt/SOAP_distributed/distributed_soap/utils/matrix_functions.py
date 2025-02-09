"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
import math
import time
from fractions import Fraction
from math import isfinite
from typing import Tuple, Union, Optional

import torch
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)

def matrix_eigendecomposition(
    A: Tensor,
    prev_eigenvectors: Optional[Tensor] = None,
) -> Tensor:
    """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        exponent_multiplier (float): exponent multiplier in the eigen method (Default: 1.0)
        root_inv_method (RootInvMethod): Specifies method to use to compute root inverse. (Default: RootInvMethod.EIGEN)
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 1000)
        tolerance (float): Tolerance for computing root inverse using coupled Newton iteration. (Default: 1e-6)
        is_diagonal (Tensor, bool): Flag for whether or not matrix is diagonal. If so, will compute root inverse by computing
            root inverse of diagonal entries. (Default: False)
        order (int): Order used in the higher-order method. (Default: 3)

    Returns:
        X (Tensor): Inverse root of matrix A.

    """

    # check if matrix is scalar
    if torch.numel(A) == 1:
        return torch.as_tensor(1.0)

    # check matrix shape
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")

    if prev_eigenvectors is None:
        X = get_orthogonal_matrix(
            mat=A,
        )
        sort_idx = None
    else:
        X, sort_idx = get_orthogonal_matrix_QR(
            mat=A,
            prev_eigenvectors=prev_eigenvectors,
        )

    return X, sort_idx

def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    if len(mat) == 0:
        return []

    if mat.dtype != torch.float:
        float_data = False
        original_type = mat.dtype
        original_device = mat.device
        matrix = mat.float()
    else:
        float_data = True
        matrix = mat

    try:
        _, Q = torch.linalg.eigh(matrix + 1e-30 * torch.eye(matrix.shape[0], device=matrix.device))
    except:
        _, Q = torch.linalg.eigh(matrix.to(torch.float64) + 1e-30 * torch.eye(matrix.shape[0], device=matrix.device))
        Q = Q.to(matrix.dtype)
    Q = torch.flip(Q, [1])

    if not float_data:
        Q = Q.to(original_device).type(original_type)

    return Q

def get_orthogonal_matrix_QR(mat, prev_eigenvectors):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration 
    followed by torch.linalg.qr decomposition.
    """
    if len(mat) == 0:
        return []

    if mat.dtype != torch.float:
        float_data = False
        original_type = mat.dtype
        original_device = mat.device
        matrix = mat.float()
        orth_matrix = prev_eigenvectors.float()
    else:
        float_data = True
        matrix = mat
        orth_matrix = prev_eigenvectors

    est_eig = torch.diag(orth_matrix.T @ matrix @ orth_matrix)
    sort_idx = torch.argsort(est_eig, descending=True)
    orth_matrix = orth_matrix[:, sort_idx]
    power_iter = matrix @ orth_matrix
    Q, _ = torch.linalg.qr(power_iter)

    if not float_data:
        Q = Q.to(original_device).type(original_type)

    return Q, sort_idx