import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass, fields
from omegaconf import OmegaConf

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from typing import Optional

from distributed_soap.distributed_soap import DistributedSOAP

from soap import SOAP
from soap_p import SOAP_P
from soap_p_layer import SOAP_P_Layer
from soap_p_layer2 import SOAP_P_Layer2
from soapsoap import SOAPSOAP2
from soapsoap3 import SOAPSOAP3

from distributed_soap.shampoo_types import DDPShampooConfig, ShampooPT2CompileConfig, PrecisionConfig
from soap_muon import SOAP_Muon
from soap_muon3 import SOAP_Muon3
from soap_muon5 import SOAP_Muon5
from soap_muon10 import SOAP_Muon10
from soap_muon11 import SOAP_Muon11
from soap_muon12 import SOAP_Muon12
from soap_muon13 import SOAP_Muon13
from soap_muon14 import SOAP_Muon14
from soap_muon15 import SOAP_Muon15
from soap_muon16 import SOAP_Muon16
from soap_muon17 import SOAP_Muon17

from soap_it import SOAP_IT

from soap_muon2 import SOAP_Muon2
from soap_muon2_abl1 import SOAP_Muon2_abl1
from soap_muon2_abl2 import SOAP_Muon2_abl2
from soap_muon2_abl3 import SOAP_Muon2_abl3

from soap_muon10_graft import SOAP_Muon10_Graft
from soap_muon10_graft2 import SOAP_Muon10_Graft2

import random

INIT_SCALE = 1.0

def seed_all(seed: int):
    """Seed all rng objects."""

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-15):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    # X = G.bfloat16()
    X = G.clone()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    
    for _ in range(4):
        X = 1.5 * X - 0.5 * (X @ X.T) @ X    
        
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()
                
class Muon_W(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    g *= (p.size(1)**0.25) * torch.mean(p.data**2)**.25
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()
                
class x (torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    g *= torch.mean(p.data**2)**.25
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()
                

class Muon16(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    
                    thr_val = torch.kthvalue(torch.abs(g.reshape(-1)), int(0.75*g.numel())).values
                    
                    g = torch.where(torch.abs(g) > thr_val, thr_val*torch.sign(g), g)
                    
                    g /= torch.norm(g)
                    g *= min(g.size(0), g.size(1))**0.5
                    g *= max(1, g.size(0)/g.size(1))**0.5

                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel() 
                
class Muon17(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    
                    thr_val = torch.kthvalue(torch.abs(g.reshape(-1)), int(0.875*g.numel())).values
                    
                    g = torch.where(torch.abs(g) > thr_val, thr_val*torch.sign(g), g)
                    
                    g /= torch.norm(g)
                    g *= min(g.size(0), g.size(1))**0.5
                    g *= max(1, g.size(0)/g.size(1))**0.5

                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()   
                
class Muon18(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    
                    thr_val = torch.kthvalue(torch.abs(g.reshape(-1)), int(0.95*g.numel())).values
                    
                    g = torch.where(torch.abs(g) > thr_val, thr_val*torch.sign(g), g)
                    
                    g /= torch.norm(g)
                    g *= min(g.size(0), g.size(1))**0.5
                    g *= max(1, g.size(0)/g.size(1))**0.5

                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()  
                
class Muon_test(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    
                    print(g.norm())
                    
                    g /= torch.norm(g)
                    g *= min(g.size(0), g.size(1))**0.5
                    
                    print(g.norm())
                    g *= max(1, g.size(0)/g.size(1))**0.5

                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()             


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.c_proj.weight.data *= INIT_SCALE
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.c_proj.weight.data *= INIT_SCALE
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head.weight.data.zero_() # @Grad62304977
        self.lm_head.weight.data *= INIT_SCALE
    
    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),)) # @Grad62304977
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = '/n/holylabs/LABS/sham_lab/Everyone/data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = '/n/holylabs/LABS/sham_lab/Everyone/data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    optimizer_name: str = "muon" # ['muon', 'soap', 'soap_muon']
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 4578 # number of iterations to run
    seed: int = 0
    # soap params
    precondition_frequency : int = 1
    max_precond_dim_diag: int = 10000 # set to 780 for one-sided SOAP
    step_delay: int = 7
    use_pf_warmup: bool = True
    beta1 : float = 0.9
    beta2 : float = 0.95
    learning_rate : float = 0.0036
    lr_wte: Optional[float] = None
    lr_lm_head: Optional[float] = None
    # soap muon params
    muon_power: float = 0.0
    newton_schulz_steps: int = 10
    # scheduling params
    warmup_iters : int = 0
    warmdown_iters : int = 1308 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    # wandb params
    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    # soap it params
    inner_loops: int = 10
    # ns_steps: int = 5 # not used, we use newton schulz steps instead
    outer_loops: int = 1
    clip: bool = False
    clip2: bool = False
    
    # sophia params
    sophia_lambda: float = 1.0
    
    
    seed: int = 0
    
    clip_threshold: float = 0.0
    clip_lr_mult: float = 1.0
    
    eps: float = 1e-8
    
    projection_type: str = "full"
    
    adam_power: float = 1.0
    
    normalize_grads: bool = True
    
    nestrov: bool = False
    
    init_scale: float = 0.01


# Override default Hyperparameter values with command line arguments
default_config = OmegaConf.structured(Hyperparameters)
print("Default config: ", default_config)
cli_config = OmegaConf.from_cli()
print("cli config: ", cli_config)
merged_config = OmegaConf.merge(default_config, cli_config)
print("Merged config: ", merged_config)
args = OmegaConf.to_object(merged_config)

seed_all(args.seed)


# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

INIT_SCALE = args.init_scale

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    wandb.init(project="soap-muon-nanogpt", name=args.wandb_run_name, group=args.wandb_group_name, config=args, entity="harvardml")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

if args.optimizer_name == 'muon':
    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    # from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    # enable_cudnn_sdp(True)
    # enable_flash_sdp(False)
    # enable_mem_efficient_sdp(False)
    # enable_math_sdp(False)

    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = Muon(raw_model.transformer.h.parameters(),           lr=args.learning_rate,  momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'muon_w':
    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    # from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    # enable_cudnn_sdp(True)
    # enable_flash_sdp(False)
    # enable_mem_efficient_sdp(False)
    # enable_math_sdp(False)

    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = Muon_W(raw_model.transformer.h.parameters(),           lr=args.learning_rate,  momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3]
# elif args.optimizer_name == 'muon_w2':
#     # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
#     # from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
#     # enable_cudnn_sdp(True)
#     # enable_flash_sdp(False)
#     # enable_mem_efficient_sdp(False)
#     # enable_math_sdp(False)
# 
#     # init the optimizer(s)
#     optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
#     optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
#     optimizer3 = Muon_W2(raw_model.transformer.h.parameters(),           lr=args.learning_rate,  momentum=0.95)
#     optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'muon16':
    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    # from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    # enable_cudnn_sdp(True)
    # enable_flash_sdp(False)
    # enable_mem_efficient_sdp(False)
    # enable_math_sdp(False)

    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = Muon16(raw_model.transformer.h.parameters(),           lr=args.learning_rate,  momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'muon17':
    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    # from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    # enable_cudnn_sdp(True)
    # enable_flash_sdp(False)
    # enable_mem_efficient_sdp(False)
    # enable_math_sdp(False)

    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = Muon17(raw_model.transformer.h.parameters(),           lr=args.learning_rate,  momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'muon18':
    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    # from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    # enable_cudnn_sdp(True)
    # enable_flash_sdp(False)
    # enable_mem_efficient_sdp(False)
    # enable_math_sdp(False)

    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = Muon18(raw_model.transformer.h.parameters(),           lr=args.learning_rate,  momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'muon_test2':
    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    # from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    # enable_cudnn_sdp(True)
    # enable_flash_sdp(False)
    # enable_mem_efficient_sdp(False)
    # enable_math_sdp(False)

    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = Muon_test(raw_model.transformer.h.parameters(),           lr=args.learning_rate,  momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap':
    optimizer1 = SOAP([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                                  {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                                  {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}], 
                                  lr=args.learning_rate, betas=(args.beta1, args.beta2),
                                  precondition_frequency=args.precondition_frequency, sophia_lambda=args.sophia_lambda,
                                  clip_threshold=args.clip_threshold, eps=args.eps, weight_decay=args.weight_decay, projection_type=args.projection_type)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_p':
    optimizer1 = SOAP_P([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                                  {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                                  {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}], 
                                  lr=args.learning_rate, betas=(args.beta1, args.beta2),
                                  precondition_frequency=args.precondition_frequency, adam_power=args.adam_power,
                                  eps=args.eps, weight_decay=args.weight_decay,
                                  normalize_grads=args.normalize_grads)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_p_layer':
    optimizer1 = SOAP_P_Layer([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                                  {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                                  {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}], 
                                  lr=args.learning_rate, betas=(args.beta1, args.beta2),
                                  precondition_frequency=args.precondition_frequency, adam_power=args.adam_power,
                                  eps=args.eps, weight_decay=args.weight_decay,
                                  normalize_grads=args.normalize_grads)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_p_layer2':
    optimizer1 = SOAP_P_Layer2([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                                  {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                                  {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}], 
                                  lr=args.learning_rate, betas=(args.beta1, args.beta2),
                                  precondition_frequency=args.precondition_frequency, adam_power=args.adam_power,
                                  eps=args.eps, weight_decay=args.weight_decay,
                                  normalize_grads=args.normalize_grads)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soapsoap':
    optimizer1 = SOAPSOAP2([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, projection_type=args.projection_type,
                            precondition_frequency=args.precondition_frequency, weight_decay=args.weight_decay)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soapsoap3':
    optimizer1 = SOAPSOAP3([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, projection_type=args.projection_type,
                            precondition_frequency=args.precondition_frequency, weight_decay=args.weight_decay)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon2':
    optimizer1 = SOAP_Muon2([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon2_exp':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon2([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon10':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon10([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon10_graft':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon10_Graft([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon10_graft2':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon10_Graft2([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon10_full':
    optimizer1 = SOAP_Muon10([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), newton_schulz_steps=10, precondition_frequency=1)
    optimizer2 = SOAP_Muon10([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), newton_schulz_steps=10, precondition_frequency=1)
    optimizer3 = SOAP_Muon10([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon10_full2':
    optimizer1 = SOAP([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), precondition_frequency=1)
    optimizer2 = SOAP([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), precondition_frequency=1)
    optimizer3 = SOAP_Muon10([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon11':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon11([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon12':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon12([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon13':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon13([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon14':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon14([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon15':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon15([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon16':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon16([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon16_2':
    optimizer1 = SOAP_Muon16([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)},
                              {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                              {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon17_2':
    optimizer1 = SOAP_Muon17([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)},
                              {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                              {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon17':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon17([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, .95)}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type, nestrov=args.nestrov)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon2_abl1':
    optimizer1 = SOAP_Muon2_abl1([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon2_abl2':
    optimizer1 = SOAP_Muon2_abl2([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon2_abl3':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon2_abl3([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (.95, args.beta2)},],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon2_beta2':
    optimizer1 = SOAP_Muon2([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate, 'betas': (args.beta1, .99)}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon3':
    optimizer1 = SOAP_Muon3([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, adam_power=.5, projection_type=args.projection_type)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon4_2':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=args.lr_wte if args.lr_wte else args.learning_rate,   betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_lm_head if args.lr_lm_head else args.learning_rate, betas=(0.9, 0.95), fused=True)
    optimizer3 = SOAP_Muon3([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}], 
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, adam_power=0.001,
                            projection_type=args.projection_type)
    optimizers = [optimizer1, optimizer2, optimizer3]
elif args.optimizer_name == 'soap_muon4':
    optimizer1 = SOAP_Muon3([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, adam_power=0.01, projection_type=args.projection_type)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon5':
    optimizer1 = SOAP_Muon5([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, projection_type=args.projection_type)
    optimizers = [optimizer1]
elif args.optimizer_name == 'soap_muon7':
    optimizer1 = SOAP_Muon3([{'params': raw_model.transformer.h.parameters(), 'lr': args.learning_rate}, 
                            {'params': raw_model.lm_head.weight, 'lr': args.lr_lm_head if args.lr_lm_head else args.learning_rate},
                            {'params': raw_model.transformer.wte.weight, 'lr': args.lr_wte if args.lr_wte else args.learning_rate}],
                            lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency, max_precond_dim=args.max_precond_dim_diag,
                            newton_schulz_steps=args.newton_schulz_steps, adam_power=.5,
                            projection_type=args.projection_type)
    optimizers = [optimizer1]
else:
    raise ValueError(f"Optimizer {args.optimizer_name} not supported.")
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = '/n/netscratch/kempner_sham_lab/Lab/soap-muon/logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = '/n/netscratch/kempner_sham_lab/Lab/soap-muon/logs/%s.txt' % run_id
    # create the log file

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            wandb.log({"val_loss": val_loss}, step=step+1)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        # log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        # torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        # print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        wandb.log({"train_loss": train_loss.item(), "step_avg": approx_time/timed_steps}, step=step+1)
    
    if torch.isnan(train_loss).any():
        print('nan loss')
        exit()
    
if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
