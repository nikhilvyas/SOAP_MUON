import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import math

from typing import Any, Dict, List, Optional, Tuple, Union

@torch.compile()
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
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
        
    for _ in range(2):
        X = 1.5 * X - 0.5 * (X @ X.T) @ X    
        
    if G.size(0) > G.size(1):
        X = X.T
    return X      
        
class SOAP_Muon15(optim.Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.98),
        shampoo_beta: float= -1,
        eps: float = 1e-15,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        projection_type: str="full",
        precondition_frequency: int=10,
        max_precond_dim: int=10000,
        newton_schulz_steps: int = 5,
        nestrov: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "projection_type": projection_type,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "newton_schulz_steps": newton_schulz_steps,
            "nestrov": nestrov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0 
                    
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                
                
                if 'ortho_matrix' not in state:
                    #state["projector"] = Projector(
                    #    precondition_frequency=group['precondition_frequency'],
                    #    projection_type=group['projection_type'],
                    #    shampoo_beta=group['shampoo_beta']
                    #)
                    
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group['precondition_frequency'],
                        projection_type=group['projection_type'],
                        shampoo_beta=(group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]),
                        max_precond_dim=group['max_precond_dim'],
                    )
                    self.update_preconditioner(grad, state, state["step"])
                    
                    #state["projector"].init_preconditioner(grad)
                    #state["projector"].update(grad, state["step"])
                    continue
                
                self.init_preconditioner2(
                    grad,
                    state,
                    projection_type=group['projection_type'],
                    max_precond_dim=group['max_precond_dim'],
                )

                #grad_projected = state["projector"].project(grad)
                grad_projected = self.project(grad, state)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))


                denom = exp_avg_sq.sqrt().add_(group["eps"])
                
                
                #exp_avg_projected = state["projector"].project(exp_avg)
                exp_avg_projected = self.project(exp_avg, state)
                
                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2 ** .5) / bias_correction1

                if group["nestrov"] and p.dim() == 2 and max(p.shape) < 10000:
                    norm_grad = (exp_avg_projected+(1.0-beta1)*grad_projected) / denom
                else:
                    norm_grad = exp_avg_projected / denom
                
                

                # update = zeroth_power_via_newtonschulz5(norm_grad, steps=group['zeropower_iters'])
                if p.dim() == 2 and max(p.shape) < 10000:
                     
                    norm_grad = zeropower_via_newtonschulz5(norm_grad, steps=group['newton_schulz_steps'], eps=1e-7)
                    
                    median = torch.median(torch.abs(norm_grad))
                    
                    norm_grad = torch.where(torch.abs(norm_grad) > median, median*torch.sign(norm_grad), norm_grad)
                    
                    norm_grad /= torch.norm(norm_grad)
                    norm_grad *= min(norm_grad.size(0), norm_grad.size(1))**0.5
                    norm_grad *= max(1, norm_grad.size(0)/norm_grad.size(1))**0.5
                    
                    
                        
                
                #norm_grad = state["projector"].project_back(norm_grad)
                norm_grad = self.project_back(norm_grad, state)    
                
                p.add_(norm_grad, alpha=-step_size)
                

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    
                self.update_preconditioner(grad, state, state["step"])
        
        return loss
    
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:


        state = {}
        
        if "exp_avg" in self.state[param]:
            state["exp_avg"] = self.state[param]["exp_avg"]
            state["exp_avg_sq"] = self.state[param]["exp_avg_sq"]
        
        return state
    
    # Code below is a modified version of https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py
    
    def init_preconditioner(self, full_rank_grad, state, precondition_frequency=10, projection_type='full', shampoo_beta=0.0, max_precond_dim=10000):
        
        if full_rank_grad.dim() == 1:
            state['projection_type'] = 'identity'
        elif projection_type == 'std':
            if full_rank_grad.shape[0] > full_rank_grad.shape[1]:
                state['projection_type'] = 'right'
            elif full_rank_grad.shape[0] < full_rank_grad.shape[1]:
                state['projection_type'] = 'left'
            else:
                print('Both dimensions equal, using "right"') # Can also use "right" see, https://wandb.ai/harvardml/ShampooAdam/workspace?nw=kf1ze4hfbkh
                state['projection_type'] = 'right'
        else:
            state['projection_type'] = projection_type
                
        if state['projection_type'] == 'full' and full_rank_grad.dim() == 2 and max_precond_dim >= 0:
            if full_rank_grad.shape[0] > max_precond_dim:
                state['projection_type'] = 'right'
            if full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'left'
            if full_rank_grad.shape[0] > max_precond_dim and full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'identity'
                
                
        if state['projection_type'] == 'right':
            state['GG'] = torch.zeros(full_rank_grad.shape[1], full_rank_grad.shape[1], device=full_rank_grad.device)
        elif state['projection_type'] == 'left':
            state['GG'] = torch.zeros(full_rank_grad.shape[0], full_rank_grad.shape[0], device=full_rank_grad.device)
        elif state['projection_type'] == 'full':
            state['GG'] = []
            state['GG'].append(torch.zeros(full_rank_grad.shape[0], full_rank_grad.shape[0], device=full_rank_grad.device))
            state['GG'].append(torch.zeros(full_rank_grad.shape[1], full_rank_grad.shape[1], device=full_rank_grad.device))
                
        state['ortho_matrix'] = None
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta
                
    def init_preconditioner2(self, full_rank_grad, state, projection_type='full', max_precond_dim=10000):
        
        if full_rank_grad.dim() == 1:
            state['projection_type'] = 'identity'
        elif projection_type == 'std':
            if full_rank_grad.shape[0] > full_rank_grad.shape[1]:
                state['projection_type'] = 'right'
            elif full_rank_grad.shape[0] < full_rank_grad.shape[1]:
                state['projection_type'] = 'left'
            else:
                print('Both dimensions equal, using "full"') # Can also use "right" see, https://wandb.ai/harvardml/ShampooAdam/workspace?nw=kf1ze4hfbkh
                state['projection_type'] = 'right'
        else:
            state['projection_type'] = projection_type
                
        if state['projection_type'] == 'full' and full_rank_grad.dim() == 2 and max_precond_dim >= 0:
            if full_rank_grad.shape[0] > max_precond_dim:
                state['projection_type'] = 'right'
            if full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'left'
            if full_rank_grad.shape[0] > max_precond_dim and full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'identity'             
        
    def project(self, full_rank_grad, state):
        if state['projection_type'] == 'identity':
            return full_rank_grad
        elif state['projection_type'] == 'left':
            return torch.matmul(state['ortho_matrix'].t(), full_rank_grad)
        elif state['projection_type'] == 'right':
            return torch.matmul(full_rank_grad, state['ortho_matrix'].t())
        elif state['projection_type'] == 'full':
            return torch.matmul(state['ortho_matrix'][0].t(), full_rank_grad) @ state['ortho_matrix'][1].t()
        
    def update_preconditioner(self, full_rank_grad, state, iter):

        if state['projection_type'] == 'right':
            state['GG'].lerp_(full_rank_grad.T @ full_rank_grad, 1-state['shampoo_beta'])
            if state['ortho_matrix'] is None:
                state['ortho_matrix'] = self.get_orthogonal_matrix(state['GG'], type='right')
            if iter > 0 and iter % state['precondition_frequency'] == 0:
                state['ortho_matrix'] = self.get_orthogonal_matrix_QR(state, type='right')
        elif state['projection_type'] == 'left':
            state['GG'].lerp_(full_rank_grad @ full_rank_grad.T, 1-state['shampoo_beta'])
            if state['ortho_matrix'] is None:
                state['ortho_matrix'] = self.get_orthogonal_matrix(state['GG'], type='left')
            if iter > 0 and iter % state['precondition_frequency'] == 0:
                state['ortho_matrix'] = self.get_orthogonal_matrix_QR(state, type='left')
        elif state['projection_type'] == 'full':
            state['GG'][0].lerp_(full_rank_grad @ full_rank_grad.T, 1-state['shampoo_beta'])
            state['GG'][1].lerp_(full_rank_grad.T @ full_rank_grad, 1-state['shampoo_beta'])
            if state['ortho_matrix'] is None:
                state['ortho_matrix'] = self.get_orthogonal_matrix(state['GG'], type='full')
            if iter > 0 and iter % state['precondition_frequency'] == 0:
                state['ortho_matrix'][0] = self.get_orthogonal_matrix_QR(state, type='left', full=True)
                state['ortho_matrix'][1] = self.get_orthogonal_matrix_QR(state, type='right', full=True)
                

    def project_back(self, low_rank_grad, state):
        
        
        if state['projection_type'] == 'identity':
            return low_rank_grad
        elif state['projection_type'] == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, state['ortho_matrix'])
        elif state['projection_type'] == 'left':
            full_rank_grad = torch.matmul(state['ortho_matrix'], low_rank_grad)
        elif state['projection_type'] == 'full':
            full_rank_grad = torch.matmul(state['ortho_matrix'][0], low_rank_grad) @ state['ortho_matrix'][1]
        
        return full_rank_grad
        
    # svd decomposition
    def get_orthogonal_matrix(self, weights, type):
        module_params = weights

        if type == 'full':
            matrix = []
            if module_params[0].data.dtype != torch.float:
                float_data = False
                original_type = module_params[0].data.dtype
                original_device = module_params[0].data.device
                matrix.append(module_params[0].data.float())
                matrix.append(module_params[1].data.float())
            else:
                float_data = True
                matrix.append(module_params[0].data)
                matrix.append(module_params[1].data)
        else:
            if module_params.data.dtype != torch.float:
                float_data = False
                original_type = module_params.data.dtype
                original_device = module_params.data.device
                matrix = module_params.data.float()
            else:
                float_data = True
                matrix = module_params.data
        


        if type=='right':
            s, Q = torch.linalg.eigh(matrix)
            Q = torch.flip(Q, [1])
            B = Q.T

            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            s, Q = torch.linalg.eigh(matrix)
            Q = torch.flip(Q, [1])
            A = Q
            
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            s, Q = torch.linalg.eigh(matrix[0])
            Q = torch.flip(Q, [1])
            A = Q
            s, Q = torch.linalg.eigh(matrix[1])
            Q = torch.flip(Q, [1])
            B = Q.T
            
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')
        
        
    # svd decomposition
    def get_orthogonal_matrix_QR(self, state, type, full=False):
        module_params = state['GG']
        orth = state['ortho_matrix']
        if type == 'left' and full:
            module_params = state['GG'][0]
            orth = state['ortho_matrix'][0]
        if type == 'right' and full:
            module_params = state['GG'][1]
            orth = state['ortho_matrix'][1]

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
            orth_matrix = orth.data.float()
        else:
            float_data = True
            matrix = module_params.data
            orth_matrix = orth.data
        


        if type=='right':
            
            est_eig = torch.diag(orth_matrix @ matrix @ orth_matrix.T)
            
            sort_idx = torch.argsort(est_eig, descending=True)
            
            state['exp_avg_sq'] = state['exp_avg_sq'].T[sort_idx].T
            
            orth_matrix = orth_matrix[sort_idx].T
            
            power_iter = (orth_matrix.T @ matrix).T
            Q, _ = torch.linalg.qr(power_iter)
            
            B = Q.T

            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            
            est_eig = torch.diag(orth_matrix.T @ matrix @ orth_matrix)
            
            sort_idx = torch.argsort(est_eig, descending=True)
            
            state['exp_avg_sq'] = state['exp_avg_sq'][sort_idx]
            
            orth_matrix = orth_matrix.T[sort_idx].T
            
            power_iter = (orth_matrix.T @ matrix).T
            Q, _ = torch.linalg.qr(power_iter)
            

            A = Q
            
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        else:
            raise ValueError('type should be left or right')