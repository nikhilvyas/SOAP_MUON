import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain

from typing import Dict, Optional, Tuple

import math



class SOAPSOAP2(optim.Optimizer):
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
        adam_power: float=1.0,
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
            "adam_power": adam_power,
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
                    state["exp_avg_sq"] = []
                    state["exp_avg_sq"].append(torch.zeros_like(grad))
                    state["exp_avg_sq"].append(torch.zeros_like(grad))
                    state["exp_avg_sq"].append(torch.zeros_like(grad))
                
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
                    self.update_preconditioner1(grad, state, state["step"])
                    self.update_preconditioner2(grad, state, state["step"])
                    
                    #state["projector"].init_preconditioner(grad)
                    #state["projector"].update(grad, state["step"])
                    continue
                
                self.init_preconditioner2(
                    grad,
                    state,
                    projection_type=group['projection_type'],
                    max_precond_dim=group['max_precond_dim'],
                )
                
                beta1, beta2 = group["betas"]
                
                exp_avg = state["exp_avg"]
                exp_avg_sq_0 = state["exp_avg_sq"][0]
                exp_avg_sq_1 = state["exp_avg_sq"][1]
                
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                # exp_avg_sq_0.mul_(beta2).add_(grad.square(), alpha=(1.0 - beta2))
                
                # grad /= exp_avg_sq_0.sqrt().add_(group["eps"])
                # exp_avg_0 = exp_avg / exp_avg_sq_0.sqrt().add_(group["eps"])

                #grad_projected = state["projector"].project(grad)
                grad_projected = self.project1(grad, state)
                

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                
                exp_avg_sq_0.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))


                denom1 = exp_avg_sq_0.sqrt().add_(group["eps"])
                
                
                #exp_avg_projected = state["projector"].project(exp_avg)
                exp_avg_projected = self.project1(exp_avg, state)
                
                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2 ** .5) / bias_correction1

                exp_avg_projected1 = exp_avg_projected / denom1
                grad_projected1 = grad_projected / denom1
                
                grad_projected2 = self.project2(grad_projected1, state)
                exp_avg_projected2 = self.project2(exp_avg_projected1, state)
                
                exp_avg_sq_1.mul_(beta2).add_(grad_projected2.square(), alpha=(1.0 - beta2))
                
                denom2 = exp_avg_sq_1.sqrt().add_(group["eps"])
                
                exp_avg_projected2 /= denom2
                
                norm_grad = self.project_back(exp_avg_projected2, state)

                # update = zeroth_power_via_newtonschulz5(norm_grad, steps=group['zeropower_iters'])
                # if p.dim() == 2 and max(p.shape) < 10000:
                #      
                #     if math.isclose(group['muon_power'], 0.0):
                #         norm_grad_U, norm_grad_S, norm_grad_Vt = torch.linalg.svd(norm_grad, full_matrices=False)
                #         norm_grad = norm_grad_U @ norm_grad_Vt
                #     elif not math.isclose(group['muon_power'], 1.0):
                #         norm_grad_U, norm_grad_S, norm_grad_Vt = torch.linalg.svd(norm_grad, full_matrices=False)
                #         norm_grad = norm_grad_U @ torch.diag(norm_grad_S**group['muon_power']) @ norm_grad_Vt
                
                # norm_grad /= torch.mean(norm_grad**2)**.5
                
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
                    
                self.update_preconditioner1(grad, state, state["step"])
                if math.isclose(group['adam_power'], 1.0):
                    self.update_preconditioner2(grad_projected1, state, state["step"])
                else:
                    self.update_preconditioner2(grad_projected1/(denom1**group['adam_power']), state, state["step"])
        
        return loss
    
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:


        state = {}
        
        if "exp_avg" in self.state[param]:
            state["exp_avg"] = self.state[param]["exp_avg"]
            # state["exp_avg_sq"] = self.state[param]["exp_avg_sq"]
        
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
                # print('Both dimensions equal, using "full"') # Can also use "right" see, https://wandb.ai/harvardml/ShampooAdam/workspace?nw=kf1ze4hfbkh
                state['projection_type'] = 'right'
        else:
            state['projection_type'] = projection_type
                
        if state['projection_type'] != 'identity' and full_rank_grad.dim() == 2 and max_precond_dim >= 0:
            if full_rank_grad.shape[0] > max_precond_dim:
                state['projection_type'] = 'only_right'
            if full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'only_left'
            if full_rank_grad.shape[0] > max_precond_dim and full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'identity'
 
                
        if state['projection_type'] == 'only_right':
            state['GG'] = []
            state['GG'].append(None)
            state['GG'].append(torch.zeros(full_rank_grad.shape[1], full_rank_grad.shape[1], device=full_rank_grad.device))
        elif state['projection_type'] == 'only_left':
            state['GG'] = []
            state['GG'].append(torch.zeros(full_rank_grad.shape[0], full_rank_grad.shape[0], device=full_rank_grad.device))
            state['GG'].append(None)
        elif state['projection_type'] != 'identity':
            state['GG'] = []
            state['GG'].append(torch.zeros(full_rank_grad.shape[0], full_rank_grad.shape[0], device=full_rank_grad.device))
            state['GG'].append(torch.zeros(full_rank_grad.shape[1], full_rank_grad.shape[1], device=full_rank_grad.device))
            
        
                
        state['ortho_matrix'] = [None, None]
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
                # print('Both dimensions equal, using "full"') # Can also use "right" see, https://wandb.ai/harvardml/ShampooAdam/workspace?nw=kf1ze4hfbkh
                state['projection_type'] = 'right'
        else:
            state['projection_type'] = projection_type
                
        if state['projection_type'] != 'identity' and full_rank_grad.dim() == 2 and max_precond_dim >= 0:
            if full_rank_grad.shape[0] > max_precond_dim:
                state['projection_type'] = 'only_right'
            if full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'only_left'
            if full_rank_grad.shape[0] > max_precond_dim and full_rank_grad.shape[1] > max_precond_dim:
                state['projection_type'] = 'identity'        
        
    def project1(self, full_rank_grad, state):
        if state['projection_type'] == 'identity':
            return full_rank_grad
        elif state['projection_type'] in ['only_left', 'left']:
            return torch.matmul(state['ortho_matrix'][0].t(), full_rank_grad)
        elif state['projection_type'] in ['only_right', 'right']:
            return torch.matmul(full_rank_grad, state['ortho_matrix'][1].t())
        else:
            print(state['projection_type'])
            raise ValueError('projection_type should be identity, only_left, only_right, left or right')
    
    def project2(self, full_rank_grad, state):
        if state['projection_type'] in ['identity', 'only_left', 'only_right']:
            return full_rank_grad
        elif state['projection_type'] == 'left':
            return torch.matmul(full_rank_grad, state['ortho_matrix'][1].t())
        elif state['projection_type'] == 'right':
            return torch.matmul(state['ortho_matrix'][0].t(), full_rank_grad)
        else:
            print(state['projection_type'])
            raise ValueError('projection_type should be identity, only_left, only_right, left or right')
        

        
    def update_preconditioner1(self, full_rank_grad, state, iter):

        if state['projection_type'] in ['only_right', 'right']:
            state['GG'][1].lerp_(full_rank_grad.T @ full_rank_grad, 1-state['shampoo_beta'])
            if state['ortho_matrix'][1] is None:
                state['ortho_matrix'][1] = self.get_orthogonal_matrix(state['GG'][1], type='right')
            if iter > 0 and iter % state['precondition_frequency'] == 0:
                state['ortho_matrix'][1] = self.get_orthogonal_matrix_QR(state, type='right', num=1)
        elif state['projection_type'] in ['only_left', 'left']:
            state['GG'][0].lerp_(full_rank_grad @ full_rank_grad.T, 1-state['shampoo_beta'])
            if state['ortho_matrix'][0] is None:
                state['ortho_matrix'][0] = self.get_orthogonal_matrix(state['GG'][0], type='left')
            if iter > 0 and iter % state['precondition_frequency'] == 0:
                state['ortho_matrix'][0] = self.get_orthogonal_matrix_QR(state, type='left', num=1)
                
    def update_preconditioner2(self, full_rank_grad, state, iter):
        
        if state['projection_type'] == 'left':
            state['GG'][1].lerp_(full_rank_grad.T @ full_rank_grad, 1-state['shampoo_beta'])
            if state['ortho_matrix'][1] is None:
                state['ortho_matrix'][1] = self.get_orthogonal_matrix(state['GG'][1], type='right')
            if iter > 0 and iter % state['precondition_frequency'] == state['precondition_frequency']//2:
                state['ortho_matrix'][1] = self.get_orthogonal_matrix_QR(state, type='right', num=2)
        elif state['projection_type'] == 'right':
            state['GG'][0].lerp_(full_rank_grad @ full_rank_grad.T, 1-state['shampoo_beta'])
            if state['ortho_matrix'][0] is None:
                state['ortho_matrix'][0] = self.get_orthogonal_matrix(state['GG'][0], type='left')
            if iter > 0 and iter % state['precondition_frequency'] == state['precondition_frequency']//2:
                state['ortho_matrix'][0] = self.get_orthogonal_matrix_QR(state, type='left', num=2)

                

    def project_back(self, low_rank_grad, state):
        
        
        if state['projection_type'] == 'identity':
            return low_rank_grad
        elif state['projection_type'] == 'only_right':
            full_rank_grad = torch.matmul(low_rank_grad, state['ortho_matrix'][1])
        elif state['projection_type'] == 'only_left':
            full_rank_grad = torch.matmul(state['ortho_matrix'][0], low_rank_grad)
        else:
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
            s, Q = torch.linalg.eigh(matrix.double())
            Q = torch.flip(Q.float(), [1])
            B = Q.T

            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            s, Q = torch.linalg.eigh(matrix.double())
            Q = torch.flip(Q.float(), [1])
            A = Q
            
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            s, Q = torch.linalg.eigh(matrix[0].double())
            Q = torch.flip(Q.float(), [1])
            A = Q
            s, Q = torch.linalg.eigh(matrix[1].double())
            Q = torch.flip(Q.float(), [1])
            B = Q.T
            
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')
        
        
    # svd decomposition
    def get_orthogonal_matrix_QR(self, state, type, full=False, num=1):
        module_params = state['GG']
        orth = state['ortho_matrix']
        if type == 'left':
            module_params = state['GG'][0]
            orth = state['ortho_matrix'][0]
        elif type == 'right':
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
            
            state['exp_avg_sq'][num] = state['exp_avg_sq'][num].T[sort_idx].T
            
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
            
            state['exp_avg_sq'][num] = state['exp_avg_sq'][num][sort_idx]
            
            orth_matrix = orth_matrix.T[sort_idx].T
            
            power_iter = (orth_matrix.T @ matrix).T
            Q, _ = torch.linalg.qr(power_iter)
            

            A = Q
            
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        else:
            raise ValueError('type should be left or right')  