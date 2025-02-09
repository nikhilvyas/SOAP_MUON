import torch
import math

from einops import einsum

# matmul is only upper tringualar should be faster.
# remove unecessary matmuls.
# QR could be done using randomized SVD like approach.
@torch.compile
@torch.no_grad()
def optimize_Q(Q_it, M2, inner_loops=10, ns_steps=5, outer_loops=1, clip=False, clip2=False):
    """
    Function to perform the full optimization step, including inner optimization
    and SVD to update Q_it.
    
    Args:
    - Q_it: Current matrix (torch.Tensor)
    - M2: Target matrix for optimization (torch.Tensor)
    - inner_loops: Number of inner iterations (int)
    
    Returns:
    - Updated Q_it (torch.Tensor)
    """

    # r = 100
    
    # Q_f = torch.zeros_like(Q_it)
    
    # Q_it2 = M2 @ Q_it[:, :r]
    
    # Q_f[:, :r] = torch.linalg.qr(Q_it2)[0]
    
    # Q_it2 = Q_it[:, r:] # - Q_f[:, :r] @ (Q_f[:, :r].T @ Q_it[:, r:])
    # Q_it2 = M2 @ Q_it2
    # Q_it2 = Q_it2 - Q_f[:, :r] @ (Q_f[:, :r].T @ Q_it2)
    
    # Q_f[:, r:] = torch.linalg.qr(Q_it2)[0]
    
    
    
    # return Q_f

    
    
    for i in range(outer_loops):
        
        torch.set_float32_matmul_precision("highest")
        
        
        
        r = min(max(100, Q_it.shape[1]//10), Q_it.shape[1])
        
        Z2 = M2 @ Q_it[:, :r] # 1 (small)
        
        

        
        if r > 0:
            if r == Q_it.shape[1]:
                Z2, _ = torch.linalg.qr(Z2)
                return Z2
            else:
                Z2, _ = torch.linalg.qr(Z2[:, :r])
                # print(torch.max(torch.norm(Z2, dim=0)), torch.min(torch.norm(Z2, dim=0)))
        
        Q_it[:, :r] = Z2
        
        if r > 0:
            
            Z = Q_it[:, r:]
            
            Z -= Z2[:, :r] @ (Z2[:, :r].T @ Z) # 2, 3 (small)
            
            # Z /= (1e-30 + torch.norm(Z))
            
            Z = M2 @ Z # 3 (tri2)
            
            # Z /= (1e-30 + torch.norm(Z))
            
            Z -= Z2[:, :r] @ (Z2[:, :r].T @ Z) # 4, 5 (small)
            
            Z /= (1e-30 + torch.norm(Z))
            
            # print('a0', torch.norm(Z[:, 0]))
            # print('b0', torch.norm(Z[:, 1]))
            # print('z0', torch.norm(Z[:, -1]))
            
            
            
            
        
        # Z = Z.bfloat16()
        
        # print('max2.5', torch.max(Z))
        
        # print('Z', Z)
        
        torch.set_float32_matmul_precision("high")
        
        # mts = torch.zeros(inner_loops, Z.shape[0], Z.shape[1], device=Z.device, dtype=Z.dtype)
        # print('c', Z.shape)
        # Inner optimization
        for j in range(inner_loops):
            Z_norm = Z / (1e-30 + torch.norm(Z, dim=0, keepdim=True))
            dots = Z_norm.T @ Z_norm # 7 (tri)
            
            # Some kind of preconditioning, can probably be improved
            # Power 1.5 works even better but just using 1 for simplicity
            # till we figure out something principled
            # dots /= (1e-30 + torch.norm(dots, dim=1, keepdim=True)**1.0)
            
            dots = torch.triu(dots, diagonal=1)
            mt = Z_norm @ dots # 8 (tri2)
            
            
            # if j > 0:
            #     mt -= torch.sum(mts[:j]*(einsum(mts[:j], mt, 'a i j, i j -> a j')[:, None, :]), dim=0)
            
            # Normalize columns of mt to maintain stability
            mt /= (1e-30 + torch.norm(mt, dim=0, keepdim=True))
            
            # mts[j] = mt
            
            # update_mult = einsum(mts[:j+1], Z, 'a i j, i j -> a j')[:, None, :]
            
            # update = mts[:j+1]*update_mult
            # update = torch.sum(update, dim=0)
            
            mt *= torch.sum(mt * Z, dim=0, keepdim=True)
            Z[:, 1+j:] -= mt[:, 1+j:]
            
            # Z[:, 1+j:] -= update[:, 1+j:]
        
            # print(j, 'Z', Z)
        # print('d', Z.shape)
        if clip:
            # norms = torch.norm(Z, dim=0, keepdim=True)
            # rat = torch.max(norms) / (1e-30 + torch.min(norms))
            # if rat > 100:
            #     alpha = 1.0-math.log(100)/math.log(rat.item())
            #     Z /= norms**alpha
            Z /= (1e-30 + torch.norm(Z))
            
            Z /= (1e-30 + torch.norm(Z, dim=0, keepdim=True))**.5
            
            # norms = torch.norm(Z, dim=0, keepdim=True)
            # norms_rat = norms/(1e-30 + torch.max(norms))
            
            # norms_rat = torch.clip(norms_rat, max=.001)*1000.0
            
            
            
            # Z /= norms_rat
        

        # Use SVD to orthogonalize Z and update Q_it
        # print('Z', Z)
        # print('max3', torch.max(Z))
        # print('e', Z.shape)
        # torch.set_float32_matmul_precision("highest")
        # Z -= Z2[:, :r] @ (Z2[:, :r].T @ Z)
        torch.set_float32_matmul_precision("high")
        Z3 = zeropower_via_newtonschulz5(Z[:, 1+inner_loops:], ns_steps)
        Z[:, 1+inner_loops:] = Z3
        
        Z[:, 1+inner_loops:] /= (1e-30 + torch.norm(Z[:, 1+inner_loops:], dim=0, keepdim=True))
        # print('max4', torch.max(Q_it))
        # U, S, Vt = torch.svd(Q_it)
        # print('S_1..10', S[:10])
        # print('S_-10..-1', S[-10:])
        # print('f', Z.shape, Z2.shape, Q_it.shape)
        # Q_it[:, :r] = Z2
        Q_it[:, r:] = Z
    
    if clip2:
        norms = torch.norm(Q_it, dim=0, keepdim=True)
        # print('final', torch.max(norms), torch.min(norms))
        Q_it /= (1e-30 + norms)**.5
        
    torch.set_float32_matmul_precision("highest")
    
    
    

    return Q_it.float()





# We should make this adptive, check if matrix norm is close to identity,
# if so, don't do any more iterations. But not sure if checking norm is efficient.
# @torch.compile
@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=10, eps=2e-1): # Modification of code form https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    torch.set_float32_matmul_precision("high")
    
    sing_val, converged = top_sv(G.clone())
    
    if not converged:
        steps = steps+2
    
    X = G.clone()
    
    # print('rat2', torch.norm(X)/sing_val)
    
    X /= (1e-30 + sing_val*(1+eps)) # ensure top singular value <= 1
    
    # X /= (1e-30 + torch.norm(X))
    # X /= 1000000000.0
    # print(sing_val, X.norm())
    
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    # for _ in range(2):
    #     X = 1.75 * X - 0.75 * X @ X.T @ X
    
    
    if G.size(0) > G.size(1):
        X = X.T
        
        
    
    return X.float()


# Todo: Add Aitken extrapolation from https://services.math.duke.edu/~jtwong/math361-2019/lectures/Lec10eigenvalues.pdf
# Maybe should also use randomized SVD?

# @torch.compile
@torch.no_grad()
def top_sv(G): # Assumes things from above functions, specifically that top columns of G are close to the top singular vectors
    # G = G.bfloat16()
    
    eigvectors = torch.randn(10, G.shape[0], device=G.device, dtype=G.dtype)
    
    eigvectors[:5] = (G[:, :5].T).clone()
    eigvectors /= (1e-30 + torch.norm(eigvectors, dim=1, keepdim=True))
    
    s_vals = torch.zeros(10, device=G.device, dtype=G.dtype)
    
    # print('eigvectors', eigvectors)
    # print('G', G)
    
    torch.set_float32_matmul_precision("high")
    
    ratios = []
    
    for i in range(8):
        if i % 2 == 0:
            eigvectors = eigvectors @ G
            s_vals_new = torch.norm(eigvectors, dim=1, keepdim=True)
            # if i > 0:
                # print(i, torch.max(s_vals_new)/(torch.max(s_vals)+1e-30))
            # print(eigvectors)
            # print(s_vals_new, s_vals)
            ratio = torch.max(s_vals_new) / (1e-30 + torch.max(s_vals))
            ratios.append(ratio)
            s_vals = s_vals_new
            eigvectors /= (1e-30 + s_vals)
            eigvectors = eigvectors.T
        if i % 2 == 1:
            eigvectors = G @ eigvectors
            s_vals_new = torch.norm(eigvectors, dim=0, keepdim=True)
            # if i > 0:
                # print(i, torch.max(s_vals_new)/(torch.max(s_vals)+1e-30))
            ratio = torch.max(s_vals_new) / (1e-30 + torch.max(s_vals))
            ratios.append(ratio)
            # print(s_vals_new, s_vals)
            s_vals = s_vals_new
            eigvectors /= (1e-30 + s_vals)
            eigvectors = eigvectors.T
            
        if len(ratios) >= 3:
            if ratios[-1] < 1.01 and ratios[-1] > 0.99:
                if ratios[-2] < 1.01 and ratios[-2] > 0.99:
                    if ratios[-3] < 1.01 and ratios[-3] > 0.99:
                        return torch.max(s_vals), True
    
    rat = ratios[-1].item()
    # print('rat', rat)
    if rat > 1.01 or rat < 0.99:
        return torch.norm(G), False
    else:
        return torch.max(s_vals), True
        
