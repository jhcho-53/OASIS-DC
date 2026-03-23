from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, cg

def poisson_gpu(E_m: torch.Tensor, DL: torch.Tensor, ML: torch.Tensor,
                lam: float = 800.0, iters: int = 300, hard: bool = False,
                dmax: float = 10.0) -> torch.Tensor:
    def neigh_sum(X):
        up    = F.pad(X, (0,0,1,0), mode='replicate')[:,:,:-1,:]
        down  = F.pad(X, (0,0,0,1), mode='replicate')[:,:,1:,:]
        left  = F.pad(X, (1,0,0,0), mode='replicate')[:,:,:,:-1]
        right = F.pad(X, (0,1,0,0), mode='replicate')[:,:,:,1:]
        return up + down + left + right

    lap_E = neigh_sum(E_m) - 4.0 * E_m
    b = (-lap_E) + lam * ML * DL
    denom = 4.0 + lam * ML

    D = E_m.clone()
    for _ in range(int(iters)):
        sumN = neigh_sum(D)
        D_new = (sumN + b) / denom
        if hard:
            D_new = ML * DL + (1.0 - ML) * D_new
        D = D_new

    return D.clamp_(0.0, float(dmax))

def build_Laplacian(H: int, W: int) -> sp.csr_matrix:
    N = H * W
    main = np.full(N, 4, np.float32)
    off  = np.full(N, -1, np.float32)
    A = sp.diags([main, off, off, off, off], [0, -1, +1, -W, +W], format='csr')
    for i in range(H):
        L = i * W
        R = i * W + (W - 1)
        if L - 1 >= 0: 
            A[L, L-1] = 0
            A[L-1, L] = 0
        if R + 1 < N: 
            A[R, R+1] = 0
            A[R+1, R] = 0
    return A

def grid_laplacian(H, W, dtype=np.float64):
    N = H * W
    ids = np.arange(N)
    has_left  = (ids % W) != 0
    has_right = (ids % W) != (W - 1)
    has_up    = ids >= W
    has_down  = ids < (N - W)

    deg = has_left.astype(np.float64) + has_right.astype(np.float64) + \
          has_up.astype(np.float64) + has_down.astype(np.float64)

    h = np.ones(N - 1, dtype=np.float64)
    h[np.arange(N - 1) % W == (W - 1)] = 0.0
    off_pm1 = -h
    v = -np.ones(N - W, dtype=np.float64)

    L = sp.diags(
        diagonals=[deg, off_pm1, off_pm1, v, v],
        offsets=[0, 1, -1, W, -W],
        shape=(N, N),
        format='csr',
        dtype=dtype
    )
    return L

def poisson_complete(sparse, mask, lam=10.0, maxiter=1000, tol=1e-5):
    H, W = sparse.shape
    if mask.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    L = grid_laplacian(H, W, dtype=np.float64)
    M_diag = mask.reshape(-1).astype(np.float64)
    d = sparse.reshape(-1).astype(np.float64)

    A = L.tolil(copy=True)
    A.setdiag(A.diagonal() + lam * M_diag)
    A = A.tocsr()

    b = lam * (M_diag * d)
    x0 = d.copy()

    try:
        x, info = cg(A, b, x0=x0, maxiter=maxiter, rtol=tol, atol=0.0)
        if info != 0:
            x = spsolve(A, b)
    except:
        x = spsolve(A, b)
    
    dense = x.reshape(H, W).astype(np.float32)
    return dense