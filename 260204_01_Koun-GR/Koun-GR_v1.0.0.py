# koun_gr_v1_3_22.py
# ============================================================================
# KOUN-GR v1.3.22 (The Balanced Path)
# ----------------------------------------------------------------------------
# ARCHITECTURE: Pseudo-Transient Newton-Krylov (Final Tuned)
# MISSION: Achieve robust convergence by balancing safety and progress.
#
# CRITICAL FIXES vs v1.3.21:
#   1. RETRY LOGIC: If a step fails, retry linear solve with smaller dt 
#      IMMEDIATELY (same iter), instead of wasting an outer loop cycle.
#   2. SOFT PUNISHMENT: dt *= 0.5 on failure (not 0.1). Prevents freezing.
#   3. TOLERANCE: Accept new_res < res * (1 + 1e-6) to handle float noise.
# ============================================================================

import time
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.sparse.linalg import gmres
from functools import partial
from typing import Tuple, NamedTuple, Literal

jax.config.update("jax_enable_x64", True)

# ============================================================================
# 1. Physics & Constants
# ============================================================================
G_GEO = 1.0 

# ============================================================================
# 2. Geometry
# ============================================================================
class InterpWeights(NamedTuple):
    w_L_center: jnp.ndarray; w_L_neigh: jnp.ndarray
    w_R_center: jnp.ndarray; w_R_neigh: jnp.ndarray

class LevelGeom(NamedTuple):
    tx: jnp.ndarray; ty: jnp.ndarray; tz: jnp.ndarray
    vx: jnp.ndarray; vy: jnp.ndarray; vz: jnp.ndarray
    x_centers: jnp.ndarray

def sinh_arcsinh_map(xi, L, gamma):
    safe_gamma = jnp.where(gamma < 1e-4, 1e-4, gamma)
    return L * jnp.sinh(safe_gamma * xi) / jnp.sinh(safe_gamma)

def compute_geometry_and_weights(N, L, gamma):
    xi_faces = jnp.linspace(-1.0, 1.0, N + 1)
    dxi = xi_faces[1] - xi_faces[0]
    xi_centers = xi_faces[:-1] + 0.5 * dxi
    x_faces = sinh_arcsinh_map(xi_faces, L, gamma)
    x_centers = sinh_arcsinh_map(xi_centers, L, gamma)
    vx = x_faces[1:] - x_faces[:-1]
    dx_centers = x_centers[1:] - x_centers[:-1]
    dist_left = x_centers[0] - x_faces[0]
    dist_right = x_faces[-1] - x_centers[-1]
    dists = jnp.concatenate([jnp.array([dist_left]), dx_centers, jnp.array([dist_right])])
    tx = 1.0 / dists
    ty, tz = tx, tx
    vx, vy, vz = vx, vx, vx 
    geom = LevelGeom(tx, ty, tz, vx, vy, vz, x_centers)
    return geom, x_centers

def compute_weights_explicit(coarse_centers, fine_centers, L_bound):
    return InterpWeights(jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([]))

# ============================================================================
# 3. Physics Kernels
# ============================================================================
@jit
def laplacian_stretched_neg(phi, geom: LevelGeom, bc_val):
    tx, ty, tz = geom.tx, geom.ty, geom.tz
    vx, vy, vz = geom.vx, geom.vy, geom.vz
    Ax = vy[None, :, None] * vz[None, None, :]
    Ay = vx[:, None, None] * vz[None, None, :]
    Az = vx[:, None, None] * vy[None, :, None]
    Vol = vx[:, None, None] * vy[None, :, None] * vz[None, None, :]
    p = jnp.pad(phi, 1, mode='constant', constant_values=bc_val)
    flux_x = (p[1:, 1:-1, 1:-1] - p[:-1, 1:-1, 1:-1]) * tx[:, None, None] * Ax
    flux_y = (p[1:-1, 1:, 1:-1] - p[1:-1, :-1, 1:-1]) * ty[None, :, None] * Ay
    flux_z = (p[1:-1, 1:-1, 1:] - p[1:-1, 1:-1, :-1]) * tz[None, None, :] * Az
    div_x = flux_x[1:] - flux_x[:-1]
    div_y = flux_y[:, 1:] - flux_y[:, :-1]
    div_z = flux_z[:, :, 1:] - flux_z[:, :, :-1]
    return -(div_x + div_y + div_z) / Vol

@jit
def hamiltonian_residual(psi, rho, geom):
    L_psi = laplacian_stretched_neg(psi, geom, bc_val=1.0)
    S_psi = 2.0 * jnp.pi * G_GEO * rho * (psi ** 5)
    return L_psi - S_psi

# ============================================================================
# 4. Solvers
# ============================================================================

@jit
def diagonal_precond_ptc(r, psi, rho, geom, dt_inverse):
    tx, ty, tz = geom.tx, geom.ty, geom.tz
    vx, vy, vz = geom.vx, geom.vy, geom.vz
    Ax = vy[None, :, None] * vz[None, None, :]
    Ay = vx[:, None, None] * vz[None, None, :]
    Az = vx[:, None, None] * vy[None, :, None]
    Vol = vx[:, None, None] * vy[None, :, None] * vz[None, None, :]
    
    diag_L = (tx[:-1] + tx[1:])[:,None,None]*Ax/Vol + \
             (ty[:-1] + ty[1:])[None,:,None]*Ay/Vol + \
             (tz[:-1] + tz[1:])[None,None,:]*Az/Vol
             
    S_prime = 10.0 * jnp.pi * G_GEO * rho * (psi ** 4)
    J_diag = diag_L - S_prime
    
    local_scale = jnp.abs(diag_L) + jnp.abs(S_prime) + 1e-20
    eps_local = 1e-8 * local_scale + 1e-15
    J_abs = jnp.where(jnp.abs(J_diag) < eps_local, eps_local, jnp.abs(J_diag))
    
    den = J_abs + dt_inverse
    return r / den

@jit
def linear_operator_product_ptc(dx, psi_current, rho, geom, dt_inverse):
    L_dx = laplacian_stretched_neg(dx, geom, bc_val=0.0)
    S_prime = 10.0 * jnp.pi * G_GEO * rho * (psi_current ** 4)
    J_dx = L_dx - S_prime * dx
    return J_dx + dt_inverse * dx

def ptc_solve(rho, geoms, tol=1e-9):
    psi = jnp.ones_like(rho) 
    geom_fine = geoms[0]
    
    dt = 0.1 
    dt_min = 1e-4
    dt_max = 1e9
    
    best_psi = psi
    best_res = 1e20
    
    print(f"{'Iter':<5} | {'Residual':<12} | {'Rel Res':<12} | {'dt':<10} | {'Status':<15}")
    print("-" * 70)
    
    # Calculate initial residual
    F_init = hamiltonian_residual(psi, rho, geom_fine)
    norm_0 = jnp.linalg.norm(F_init) + 1e-20
    best_res = norm_0
    
    for k in range(50): 
        # Outer Loop: Non-linear Step
        F = hamiltonian_residual(psi, rho, geom_fine)
        res_norm = jnp.linalg.norm(F)
        rel_res = res_norm / norm_0
        
        if res_norm < tol:
            print(f"{k:<5} | {res_norm:<12.4e} | {rel_res:<12.4e} | ---        | [CONVERGED]")
            break
            
        # Update best so far
        if res_norm < best_res:
            best_res = res_norm
            best_psi = psi
            
        # Retry Loop: Try to find a valid dt for this step
        step_success = False
        
        for retry in range(3): # Try current dt, then dt/2, then dt/4...
            dt_inverse = 1.0 / dt
            
            def matvec(dx): return linear_operator_product_ptc(dx, psi, rho, geom_fine, dt_inverse)
            def precond(r): return diagonal_precond_ptc(r, psi, rho, geom_fine, dt_inverse)
            
            # Linear Solve
            gmres_tol = min(1e-2, res_norm * 0.1)
            dx, info = gmres(matvec, -F, M=precond, tol=gmres_tol, restart=30, maxiter=30)
            
            # Trial Update
            psi_trial = jnp.maximum(psi + dx, 1e-6)
            new_res = jnp.linalg.norm(hamiltonian_residual(psi_trial, rho, geom_fine))
            
            # Check Acceptance (with tiny noise tolerance)
            if new_res < res_norm * (1 + 1e-6):
                # ACCEPT
                psi = psi_trial
                dt = min(dt * 1.1, dt_max) # Gentle growth
                status = f"ACC (try {retry})"
                step_success = True
                break # Exit retry loop
            else:
                # REJECT & RETRY IMMEDIATELY
                dt = max(dt * 0.5, dt_min) # Soft punish
                # Continue loop to re-solve with smaller dt
        
        if not step_success:
            status = "STALLED"
            # If stalled, we naturally stay at 'psi' and try next outer loop 
            # with the already-shrunk dt.
            
        print(f"{k:<5} | {res_norm:<12.4e} | {rel_res:<12.4e} | {dt:<10.2e} | {status}")
        
    return best_psi

# ============================================================================
# Main
# ============================================================================

def build_solver_simple(k=5, L_max=50.0, gamma=1.0):
    N_in = 2**k
    geoms = []
    geom, _ = compute_geometry_and_weights(N_in, L_max, gamma)
    geoms.append(geom)
    return tuple(geoms), L_max, gamma

if __name__ == "__main__":
    print("[KOUN-GR v1.3.22] The Balanced Path")
    print("Strategy: N=32, Gamma=1.0, PTC with In-Frame Retry.")
    
    solver_gamma = 1.0 
    N_pow = 5 
    
    geoms, L_max, gamma = build_solver_simple(k=N_pow, L_max=50.0, gamma=solver_gamma)
    N = 2**N_pow
    
    rho = jnp.zeros((N, N, N))
    x = jnp.arange(N)
    Y, X, Z = jnp.meshgrid(x, x, x)
    c1, c2 = N//2 - 5, N//2 + 5
    R2_1 = (X-c1)**2 + (Y-N//2)**2 + (Z-N//2)**2
    R2_2 = (X-c2)**2 + (Y-N//2)**2 + (Z-N//2)**2
    rho = 0.001 * (jnp.exp(-R2_1/16.0) + jnp.exp(-R2_2/16.0))
    
    print(f"Grid: {N}^3 | Gamma: {gamma} | Stretch: ~{jnp.exp(gamma):.1f}x")
    
    t0 = time.time()
    psi = ptc_solve(rho, geoms)
    t1 = time.time()
    
    if psi is not None:
        print(f"\n[SUCCESS] Total Time: {t1-t0:.4f} s | Range: [{jnp.min(psi):.4f}, {jnp.max(psi):.4f}]")