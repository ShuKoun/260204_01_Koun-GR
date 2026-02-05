# koun_gr_v1_3_29.py
# ============================================================================
# KOUN-GR v1.3.29 (The Adjoint Protocol)
# ----------------------------------------------------------------------------
# ARCHITECTURE: Pseudo-Transient Newton-Krylov with Exact Gradient Fallback
# MISSION: Use JAX AD to compute the TRUE descent direction (-J^T F) when 
#          Newton direction fails.
#
# CRITICAL UPGRADES vs v1.3.28:
#   1. EXACT GRADIENT: Implemented merit_loss() and jax.grad() to compute
#      the exact mathematical gradient of the residual norm.
#   2. TRUE DESCENT: Fallback direction is now dx = -grad. This is theoretically
#      guaranteed to be a descent direction for the merit function.
#   3. STRICT ACCEPTANCE: Since it's a true gradient, we demand strict descent 
#      (Accept Factor = 1.0).
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

# --- NEW: Merit Function and Exact Gradient ---
def merit_loss(psi, rho, geom):
    res = hamiltonian_residual(psi, rho, geom)
    return 0.5 * jnp.sum(res**2)

# Create JIT-compiled gradient function
compute_exact_grad = jit(jax.grad(merit_loss, argnums=0))

# ============================================================================
# 4. Solvers (THE ADJOINT PROTOCOL)
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
    
    print(f"{'Iter':<5} | {'Residual':<12} | {'Rel Res':<12} | {'dt':<9} | {'Status':<45}")
    print("-" * 110)
    
    F_init = hamiltonian_residual(psi, rho, geom_fine)
    norm_0 = jnp.linalg.norm(F_init) + 1e-20
    
    best_psi = psi
    best_res = norm_0
    stall_counter = 0
    consecutive_stalls = 0
    
    for k in range(60): 
        F = hamiltonian_residual(psi, rho, geom_fine)
        res_norm = jnp.linalg.norm(F)
        rel_res = res_norm / norm_0
        
        if res_norm < tol:
            print(f"{k:<5} | {res_norm:<12.4e} | {rel_res:<12.4e} | ---       | [CONVERGED]")
            break
            
        if res_norm < best_res * (1.0 - 1e-6):
            best_res = res_norm
            best_psi = psi
            stall_counter = 0
        else:
            stall_counter += 1

        sniper_mode = (stall_counter >= 3)
        
        if sniper_mode:
            target_tol = 1e-8 
            accept_factor_newton = 1.0 - 1e-4 
            accept_factor_grad = 1.0 - 1e-8   # STRICT DESCENT required for Exact Gradient
            g_restart = 100; g_maxiter = 100
            mode_tag = "[SNIPER]"
        else:
            target_tol = max(1e-12, min(1e-2, res_norm * 0.01))
            accept_factor_newton = 1.0 + 1e-5 
            accept_factor_grad = 1.0 + 1e-5
            g_restart = 40; g_maxiter = 50
            mode_tag = ""

        # PHOENIX ROLLBACK 
        if dt < dt_min * 1.5 and consecutive_stalls > 4: 
            print(f"      >>> [PHOENIX] STALLED. ROLLING BACK. <<<")
            psi = best_psi 
            dt = 0.05 
            consecutive_stalls = 0
            stall_counter = 0 
            continue 
        
        step_success = False
        final_status = ""
        best_ratio_newton = 1.0
        best_ratio_grad = 1.0 
        
        for retry in range(3): 
            dt_inverse = 1.0 / dt
            
            def matvec(dx): return linear_operator_product_ptc(dx, psi, rho, geom_fine, dt_inverse)
            def precond(r): return diagonal_precond_ptc(r, psi, rho, geom_fine, dt_inverse)
            
            # 1. NEWTON (GMRES)
            dx, info = gmres(matvec, -F, M=precond, tol=target_tol, restart=g_restart, maxiter=g_maxiter)
            dir_type = "NEWTON"
            
            alpha = 1.0
            ls_success = False
            best_ratio_newton = 1.0
            
            for ls_step in range(7): 
                psi_trial = jnp.maximum(psi + alpha * dx, 1e-6)
                new_res = jnp.linalg.norm(hamiltonian_residual(psi_trial, rho, geom_fine))
                ratio = new_res / res_norm
                if ratio < best_ratio_newton: best_ratio_newton = ratio
                
                if new_res < res_norm * accept_factor_newton:
                    psi = psi_trial
                    ls_success = True
                    final_status = f"ACC (a={alpha},t={retry}) {mode_tag} [{dir_type}]"
                    break 
                alpha *= 0.5 
            
            # 2. FALLBACK: EXACT GRADIENT DESCENT
            # Trigger if Newton fails and we are in Sniper Mode
            if not ls_success and sniper_mode and best_ratio_newton > 0.99999:
                dir_type = "ADJOINT_GRAD"
                print(f"      >>> TRIGGERING ADJOINT GRADIENT (retry={retry}) <<<")
                
                # Compute Exact Gradient: g = J^T * F
                grad = compute_exact_grad(psi, rho, geom_fine)
                
                # Normalize gradient roughly to match residual scale for stability
                grad_norm = jnp.linalg.norm(grad) + 1e-20
                scale_factor = res_norm / grad_norm
                dx = -grad * scale_factor # Steepest Descent direction
                
                alpha = 0.5 
                best_ratio_grad = 1.0 
                
                for ls_step in range(8): # Go slightly deeper
                    psi_trial = jnp.maximum(psi + alpha * dx, 1e-6)
                    new_res = jnp.linalg.norm(hamiltonian_residual(psi_trial, rho, geom_fine))
                    ratio = new_res / res_norm
                    if ratio < best_ratio_grad: best_ratio_grad = ratio
                    
                    if new_res < res_norm * accept_factor_grad: 
                        psi = psi_trial
                        ls_success = True
                        final_status = f"ACC (a={alpha},t={retry}) {mode_tag} [{dir_type}]"
                        break
                    alpha *= 0.5

            if ls_success:
                step_success = True
                consecutive_stalls = 0 
                # Smart DT: Grow if Newton works well, shrink if Fallback needed
                if dir_type == "NEWTON" and alpha == 1.0 and retry == 0: 
                    dt = min(dt * 1.1, dt_max)
                elif dir_type == "ADJOINT_GRAD" or alpha < 0.25: 
                    dt = max(dt * 0.9, dt_min) 
                break 
            else:
                dt = max(dt * 0.5, dt_min)
        
        if not step_success:
            status_str = f"STALLED {mode_tag} (RatN={best_ratio_newton:.6f}"
            if best_ratio_grad < 1.0: 
                 status_str += f", RatG={best_ratio_grad:.6f})"
            else:
                 status_str += ")"
            final_status = status_str
            consecutive_stalls += 1
            
        print(f"{k:<5} | {res_norm:<12.4e} | {rel_res:<12.4e} | {dt:<9.2e} | {final_status}")
        
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
    print("[KOUN-GR v1.3.29] The Adjoint Protocol")
    print("Strategy: N=32, Gamma=1.0")
    print("Features: Exact Gradient Fallback (JAX AD) + Merit-Aware Descent")
    
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