# koun_gr_v1_4_2.py
# ============================================================================
# KOUN-GR v1.4.2 (The Protocol Freeze)
# ----------------------------------------------------------------------------
# ARCHITECTURE: Pseudo-Transient Newton-Krylov with Normalized Structural Sentinel
# STATUS: GOLD MASTER CANDIDATE
#
# PROTOCOL DEFINITIONS (Scale-Invariant):
#   1. ALIGNMENT: Uses Cosine Similarity (c).
#      - Type-1 (Uphill): c > 0
#      - Type-2 (Vanishing): |c| <= 1e-14
#      - Descent: c < -1e-14
#   2. SNIPER STEP: d_psi = - alpha * dt * (grad / |grad|) * |res|
#      - Explicitly links dt, residual scale, and gradient direction.
#   3. ACCEPTANCE: Merit_new <= Merit_old * (1 + 1e-12) + 1e-15
#      - Dual tolerance for float noise.
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
def get_stretched_grid_1d(N, L_max, gamma):
    xi = jnp.linspace(-1.0, 1.0, N)
    if abs(gamma) < 1e-9:
        x = xi * L_max
    else:
        x = L_max * jnp.sinh(gamma * xi) / jnp.sinh(gamma)
    return x

def compute_metrics_1d(x):
    N = x.shape[0]
    dx_local = jnp.zeros(N)
    dx_local = dx_local.at[1:-1].set(0.5 * (x[2:] - x[:-2]))
    dx_local = dx_local.at[0].set(x[1] - x[0])
    dx_local = dx_local.at[-1].set(x[-1] - x[-2])
    return dx_local

def compute_geometry_and_weights(N, L_max, gamma):
    x_1d = get_stretched_grid_1d(N, L_max, gamma)
    dx_1d = compute_metrics_1d(x_1d)
    X, Y, Z = jnp.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
    geom = {"x_1d": x_1d, "dx_1d": dx_1d, "X": X, "Y": Y, "Z": Z}
    return geom, None

# ============================================================================
# 3. Physics Kernel
# ============================================================================

@jit
def laplacian_3d_stretched(psi, x_1d):
    N = x_1d.shape[0]
    h = x_1d[1:] - x_1d[:-1]

    def d2_dx2(f_line):
        df = f_line[1:] - f_line[:-1]
        grad_mid = df / h
        d2f = (grad_mid[1:] - grad_mid[:-1]) * 2.0 / (h[1:] + h[:-1])
        return jnp.concatenate([jnp.array([0.]), d2f, jnp.array([0.])])

    L_xx = jax.vmap(jax.vmap(d2_dx2, in_axes=0), in_axes=0)(psi)

    psi_y = psi.transpose((0, 2, 1))
    L_yy = jax.vmap(jax.vmap(d2_dx2, in_axes=0), in_axes=0)(psi_y).transpose((0, 2, 1))

    psi_z = psi.transpose((2, 1, 0))
    L_zz = jax.vmap(jax.vmap(d2_dx2, in_axes=0), in_axes=0)(psi_z).transpose((2, 1, 0))

    return L_xx + L_yy + L_zz

@jit
def residual_fn(psi, rho, x_1d):
    lap = laplacian_3d_stretched(psi, x_1d)
    source = 2.0 * jnp.pi * rho * (psi ** 5)
    res = lap - source
    mask = jnp.ones_like(psi)
    mask = mask.at[0,:,:].set(0).at[-1,:,:].set(0)
    mask = mask.at[:,0,:].set(0).at[:,-1,:].set(0)
    mask = mask.at[:,:,0].set(0).at[:,:,-1].set(0)
    return res * mask

@jit
def merit_loss(psi, rho, x_1d):
    F = residual_fn(psi, rho, x_1d)
    return 0.5 * jnp.sum(F**2)

compute_merit_grad = jit(jax.grad(merit_loss, argnums=0))

# ============================================================================
# 4. Solver Core (v1.4.2 Protocol Freeze)
# ============================================================================

def solve_linear_system(J_func, F, dt, guess=None):
    def matvec(v):
        j_v = J_func(v)
        return (1.0/dt) * v - j_v
    dx, info = gmres(matvec, F, x0=guess, tol=1e-5, maxiter=50, restart=20)
    return dx, info

def ptc_solve(psi_init, rho, x_1d, tol=1e-9, max_iter=60):
    psi = psi_init
    dt = 1e-1
    dt_min, dt_max = 1e-4, 1e5
    sniper_mode = False

    # PROTOCOL: Logging Headers
    print(f"{'Iter':<5} | {'Residual':<12} | {'Merit':<12} | {'dt':<9} | {'CosAlign':<9} | {'Status':<25}")
    print("-" * 110)

    for k in range(max_iter):
        res = residual_fn(psi, rho, x_1d)
        res_norm = jnp.linalg.norm(res)
        merit_val = 0.5 * res_norm**2

        if res_norm < tol:
            print(f"{k:<5} | {res_norm:<12.4e} | {merit_val:<12.4e} | {dt:<9.2e} | {'N/A':<9} | [CONVERGED]")
            return psi, "CONVERGED"

        merit_grad = compute_merit_grad(psi, rho, x_1d)
        grad_norm = jnp.linalg.norm(merit_grad)

        # --- NEWTON CANDIDATE ---
        def J_func(v):
            return jax.jvp(partial(residual_fn, rho=rho, x_1d=x_1d), (psi,), (v,))[1]

        dx_newton, info = solve_linear_system(J_func, res, dt)

        # --- SENTINEL CHECK (Normalized Cosine) ---
        norm_dx = jnp.linalg.norm(dx_newton)
        denominator = grad_norm * norm_dx + 1e-16
        cosine_align = jnp.sum(merit_grad * dx_newton) / denominator

        # PROTOCOL: Type-1 (Uphill) Check
        # If cosine > 0, we are strictly going uphill against the gradient.
        # If cosine ~ 0, we are orthogonal (signal loss), but not necessarily wrong if at limit.
        is_type1_uphill = (cosine_align > 1e-14)

        if is_type1_uphill or sniper_mode:
            # --- SNIPER PROTOCOL ---
            mode_tag = "[SNIPER]"
            if is_type1_uphill: mode_tag += "[TYPE-1]"

            if grad_norm < 1e-16:
                 print(f"{k:<5} | {res_norm:<12.4e} | ... | [FAILURE] Grad Vanished")
                 return psi, "FAILED_LOCAL_MIN"

            # PROTOCOL: Standardized Sniper Step
            # d_psi = - alpha * dt * (grad / |grad|) * |res|
            # Alpha = 0.5 (Conservative descent)
            # This ensures units are consistent: [psi] ~ [dt] * [res] (roughly from evolution eq)
            # Normalizing gradient ensures we just take direction.
            # Scaling by |res| ensures we shrink as we get closer.

            alpha_sniper = 0.5
            p_dir = - merit_grad / (grad_norm + 1e-16)
            dx_final = p_dir * res_norm * dt * alpha_sniper

            current_step_type = "ADJOINT"
        else:
            # --- NEWTON PROTOCOL ---
            mode_tag = "[NEWTON]"
            dx_final = dx_newton
            current_step_type = "NEWTON"

        # --- EXECUTE & ACCEPTANCE (Dual Tolerance) ---
        psi_new = psi + dx_final
        res_new = residual_fn(psi_new, rho, x_1d)
        res_norm_new = jnp.linalg.norm(res_new)
        merit_new = 0.5 * res_norm_new**2

        # PROTOCOL: Dual Tolerance Acceptance
        # Accept if: new <= old * (1 + rel) + abs
        tol_rel = 1e-12
        tol_abs = 1e-15
        threshold = merit_val * (1.0 + tol_rel) + tol_abs

        is_accepted = (merit_new < threshold)

        if is_accepted:
            # ACC
            psi = psi_new

            if current_step_type == "NEWTON":
                dt = min(dt * 1.5, dt_max)
                sniper_mode = False
            else:
                # Sniper success: cautiously grow dt
                dt = min(dt * 1.1, dt_max)
                sniper_mode = False
            status_msg = f"ACC {mode_tag}"
        else:
            # REJ
            status_msg = f"REJ {mode_tag} -> RETRY"
            dt = max(dt * 0.5, dt_min)
            if current_step_type == "NEWTON":
                sniper_mode = True

        print(f"{k:<5} | {res_norm:<12.4e} | {merit_val:<12.4e} | {dt:<9.2e} | {cosine_align:<9.2e} | {status_msg}")

    return psi, "HIT_CAP_WITH_BEST"

# ============================================================================
# Main Execution (Regression Suite Candidate)
# ============================================================================

def build_problem(N=32, L_max=50.0, gamma=1.0):
    geom, _ = compute_geometry_and_weights(N, L_max, gamma)
    x_1d = geom['x_1d']
    X, Y, Z = geom['X'], geom['Y'], geom['Z']
    r2 = X**2 + Y**2 + Z**2
    sigma = 4.0
    rho = jnp.exp(-r2 / (2 * sigma**2))
    rho = rho / jnp.max(rho) * 0.1
    psi = jnp.ones((N, N, N))
    return psi, rho, x_1d

if __name__ == "__main__":
    print("[KOUN-GR v1.4.2] The Protocol Freeze")
    print("Standard: Cosine Sentinel | Dual Tolerance | Linked Sniper Step")

    N = 32
    psi_init, rho, x_1d = build_problem(N=N)

    print("JIT Compiling...")
    _ = residual_fn(psi_init, rho, x_1d)
    _ = compute_merit_grad(psi_init, rho, x_1d)
    print("Starting Solver...")

    psi_final, status = ptc_solve(psi_init, rho, x_1d)

    print(f"\nFinal Solver State: {status}")