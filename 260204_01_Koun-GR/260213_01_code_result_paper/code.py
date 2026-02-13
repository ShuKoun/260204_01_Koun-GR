# koun_a1_evidence_harness_v7.py
# ============================================================================
# KOUN-A1 EVIDENCE HARNESS v7.0 (The Governance Proof)
# ----------------------------------------------------------------------------
# Objective: Demonstrate that Mode C (Governed PTC) can survive and converge
#            where Mode B (Blind PTC) dies, even if Mode A (Newton) works.
#
# LOGIC UPDATE:
#   1. Mode C is now HYBRID:
#      - If Trusted Type-1 (Cos>0, LinErr<0.1) -> SNIPER (Rescuing)
#      - If Normal (Cos<0) -> NEWTON (Accelerating)
#   2. Increased MAX_ITER to 500 to allow Sniper to exit the trap.
#   3. Success metric: "GOVERNED" if converged or significantly reduced.
# ============================================================================

import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.sparse.linalg import gmres
from functools import partial
from typing import NamedTuple, Literal

jax.config.update("jax_enable_x64", True)

# --- CONFIGURATION ---

class EvidenceProfile(NamedTuple):
    rho_peak: float = 10.0
    sigma: float = 1.5
    gamma: float = 2.0
    init_mode: str = 'dip'
    gmres_tol: float = 1e-6
    gmres_maxiter: int = 100

# --- PHYSICS ---
# (Same as v6.0, omitted for brevity but fully included in execution)
def get_grid(N, L_max, gamma):
    xi = jnp.linspace(-1.0, 1.0, N)
    safe_gamma = jnp.where(jnp.abs(gamma)<1e-9, 1e-9, gamma)
    x = L_max * jnp.sinh(safe_gamma * xi) / jnp.sinh(safe_gamma)
    return x

@jit
def residual_fn(psi, rho, x_1d):
    N = x_1d.shape[0]
    h = x_1d[1:] - x_1d[:-1]
    def d2(f):
        df = f[1:] - f[:-1]; g = df/h
        d2f = (g[1:] - g[:-1]) * 2.0 / (h[1:] + h[:-1])
        return jnp.concatenate([jnp.array([0.]), d2f, jnp.array([0.])])
    L = vmap(vmap(d2,0),0)(psi) + \
        vmap(vmap(d2,0),0)(psi.transpose(0,2,1)).transpose(0,2,1) + \
        vmap(vmap(d2,0),0)(psi.transpose(2,1,0)).transpose(2,1,0)
    res = L - 2.0 * jnp.pi * rho * (psi**5)
    mask = jnp.ones_like(psi).at[0].set(0).at[-1].set(0) \
           .at[:,0].set(0).at[:,-1].set(0).at[:,:,0].set(0).at[:,:,-1].set(0)
    return res * mask

@jit
def merit_loss(psi, rho, x_1d):
    F = residual_fn(psi, rho, x_1d)
    return 0.5 * jnp.sum(F**2)

grad_merit = jit(jax.grad(merit_loss, argnums=0))

# --- PROTOCOL ---

def check_acceptance(merit_new, merit_old):
    return merit_new <= merit_old * (1.0 + 1e-12) + 1e-15

def run_trial_governed(profile: EvidenceProfile, mode: Literal['A', 'B', 'C'], max_iter=500):
    # Setup (Same as v6.0)
    N, L_max = 32, 50.0
    x_1d = get_grid(N, L_max, profile.gamma)
    X, Y, Z = jnp.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    rho = profile.rho_peak * jnp.exp(-r2 / (2 * profile.sigma**2))
    
    if profile.init_mode == 'dip': 
        psi = jnp.maximum(1.0 - 0.9*jnp.exp(-r2/20.0), 0.1)
    else: 
        psi = jnp.ones((N,N,N))
    
    dt = 1e9 if mode == 'A' else 0.1
    dt_min = 1e-5
    
    def jvp_op(v, p): return jax.jvp(partial(residual_fn, rho=rho, x_1d=x_1d), (p,), (v,))[1]
    
    initial_merit = merit_loss(psi, rho, x_1d)
    print(f"Start {mode}: Merit={initial_merit:.2e}")
    
    for k in range(max_iter):
        res = residual_fn(psi, rho, x_1d)
        res_norm = jnp.linalg.norm(res)
        merit = 0.5 * res_norm**2
        
        if res_norm < 1e-8: return (True, res_norm, f"CONVERGED@{k}")
        if merit > initial_merit * 1e5: return (False, res_norm, "DIVERGED")
        
        g = grad_merit(psi, rho, x_1d)
        g_norm = jnp.linalg.norm(g) + 1e-20
        
        # Linear Solve
        def matvec(v):
            op = jvp_op(v, psi)
            if mode != 'A': op += (1.0/dt)*v
            return op
        
        dx, _ = gmres(matvec, -res, tol=profile.gmres_tol, maxiter=profile.gmres_maxiter)
        
        # Sentinel (Only C)
        dx_norm = jnp.linalg.norm(dx) + 1e-20
        cos_align = jnp.sum(g * dx) / (g_norm * dx_norm)
        
        # Trust Check
        lin_res_vec = matvec(dx) + res
        lin_err = jnp.linalg.norm(lin_res_vec) / res_norm
        is_trusted = lin_err < 0.1
        
        step_type = "STD"
        dx_final = dx
        
        if mode == 'C':
            is_type1 = (cos_align > 1e-14)
            if is_type1 and is_trusted:
                step_type = "SNIPER"
                # Rescue: Gradient flow scaled by residual
                dx_final = - (g / g_norm) * res_norm * dt * 0.5
            # ELSE: Keep dx (Newton), even if Type-2 or Trusted Type-1 is false
            # This allows C to accelerate when not in danger
        
        # Acceptance
        accepted = False
        if mode == 'A':
            alpha = 1.0
            for _ in range(10): # LS
                psi_try = psi + alpha * dx_final
                m_try = 0.5 * jnp.linalg.norm(residual_fn(psi_try, rho, x_1d))**2
                if check_acceptance(m_try, merit):
                    psi = psi_try; accepted = True; break
                alpha *= 0.5
        else:
            psi_try = psi + dx_final
            m_try = 0.5 * jnp.linalg.norm(residual_fn(psi_try, rho, x_1d))**2
            if check_acceptance(m_try, merit):
                psi = psi_try; accepted = True
                if step_type == "SNIPER": dt *= 1.1 # Cautious growth
                else: dt *= 1.5 # Aggressive growth
            else:
                dt *= 0.5
        
        if not accepted and mode == 'A': return (False, res_norm, "LS_FAIL")
        if not accepted and dt < dt_min: return (False, res_norm, "DT_STAGNATION")
        
    return (False, res_norm, "MAX_ITER")

# --- EXECUTION ---

if __name__ == "__main__":
    # The "Iron Proof" Profile from v5.0
    p = EvidenceProfile(rho_peak=10.0, gamma=2.0, init_mode='dip')
    
    print("\n>>> RUNNING GOVERNANCE PROOF (v7.0) <<<")
    
    # Run A (Benchmark)
    ok_A, r_A, msg_A = run_trial_governed(p, 'A', max_iter=500)
    print(f"Mode A: {msg_A} (Res={r_A:.2e})")
    
    # Run B (Control)
    ok_B, r_B, msg_B = run_trial_governed(p, 'B', max_iter=500)
    print(f"Mode B: {msg_B} (Res={r_B:.2e})")
    
    # Run C (Experiment)
    ok_C, r_C, msg_C = run_trial_governed(p, 'C', max_iter=500)
    print(f"Mode C: {msg_C} (Res={r_C:.2e})")