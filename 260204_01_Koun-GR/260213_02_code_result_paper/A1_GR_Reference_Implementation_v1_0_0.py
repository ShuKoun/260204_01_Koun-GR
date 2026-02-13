# koun_a1_gr_final.py
# ============================================================================
# KOUN-A1-GR: STRUCTURAL GOVERNANCE REFERENCE IMPLEMENTATION (v1.0)
# ----------------------------------------------------------------------------
# Author: Shu Koun
# Date:   2026-02-13
#
# PURPOSE:
#   Demonstrate the "Governance Gap" in Pseudo-Transient Continuation (PTC)
#   for Indefinite Hamiltonian Constraints.
#
# SCENARIO:
#   High-Gravity (Rho=10), Extreme-Stretch (Gamma=2.0), Topological Trap (Dip).
#
# MODES:
#   [A] Pure Newton (Benchmark): Proves physics is valid (Converges).
#   [B] Blind PTC (Control):     Proves PTC gets stuck in geometric traps.
#   [C] Koun-A1   (Governance):  Proves Sentinel+Rescue unblocks PTC.
# ============================================================================

import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.sparse.linalg import gmres
from functools import partial
from typing import NamedTuple, Literal

jax.config.update("jax_enable_x64", True)

# --- 1. THE PHYSICS KERNEL (HAMILTONIAN CONSTRAINT) ---

def get_grid(N, L_max, gamma):
    xi = jnp.linspace(-1.0, 1.0, N)
    safe_gamma = jnp.where(jnp.abs(gamma)<1e-9, 1e-9, gamma)
    x = L_max * jnp.sinh(safe_gamma * xi) / jnp.sinh(safe_gamma)
    return x

@jit
def residual_fn(psi, rho, x_1d):
    # Eq: Laplacian(psi) - 2*pi*rho*psi^5 = 0
    N = x_1d.shape[0]
    h = x_1d[1:] - x_1d[:-1]
    
    # 3D Laplacian on Stretched Grid
    def d2(f):
        df = f[1:] - f[:-1]; g = df/h
        d2f = (g[1:] - g[:-1]) * 2.0 / (h[1:] + h[:-1])
        return jnp.concatenate([jnp.array([0.]), d2f, jnp.array([0.])])
    
    L = vmap(vmap(d2,0),0)(psi) + \
        vmap(vmap(d2,0),0)(psi.transpose(0,2,1)).transpose(0,2,1) + \
        vmap(vmap(d2,0),0)(psi.transpose(2,1,0)).transpose(2,1,0)
    
    res = L - 2.0 * jnp.pi * rho * (psi**5)
    
    # Boundary Conditions (psi=1 at boundary, res=0)
    mask = jnp.ones_like(psi).at[0].set(0).at[-1].set(0) \
           .at[:,0].set(0).at[:,-1].set(0).at[:,:,0].set(0).at[:,:,-1].set(0)
    return res * mask

@jit
def merit_loss(psi, rho, x_1d):
    F = residual_fn(psi, rho, x_1d)
    return 0.5 * jnp.sum(F**2)

grad_merit = jit(jax.grad(merit_loss, argnums=0))

# --- 2. THE GOVERNANCE PROTOCOL ---

def check_acceptance(merit_new, merit_old):
    # Dual Tolerance Law (Protocol Freeze v1.4.2)
    return merit_new <= merit_old * (1.0 + 1e-12) + 1e-15

def run_solver(mode: Literal['A', 'B', 'C'], max_iter=100):
    # -- Setup --
    N, L_max = 32, 50.0
    rho_peak, gamma, sigma = 10.0, 2.0, 1.5
    
    x_1d = get_grid(N, L_max, gamma)
    X, Y, Z = jnp.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    rho = rho_peak * jnp.exp(-r2 / (2 * sigma**2))
    
    # Initial Guess: "The Trap"
    psi = jnp.maximum(1.0 - 0.9*jnp.exp(-r2/20.0), 0.1)
    
    dt = 1e9 if mode == 'A' else 0.1
    dt_min, dt_max = 1e-5, 1e9
    
    def jvp_op(v, p): return jax.jvp(partial(residual_fn, rho=rho, x_1d=x_1d), (p,), (v,))[1]
    
    print(f"\n>>> MODE [{mode}] START <<<")
    print(f"{'It':<3}| {'ResNorm':<9}| {'Merit':<9}| {'dt':<8}| {'CosAlign':<8}| {'Step':<6}| {'Status'}")
    print("-" * 80)
    
    for k in range(max_iter):
        # 1. State Assessment
        res = residual_fn(psi, rho, x_1d)
        res_norm = jnp.linalg.norm(res)
        merit = 0.5 * res_norm**2
        
        if res_norm < 1e-8:
            print(f"{k:<3}| {res_norm:<9.3e}| {merit:<9.3e}| ---     | ---     | ---   | [CONVERGED]")
            return
            
        g = grad_merit(psi, rho, x_1d)
        g_norm = jnp.linalg.norm(g) + 1e-20
        
        # 2. Linear Solve
        # A: J dx = -F
        # B/C: (J + I/dt) dx = -F
        def matvec(v):
            op = jvp_op(v, psi)
            if mode != 'A': op += (1.0/dt)*v
            return op
        
        dx, _ = gmres(matvec, -res, tol=1e-6, maxiter=100)
        
        # 3. Sentinel Audit (CosAlign)
        dx_norm = jnp.linalg.norm(dx) + 1e-20
        cos_align = jnp.sum(g * dx) / (g_norm * dx_norm)
        
        # 4. Governance Logic
        step_type = "STD"
        dx_final = dx
        
        if mode == 'C':
            # Koun-A1 Protocol
            is_type1 = cos_align > 1e-14
            is_type2 = abs(cos_align) <= 1e-14
            
            if is_type1:
                step_type = "SNIPER"
                # Rescue: Gradient flow
                dx_final = - (g / g_norm) * res_norm * dt * 0.5
            elif is_type2:
                step_type = "CONS" # Conservative
        
        # 5. Acceptance & Update
        accepted = False
        
        # Trial
        if mode == 'A': # Line Search for Newton
            alpha = 1.0
            for _ in range(8):
                p_try = psi + alpha * dx_final
                m_try = 0.5 * jnp.linalg.norm(residual_fn(p_try, rho, x_1d))**2
                if check_acceptance(m_try, merit):
                    psi = p_try; accepted = True; status = f"ACC(LS={alpha:.2g})"
                    break
                alpha *= 0.5
            if not accepted: status = "REJ(LS)"
        else: # PTC Retry
            p_try = psi + dx_final
            m_try = 0.5 * jnp.linalg.norm(residual_fn(p_try, rho, x_1d))**2
            if check_acceptance(m_try, merit):
                psi = p_try; accepted = True
                tag = "[Rescue]" if step_type=="SNIPER" else ""
                status = f"ACC{tag}"
                dt = min(dt*1.1 if step_type=="SNIPER" else dt*1.5, dt_max)
            else:
                accepted = False
                tag = "[T1]" if (mode=='C' and step_type=="SNIPER") else ""
                status = f"REJ{tag}->RETRY"
                dt = max(dt * 0.5, dt_min)
        
        # Logging
        print(f"{k:<3}| {res_norm:<9.3e}| {merit:<9.3e}| {dt:<8.2e}| {cos_align:<8.2f}| {step_type:<6}| {status}")
        
        # Failure Exit
        if not accepted and mode == 'A':
            print(">>> FAILED: Line Search Stagnation"); return
        if not accepted and dt <= dt_min*1.1:
            print(">>> FAILED: DT Stagnation"); return

if __name__ == "__main__":
    run_solver('A') # Benchmark
    run_solver('B') # Control (The Failure)
    run_solver('C') # Governance (The Success)