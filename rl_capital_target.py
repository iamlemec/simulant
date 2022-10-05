# simple reinforcement learning - capital accumulation

import jax
import jax.lax as lax
import jax.numpy as np
import optax

from valjax import solve_binary, interp
from rl_tools import RandomState, logit, rectify_lower

# random init
# rs = RandomState(42)

# algo params
N = 1000 # capital grid size
M = 3 # degree of valfunc polynomial
ε = 1e-4 # utilty smoother

# parameters
β = 0.95 # discount rate
α = 0.65 # cobb-douglas factor (capital)
δ = 0.10 # depreciation rate
z = 0.50 # productivity

# functions
u = rectify_lower(np.log, ε)
f = lambda k: z*k**α
up = jax.grad(u)
fp = jax.grad(f)

# steady state
kss_obj = lambda k: 1 - β * ( fp(k) + (1-δ) )
kss = solve_binary(kss_obj, 0.01, 100.0)

# get max reasonable
kmax_obj = lambda k: fp(k) - δ
kmax = solve_binary(kmax_obj, 0.01, 100.0)

# make capital grid
klo, khi = 0.2*kss, 2.0*kss
kgrid = np.linspace(klo, khi, N)

# policy function
def pol(k, θk):
    x = (k-kss)**np.arange(M)
    return np.sum(x*θk)

# value function
def val(k, θv):
    x = (k-kss)**np.arange(M)
    return np.sum(x*θv)

# evaluate policy level
def eval_policy(k, θk, θv):
    yp = f(k) + (1-δ)*k
    kp = pol(k, θk)
    vp = val(kp, θv)
    up = u(yp-kp)
    return up + β*vp

# evaluate valfunc fit
def eval_value(k, θk, θv, θkp, θvp):
    vp = val(k, θv)
    vn = eval_policy(k, θkp, θvp)
    return -(vn-vp)**2

# vectorized values
pol_vec = jax.vmap(pol, in_axes=(0, None))
val_vec = jax.vmap(val, in_axes=(0, None))
eval_policy_vec = jax.vmap(eval_policy, in_axes=(0, None, None))
eval_value_vec = jax.vmap(eval_value, in_axes=(0, None, None, None, None))

# vectorized grads
grad_policy_vec = jax.vmap(jax.grad(eval_policy, argnums=1), in_axes=(0, None, None))
grad_value_vec = jax.vmap(jax.grad(eval_value, argnums=2), in_axes=(0, None, None, None, None))

# applied to kgrid
def grad_policy_obj(θk, θv):
    return grad_policy_vec(kgrid, θk, θv).mean(axis=0)
def grad_value_obj(θk, θv, θkp, θvp):
    return grad_value_vec(kgrid, θk, θv, θkp, θvp).mean(axis=0)

def solve_iterate(R=10, Δk=0.01, Δv=0.01, ϵk=1e-4, ϵv=1e-4, Mk=0.1, Mv=0.1, τ=0.01):
    # custom optimizers
    optim_k = optax.chain(optax.clip(Mk), optax.scale(Δk))
    optim_v = optax.chain(optax.clip(Mv), optax.scale(Δv))

    @jax.jit
    def step(θk, θv, θkp, θvp, state_k, state_v, i):
        # compute gradients
        grad_k = grad_policy_obj(θk, θv)
        grad_v = grad_value_obj(θk, θv, θkp, θvp)

        # update states
        upd_k, state_k = optim_k.update(grad_k, state_k)
        upd_v, state_v = optim_v.update(grad_v, state_v)

        # if i % 100 == 0:
        #     print(i)
        #     print(θv, -eval_value_vec(kgrid, θk, θv, θkp, θvp).mean())
        #     print(grad_v)
        #     print(upd_v)
        #     print()

        # apply updates
        θk = optax.apply_updates(θk, upd_k)
        θv = optax.apply_updates(θv, upd_v)

        # update targets
        θkp = τ*θk + (1-τ)*θkp
        θvp = τ*θv + (1-τ)*θvp

        return θk, θv, θkp, θvp, state_k, state_v

    # init params
    θk = 10**(-np.arange(M, dtype=np.float32))
    θv = 10**(-np.arange(M, dtype=np.float32))
    θkp = θk.copy()
    θvp = θv.copy()

    # init optimizers
    state_k = optim_k.init(θk)
    state_v = optim_v.init(θv)

    # iterate and optimize
    for i in range(R):
        θk, θv, θkp, θvp, state_k, state_v = step(θk, θv, θkp, θvp, state_k, state_v, i)

    return θk, θv, θkp, θvp
