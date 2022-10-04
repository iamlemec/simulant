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
N = 100 # capital grid size
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
klo, khi = 0.2*kss, 2*kss
kgrid = np.linspace(klo, khi, N)

# policy function
def pol(k, θk):
    x = (k-kss)**np.arange(M)
    return np.dot(x, θk)
pol_vec = jax.vmap(pol, in_axes=(0, None))

# value function
def val(k, θv):
    x = (k-kss)**np.arange(M)
    return np.dot(x, θv)
val_vec = jax.vmap(val, in_axes=(0, None))

# evaluate policy/valfunc pair
def eval_policy(k, θk, θv):
    yp = f(k) + (1-δ)*k
    kp, vp = np.maximum(0, pol(k, θk)), val(k, θv)
    return u(yp-kp) + β*vp
eval_policy_vec = jax.jit(jax.vmap(eval_policy, in_axes=(0, None, None)))
grad_policy_vec = jax.jit(jax.vmap(jax.grad(eval_policy, argnums=1), in_axes=(0, None, None)))

def grad_policy_obj(θk, θv):
    return -grad_policy_vec(kgrid, θk, θv).mean(axis=0)

# evaluate fit of value function projection
def eval_value(k, θk, θv):
    vp = val(k, θv)
    vn = eval_policy(k, θk, θv)
    return -(vn-vp)**2
eval_value_vec = jax.jit(jax.vmap(eval_value, in_axes=(0, None, None)))
eval_value_vec = jax.jit(jax.vmap(jax.grad(eval_value, argnums=2), in_axes=(0, None, None)))

def grad_value_obj(θk, θv):
    return -eval_value_vec(kgrid, θk, θv).mean(axis=0)

def solve_iterate(R=10, Δk=0.01, Δv=0.01, ϵk=1e-4, ϵv=1e-4, Mk=100, Mv=100):
    # custom optimizers
    optim_k = optax.chain(
        optax.clip(Mk), optax.scale_by_adam(eps=ϵk), optax.scale(-Δk)
    )
    optim_v = optax.chain(
        optax.clip(Mv), optax.scale_by_adam(eps=ϵv), optax.scale(-Δv)
    )

    @jax.jit
    def step(kpoly, theta, state_k, state_v):
        # compute gradients
        grad_k = grad_policy_obj(kpoly, theta)
        grad_v = grad_value_obj(kpoly, theta)

        # update states
        upd_k, state_k = optim_k.update(grad_k, state_k)
        upd_v, state_v = optim_v.update(grad_v, state_v)

        # apply updates
        kpoly = optax.apply_updates(kpoly, upd_k)
        theta = optax.apply_updates(theta, upd_v)

        return kpoly, theta, state_k, state_v

    # init params
    kpoly = np.array([5.0, 0.1, 0.0])
    theta = np.array([0.0, 1.0, 0.0])

    # init optimizers
    state_k = optim_k.init(kpoly)
    state_v = optim_v.init(theta)

    # iterate and optimize
    for i in range(R):
        kpoly, theta, state_k, state_v = step(kpoly, theta, state_k, state_v)

    return kpoly, theta
