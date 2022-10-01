# simple reinforcement learning - capital accumulation

import jax
import jax.numpy as np

from valjax import solve_binary
from rl_tools import RandomState

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
u = lambda c: np.where(c < ε, (c-ε)/ε+np.log(ε), np.log(c))
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
ypgrid = f(kgrid) + (1-δ)*kgrid

# value function
def val(k, θ):
    x = (k-kss)**np.arange(M)
    return np.dot(x, θ)
val_vec = jax.vmap(val, in_axes=(0, None))

# evaluate policy/valfunc pair
# kp = policy choices on capital grid
# θ = value function parameters
def eval_policy(kp, k, θ):
    yp = f(k) + (1-δ)*k
    return u(yp-kp) + β*val(kp, θ)
eval_policy_vec = jax.vmap(eval_policy, in_axes=(0, 0, None))
grad_policy_vec = jax.jit(jax.vmap(jax.grad(eval_policy), in_axes=(0, 0, None)))

def grad_policy_obj(kp, θ):
    return grad_policy_vec(kp, kgrid, θ)

# evaluate fit of value function projection
def value_obj(kp, θ):
    vprev = val_vec(kgrid, θ)
    vnext = eval_policy_vec(kp, kgrid, θ)
    return -np.sum((vnext-vprev)**2)
grad_value_obj = jax.jit(jax.grad(value_obj, argnums=1))

def solve_iterate(R=10, Δk=0.01, Δv=0.01, Kmax=100, Vmax=100):
    # init value function
    kpoly = kgrid.copy()
    theta = np.zeros(M)

    # iterate and optimize
    for i in range(R):
        # update policy
        kgrad = grad_policy_obj(kpoly, theta)
        kpoly = np.clip(kpoly + Δk*kgrad, 0, Kmax)

        # update value
        vgrad = grad_value_obj(kpoly, theta)
        theta = np.clip(theta + Δv*vgrad, -Vmax, Vmax)

    return kpoly, theta
