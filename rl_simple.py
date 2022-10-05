# simple reinforcement learning - capital accumulation

import jax
import jax.lax as lax
import jax.numpy as np
import optax

from valjax import solve_binary
from rl_tools import rectify_lower, polynomial, chebyshev

# random init
# rs = RandomState(42)

# algo params
N = 1000 # capital grid size
ε = 1e-4 # utilty smoother

# parameters
β = 0.95 # discount rate
α = 0.00 # movement cost

# functions
u = rectify_lower(lambda x: -0.5*(np.log(x)**2), ε)
c = lambda d: -0.5*α*(d**2)
up = jax.grad(u)
cp = jax.grad(c)

# make state grid
xlo, xhi = 0.5, 2.0
xgrid = np.linspace(xlo, xhi, N)

# policy and value functions
pol = polynomial(4, zero=0.0)
val = polynomial(4, zero=0.0)

# evaluate policy level
def eval_policy(x, θp, θv):
    xp = pol(x, θp)
    vp = val(xp, θv)
    up = u(x) + c(xp-x)
    return up + β*vp

# evaluate valfunc fit
def eval_value(x, θp, θv):
    vp = val(x, θv)
    vn = eval_policy(x, θp, θv)
    return -(vn-vp)**2

# vectorized values
pol_vec = jax.vmap(pol, in_axes=(0, None))
val_vec = jax.vmap(val, in_axes=(0, None))
eval_policy_vec = jax.vmap(eval_policy, in_axes=(0, None, None))
eval_value_vec = jax.vmap(eval_value, in_axes=(0, None, None))

# vectorized grads
grad_policy_vec = jax.vmap(jax.grad(eval_policy, argnums=1), in_axes=(0, None, None))
grad_value_vec = jax.vmap(jax.grad(eval_value, argnums=2), in_axes=(0, None, None))

# applied to kgrid
def grad_policy_obj(θk, θv):
    return grad_policy_vec(xgrid, θk, θv).mean(axis=0)
def grad_value_obj(θk, θv):
    return grad_value_vec(xgrid, θk, θv).mean(axis=0)

def solve_iterate(R=1000, Δp=0.01, Δv=0.01, Mp=0.1, Mv=0.1):
    # custom optimizers
    optim_p = optax.chain(optax.clip(Mp), optax.scale(Δp))
    optim_v = optax.chain(optax.clip(Mv), optax.scale(Δv))

    @jax.jit
    def step(θp, θv, state_p, state_v):
        # compute gradients
        grad_p = grad_policy_obj(θp, θv)
        grad_v = grad_value_obj(θp, θv)

        # update states
        upd_p, state_p = optim_p.update(grad_p, state_p)
        upd_v, state_v = optim_v.update(grad_v, state_v)

        # apply updates
        θp = optax.apply_updates(θp, upd_p)
        θv = optax.apply_updates(θv, upd_v)

        return θp, θv, state_p, state_v

    # init params
    θp = np.array([1.0, 0.0, 0.0, 0.0])
    θv = np.array([-1.0, 2.0, -1.0, 0.0])

    # init optimizers
    state_p = optim_p.init(θp)
    state_v = optim_v.init(θv)

    # iterate and optimize
    for i in range(R):
        θp, θv, state_p, state_v = step(θp, θv, state_p, state_v)

    return θp, θv
