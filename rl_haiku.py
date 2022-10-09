import jax
import jax.numpy as np
import jax.tree_util as trees
import jax.lax as lax
import jax.nn as nn
import haiku as hk
import optax

import valjax as vj

# algo
N = 21 # grid size
K = 10 # network width

# params
β = 0.95 # discount rate

# randoms
rng = jax.random.PRNGKey(42)

# grid
grid = np.linspace(-1, 1, N)
ugrid = np.stack([
    np.hstack([[-1.0], grid[:-1]]),
    grid,
    np.hstack([grid[1:], [1.0]])
], axis=1)

# functions
u = lambda x: 1 - x**2
c = 1.0*(1/N)*np.array([1.0, 0.0, 1.0])

# expanded
grid1 = grid[:, None].copy()
ugrid1 = ugrid[:, :, None].copy()

# value function: position → value
def val(x):
    mlp = hk.Sequential([
        hk.Linear(K), nn.relu,
        hk.Linear(K), nn.relu,
        hk.Linear(K), nn.relu,
        hk.Linear(1)
    ])
    levels = mlp(x)
    return levels

# initialize value function
fval = hk.without_apply_rng(hk.transform(val))
θval0 = fval.init(rng, grid1)

# get squeezed values
def get_value(θval, grid):
    return fval.apply(θval, grid).squeeze(axis=-1)

# find optimal policy
def max_policy(θval):
    vp = get_value(θval, ugrid1)
    xs = vp.argmax(axis=1)
    return xs

# generate expected value
def eval_policy(θval1, θval2):
    vp = get_value(θval1, ugrid1)
    xs = max_policy(θval2)
    vs = vj.address(vp, xs, axis=1)
    vn = u(grid) - c[xs] + β*vs
    return vn

# evaluate valfunc fit
def eval_value(θval, val):
    vtarg = get_value(θval, grid1)
    return -optax.l2_loss(val, vtarg).mean()
grad_value = jax.grad(eval_value)

def solve_iterate(R=1000, Δval=0.01, Mval=0.1, τ=0.01):
    # custom optimizers
    optim_val = optax.chain(optax.clip(Mval), optax.scale(Δval))

    @jax.jit
    def step(θval1, θval2, state_val):
        # update value function
        targ_val = eval_policy(θval1, θval2)
        grad_val = grad_value(θval1, targ_val)
        upd_val, state_val = optim_val.update(grad_val, state_val)
        θval1 = optax.apply_updates(θval1, upd_val)

        # update value target
        θval2 = trees.tree_map(lambda t1, t2: τ*t1 + (1-τ)*t2, θval1, θval2)

        return θval1, θval2, state_val

    # init params
    θval1 = θval0
    θval2 = θval0

    # init optimizers
    state_val = optim_val.init(θval1)

    # iterate and optimize
    for i in range(R):
        θval1, θval2, state_val = step(θval1, θval2, state_val)

    return θval1
