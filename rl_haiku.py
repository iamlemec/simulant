import jax
import jax.numpy as np
import jax.lax as lax
import jax.nn as nn
import haiku as hk
import optax

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
c = 0.1*(1/N)*np.array([1.0, 0.0, 1.0])

# expanded
grid1 = grid[:, None].copy()
ugrid1 = ugrid[:, :, None].copy()

# value function: position → value
def val(x):
    mlp = hk.Sequential([
        hk.Linear(K), nn.relu,
        hk.Linear(K), nn.relu,
        hk.Linear(1)
    ])
    levels = mlp(x)
    return levels

# initialize value function
fval = hk.without_apply_rng(hk.transform(val))
θval0 = fval.init(rng, grid1)

# generate expected value
def eval_policy(θval1, θval2):
    vp1 = fval.apply(θval1, ugrid1)[:, :, 0]
    vp2 = fval.apply(θval2, ugrid1)[:, :, 0]
    xs = vp2.argmax(axis=1)
    vs = vp1[np.arange(N), xs]
    vn = u(grid) - c[xs] + β*vs
    return vn

# evaluate valfunc fit
def eval_value(θval1, θval2):
    vp = fval.apply(θval1, grid1)[:, 0]
    vn = eval_policy(θval1, θval2)
    return -optax.l2_loss(vn, vp).mean()
grad_value = jax.grad(value_obj)

def solve_iterate(R=1000, Δval=0.01, εval=1e-4, Mval=0.1):
    # custom optimizers
    optim_val = optax.chain(
        optax.clip(Mval), optax.scale_by_adam(eps=εval), optax.scale(Δval)
    )

    @jax.jit
    def step(θval1, θval2, state_val):
        # update value function
        grad_val = grad_value(θval1, θval2)
        upd_val, state_val = optim_val.update(grad_val, state_val)
        θval1 = optax.apply_updates(θval1, upd_val)

        # update value target
        θval2 = τ*θval1 + (1-τ)*θval2

        return θval1, θval2, state_val

    # init params
    θval1 = θval0
    θval2 = θval0

    # init optimizers
    state_val = optim_val.init(θval)

    # iterate and optimize
    for i in range(R):
        θval1, θval2, state_val = step(θval1, θval2, state_val)

    return θval
