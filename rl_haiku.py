import jax
import jax.numpy as np
import jax.nn as nn
import haiku as hk
import optax

# algo
N = 100 # grid size
K = 10 # network width

# params
β = 0.95 # discount rate

# randoms
rng = jax.random.PRNGKey(42)

# grid
grid = np.linspace(0, 1, N)
ugrid = np.stack([
    np.hstack([[0.0], grid[:-1]]),
    grid,
    np.hstack([grid[1:], [1.0]])
], axis=1)

# expanded
grid1 = grid[:, None].copy()
ugrid1 = ugrid[:, :, None].copy()

# functions
u = lambda x: -0.5*x**2
c = 1.0*np.array([1.0, 0.0, 1.0])

# policy function: position → dist(action)
def pol(x):
    mlp = hk.Sequential([
        hk.Linear(K), nn.relu,
        hk.Linear(K), nn.relu,
        hk.Linear(3)
    ])
    logits = mlp(x)
    probs = nn.softmax(logits)
    return probs

# value function: position → value
def val(x):
    mlp = hk.Sequential([
        hk.Linear(K), nn.relu,
        hk.Linear(K), nn.relu,
        hk.Linear(1)
    ])
    logits = mlp(x)
    probs = nn.softmax(logits)
    return probs

# initialize policy function
fpol = hk.without_apply_rng(hk.transform(pol))
θpol0 = fpol.init(rng, grid1)

# initialize value function
fval = hk.without_apply_rng(hk.transform(val))
θval0 = fval.init(rng, grid1)

# generate expected value
def eval_policy(θpol, θval):
    xp = fpol.apply(θpol, grid1)
    vp = fval.apply(θval, ugrid1)[:, :, 0]
    up = u(grid)[:, None] + c[None, :]
    vn = up + β*vp
    ve = (xp*vn).sum(axis=1)
    return ve

# evaluate valfunc fit
def eval_value(θpol, θval):
    vp = fval.apply(θval, grid1)[:, 0]
    vn = eval_policy(θpol, θval)
    return -(vn-vp)**2

def policy_obj(θpol, θval):
    return eval_policy(θpol, θval).mean()
grad_policy_obj = jax.grad(policy_obj, argnums=0)

def value_obj(θpol, θval):
    return eval_value(θpol, θval).mean()
grad_value_obj = jax.grad(value_obj, argnums=1)

def solve_iterate(R=1000, Δpol=0.01, Δval=0.01, Mpol=0.1, Mval=0.1):
    # custom optimizers
    optim_pol = optax.chain(optax.clip(Mpol), optax.scale(Δpol))
    optim_val = optax.chain(optax.clip(Mval), optax.scale(Δval))

    @jax.jit
    def step(θpol, θval, state_pol, state_val):
        # compute gradients
        grad_pol = grad_policy_obj(θpol, θval)
        grad_val = grad_value_obj(θpol, θval)

        # update states
        upd_pol, state_pol = optim_pol.update(grad_pol, state_pol)
        upd_val, state_val = optim_val.update(grad_val, state_val)

        # apply updates
        θpol = optax.apply_updates(θpol, upd_pol)
        θval = optax.apply_updates(θval, upd_val)

        return θpol, θval, state_pol, state_val

    # init params
    θpol = θpol0
    θval = θval0

    # init optimizers
    state_pol = optim_pol.init(θpol)
    state_val = optim_val.init(θval)

    # iterate and optimize
    for i in range(R):
        θpol, θval, state_pol, state_val = step(θpol, θval, state_pol, state_val)

    return θpol, θval
