# reinforcement learning tools

import jax
import jax.lax as lax
import jax.numpy as np

from collections import deque
from jax import random

def logit(x):
    return 0.5 * (np.tanh(x / 2) + 1)

def rectify_lower(f, ε):
    df = jax.grad(f)
    fε, dε = f(ε), df(ε)
    def f1(x):
        return lax.cond(x < ε,
            lambda: (x-ε)*dε + fε,
            lambda: f(x)
        )
    return f1

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque([], maxlen=size)

    def push(self, x):
        self.buffer.append(x)

    def get(self):
        return list(self.buffer)

class RandomState:
    def __init__(self, seed):
        self.state = random.PRNGKey(seed)

    def uniform(self, shape=()):
        key, self.state = random.split(self.state)
        return random.uniform(key, shape=shape)

    def normal(self, shape=()):
        key, self.state = random.split(self.state)
        return random.normal(key, shape=shape)

    def randint(self, minval, maxval=None, shape=()):
        if maxval is None:
            minval, maxval = 0, minval
        key, self.state = random.split(self.state)
        return random.randint(key, shape=shape, minval=minval, maxval=maxval)
