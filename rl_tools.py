# reinforcement learning tools

from collections import deque
from jax import random

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
