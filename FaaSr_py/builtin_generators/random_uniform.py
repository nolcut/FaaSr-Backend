import random


def generate(n: int, low: float, high: float, seed: int = 0):
    for i in range(n):
        yield random.Random(seed + i).uniform(low, high)
