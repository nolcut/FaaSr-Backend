import random


def generate(n: int, values: list, seed: int = 0):
    if not values:
        raise ValueError("random_choice generator requires a non-empty 'values' list")
    for i in range(n):
        yield random.Random(seed + i).choice(values)
