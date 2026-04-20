def generate(n: int, values: list):
    if not values:
        raise ValueError("from_list generator requires a non-empty 'values' list")
    for i in range(n):
        yield values[i % len(values)]
