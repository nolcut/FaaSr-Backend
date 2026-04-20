def generate(n: int, start: int = 0, step: int = 1):
    for i in range(n):
        yield start + i * step
