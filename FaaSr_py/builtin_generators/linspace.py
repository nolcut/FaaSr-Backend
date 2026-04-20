def generate(n: int, start: float, stop: float):
    for i in range(n):
        if n == 1:
            yield float(start)
        else:
            yield start + (stop - start) * i / (n - 1)
