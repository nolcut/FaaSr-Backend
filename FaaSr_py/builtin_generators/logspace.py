def generate(n: int, start: float, stop: float, base: float = 10.0):
    for i in range(n):
        if n == 1:
            yield float(base ** start)
        else:
            exponent = start + (stop - start) * i / (n - 1)
            yield float(base ** exponent)
