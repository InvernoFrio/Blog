"""CPU reference checks for the operator learning notes. Python standard library only."""
import math
import random


def offset(index, strides, storage_offset=0):
    return storage_offset + sum(i * s for i, s in zip(index, strides))


def stable_softmax(values):
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def tiled_matmul(a, b, tile=3):
    m, k, n = len(a), len(b), len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for row in range(0, m, tile):
        for column in range(0, n, tile):
            for reduction in range(0, k, tile):
                for i in range(row, min(row + tile, m)):
                    for j in range(column, min(column + tile, n)):
                        for r in range(reduction, min(reduction + tile, k)):
                            out[i][j] += a[i][r] * b[r][j]
    return out


def attention_reference(q, keys, values):
    scores = [sum(a * b for a, b in zip(q, key)) / math.sqrt(len(q)) for key in keys]
    probabilities = stable_softmax(scores)
    return [sum(p * value[d] for p, value in zip(probabilities, values))
            for d in range(len(values[0]))]


def online_attention(q, keys, values, block=3):
    maximum, normalizer = -math.inf, 0.0
    accumulator = [0.0] * len(values[0])
    for start in range(0, len(keys), block):
        scores = [sum(a * b for a, b in zip(q, key)) / math.sqrt(len(q))
                  for key in keys[start:start + block]]
        next_maximum = max(maximum, max(scores))
        correction = math.exp(maximum - next_maximum)
        weights = [math.exp(score - next_maximum) for score in scores]
        accumulator = [correction * old + sum(weight * value[d]
                       for weight, value in zip(weights, values[start:start + block]))
                       for d, old in enumerate(accumulator)]
        normalizer = correction * normalizer + sum(weights)
        maximum = next_maximum
    return [value / normalizer for value in accumulator]


def assert_close(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10), (left, right)


def main():
    # A 3 x 4 row-major array, a transpose view, a sliced view, and a broadcast.
    assert offset((2, 3), (4, 1)) == 11
    assert offset((3, 2), (1, 4)) == 11
    assert offset((1, 1), (4, 2), 1) == 7
    assert offset((2, 3), (0, 1)) == 3
    cases = 0
    for n in (0, 1, 255, 256, 257, 1025):
        covered = [index for program in range((n + 255) // 256)
                   for index in range(program * 256, (program + 1) * 256) if index < n]
        assert covered == list(range(n))
        cases += 1
    for row in ([0, 0, 0], [10000, 9999, -10000], [-10000, -10001], [5]):
        result = stable_softmax(row)
        assert math.isclose(sum(result), 1.0, abs_tol=1e-12)
        assert_close(result, stable_softmax([value + 37 for value in row]))
        cases += 1
    rng = random.Random(17)
    for m, n, k in ((1, 1, 1), (3, 5, 7), (7, 3, 5)):
        a = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(m)]
        b = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(k)]
        expected = matmul(a, b)
        for tile in (1, 2, 4):
            actual = tiled_matmul(a, b, tile)
            for left, right in zip(actual, expected):
                assert_close(left, right)
            cases += 1
    for scale in (1, 100):
        q = [rng.uniform(-scale, scale) for _ in range(7)]
        keys = [[rng.uniform(-scale, scale) for _ in range(7)] for _ in range(13)]
        values = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(13)]
        expected = attention_reference(q, keys, values)
        for block in (1, 3, 8, 16):
            assert_close(online_attention(q, keys, values, block), expected)
            cases += 1
    print(f"PASS: 4 layout checks and {cases} mask, softmax, GEMM, and online-attention cases")


if __name__ == "__main__":
    main()
