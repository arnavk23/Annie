import numpy as np
import time
from rust_annie import PyHnswIndex, AnnIndex


def benchmark(index_cls, name, dim=128, n=10_000, q=100, k=10):
    print(f"\nBenchmarking {name} with {n} vectors (dim={dim})...")

    # Data
    data = np.random.rand(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    queries = np.random.rand(q, dim).astype(np.float32)

    # Index setup
    index = index_cls(dims=dim)
    index.add(data, ids)

    # Warm-up + Timing
    times = []
    for i in range(q):
        start = time.perf_counter()
        _ = index.search(queries[i], k=k)
        times.append((time.perf_counter() - start) * 1000)

    print(f"  Avg query time: {np.mean(times):.3f} ms")
    print(f"  Max query time: {np.max(times):.3f} ms")
    print(f"  Min query time: {np.min(times):.3f} ms")


if __name__ == "__main__":
    benchmark(PyHnswIndex, "HNSW")
    benchmark(AnnIndex, "Brute-Force")
