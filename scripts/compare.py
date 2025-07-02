from rust_annie import PyHnswIndex, AnnIndex
import numpy as np
import time

def bench(index_cls, dim=128, n=10000, q=100, k=10):
    data = np.random.rand(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    queries = np.random.rand(q, dim).astype(np.float32)

    index = index_cls(dims=dim)
    index.add(data, ids)

    times = []
    for i in range(q):
        start = time.time()
        _ = index.search(queries[i], k=k)
        times.append((time.time() - start) * 1e3)

    print(f"{index_cls.__name__}: {np.mean(times):.3f} ms/query")

bench(PyHnswIndex)
bench(AnnIndex)
