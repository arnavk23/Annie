import time
import json
import argparse
import numpy as np
from rust_annie import AnnIndex, Distance
import os
from datetime import datetime

def pure_python_search(data, ids, q, k):
    dists = np.linalg.norm(data - q, axis=1)
    idx = np.argsort(dists)[:k]
    return ids[idx], dists[idx]

def benchmark(N, D, k, repeats):
    data = np.random.rand(N, D).astype(np.float32)
    ids = np.arange(N, dtype=np.int64)
    q = data[0]

    idx = AnnIndex(D, Distance.EUCLIDEAN)
    idx.add(data, ids)
    idx.search(q, k)  # warm-up

    t0 = time.perf_counter()
    for _ in range(repeats):
        idx.search(q, k)
    t_rust = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        pure_python_search(data, ids, q, k)
    t_py = (time.perf_counter() - t0) / repeats

    result = {
        "N": N,
        "D": D,
        "k": k,
        "repeats": repeats,
        "rust_avg_ms": round(t_rust * 1e3, 4),
        "python_avg_ms": round(t_py * 1e3, 4),
        "speedup": round(t_py / t_rust, 4),
        "timestamp": time.time()
    }

    return result

def save_result(result, out_dir="benchmarks"):
    os.makedirs(out_dir, exist_ok=True)
    commit_hash = os.popen("git rev-parse --short HEAD").read().strip()
    now = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    filename = f"{out_dir}/{now}_{commit_hash}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Benchmark saved to: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--save", action="store_true", help="Save benchmark result to file")
    args = parser.parse_args()

    result = benchmark(args.N, args.D, args.k, args.repeats)

    print(json.dumps(result, indent=2))

    if args.save:
        save_result(result)
