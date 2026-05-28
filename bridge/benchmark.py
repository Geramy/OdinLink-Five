#!/usr/bin/env python3
"""TB-Bridge bandwidth + latency benchmark.

Run the server first:
    # on Mac (or any host across the TB5 cable):
    python3 bridge/tb_bridge_server.py --bind 0.0.0.0 -v

Then on the Linux/CUDA training host:
    python3 bridge/benchmark.py --host <mac-tb-net-ip>

Reports:
    - PUT bandwidth (host RAM → server RAM)
    - GET bandwidth (server RAM → host RAM)
    - Small-payload round-trip latency
    - If torch+CUDA available: PUT/GET against CUDA tensors
"""
import argparse
import statistics
import time

import numpy as np

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False

from tb_bridge_client import TBBridgeClient


def bench_bandwidth(cli, size_bytes: int, iters: int, label: str, use_cuda: bool = False):
    nelem = size_bytes // 2  # bfloat16 == 2 bytes
    if use_cuda:
        t = torch.randn(nelem, dtype=torch.bfloat16, device="cuda")
    elif _HAS_CUDA and False:  # opt-in only
        t = torch.randn(nelem, dtype=torch.bfloat16)
    else:
        t = np.random.randn(nelem).astype("float32")[:nelem]
        size_bytes = t.nbytes  # corrected actual size

    put_times = []
    get_times = []
    key = f"bench/{label}"
    for i in range(iters):
        t0 = time.perf_counter()
        cli.put(key, t)
        put_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = cli.get(key, device="cuda" if use_cuda else None)
        get_times.append(time.perf_counter() - t0)
    cli.delete(key)

    def mb_per_s(times):
        return size_bytes / statistics.median(times) / 1e6

    return {
        "size_mb": size_bytes / 1e6,
        "put_mb_s": mb_per_s(put_times),
        "get_mb_s": mb_per_s(get_times),
        "put_median_ms": statistics.median(put_times) * 1e3,
        "get_median_ms": statistics.median(get_times) * 1e3,
    }


def bench_latency(cli, iters: int):
    """Small-payload round-trip latency via STAT (no data transfer)."""
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cli.stat()
        times.append(time.perf_counter() - t0)
    return {
        "median_us": statistics.median(times) * 1e6,
        "p99_us": sorted(times)[int(iters * 0.99)] * 1e6,
        "min_us": min(times) * 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="bridge server IP (TB-net address)")
    ap.add_argument("--port", type=int, default=29800)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cuda", action="store_true", help="test CUDA tensor PUT/GET")
    args = ap.parse_args()

    cli = TBBridgeClient(args.host, args.port)

    print(f"\nLatency ({args.iters * 4} stat round-trips):")
    lat = bench_latency(cli, args.iters * 4)
    print(f"  median={lat['median_us']:.1f}µs  p99={lat['p99_us']:.1f}µs  min={lat['min_us']:.1f}µs")

    print(f"\nBandwidth (numpy float32, {args.iters} iters per size):")
    print(f"  {'size':>10} {'PUT MB/s':>12} {'GET MB/s':>12} {'PUT ms':>10} {'GET ms':>10}")
    for size_mb in [1, 16, 64, 256, 1024]:
        try:
            r = bench_bandwidth(cli, size_mb * (1 << 20), args.iters, f"np-{size_mb}MB")
            print(f"  {r['size_mb']:>9.0f}MB {r['put_mb_s']:>12.0f} {r['get_mb_s']:>12.0f} "
                  f"{r['put_median_ms']:>10.1f} {r['get_median_ms']:>10.1f}")
        except Exception as e:
            print(f"  {size_mb}MB -> FAIL: {e}")
            break

    if args.cuda and _HAS_CUDA:
        print(f"\nBandwidth (CUDA bfloat16 tensors, {args.iters} iters per size):")
        print(f"  {'size':>10} {'PUT MB/s':>12} {'GET MB/s':>12} {'PUT ms':>10} {'GET ms':>10}")
        for size_mb in [16, 64, 256, 1024]:
            try:
                r = bench_bandwidth(cli, size_mb * (1 << 20), args.iters, f"cuda-{size_mb}MB",
                                    use_cuda=True)
                print(f"  {r['size_mb']:>9.0f}MB {r['put_mb_s']:>12.0f} {r['get_mb_s']:>12.0f} "
                      f"{r['put_median_ms']:>10.1f} {r['get_median_ms']:>10.1f}")
            except Exception as e:
                print(f"  {size_mb}MB -> FAIL: {e}")
                break


if __name__ == "__main__":
    main()
