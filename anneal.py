"""
Simulated annealing for Ising-like models on a NetworkX graph.

Hamiltonian: H = -sum_i h_i s_i - sum_{<ij>} J_ij s_i s_j.

Build the C library first:
  macOS:    cc -O3 -march=native -std=c99 -Wall -Wextra -fPIC -shared anneal.c -o libanneal.dylib -lm
  Linux:    cc -O3 -march=native -std=c99 -Wall -Wextra -fPIC -shared anneal.c -o libanneal.so    -lm
  Windows:  cc -O3 -march=native -std=c99 -Wall -Wextra       -shared anneal.c -o anneal.dll      -lm   (MinGW-w64 / MSYS2)
"""

import ctypes, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

_libname = {"darwin": "libanneal.dylib", "win32": "anneal.dll"}.get(sys.platform, "libanneal.so")
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), _libname))
_pi32 = ctypes.POINTER(ctypes.c_int32)
_pi8  = ctypes.POINTER(ctypes.c_int8)
_pf64 = ctypes.POINTER(ctypes.c_double)
_lib.anneal_ising.argtypes = [
    ctypes.c_int32, _pi32, _pi32, _pf64, _pf64,
    _pi8, _pi8,
    ctypes.c_double, ctypes.c_int32, ctypes.c_uint64,
    _pf64, _pf64,
]
_lib.anneal_ising.restype = ctypes.c_int

def anneal(G, beta_final, n_sweeps, weight="weight", field="weight", initial_spins=None):
    n = G.number_of_nodes()
    deg = np.fromiter((G.degree(i) for i in range(n)), dtype=np.int32, count=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(deg, out=offsets[1:])
    neighbors = np.empty(int(offsets[-1]), dtype=np.int32)
    J = np.empty(int(offsets[-1]), dtype=np.float64)
    for i in range(n):
        for k, (j, attrs) in enumerate(G[i].items()):
            neighbors[offsets[i] + k] = j
            J[offsets[i] + k] = attrs.get(weight, 1.0)
    h = np.fromiter((G.nodes[i].get(field, 0.0) for i in range(n)), dtype=np.float64, count=n)

    if initial_spins is None:
        spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=n)
    else:
        spins = np.ascontiguousarray(initial_spins, dtype=np.int8).copy()
    best_spins = np.empty(n, dtype=np.int8)
    final_E = ctypes.c_double(); best_E = ctypes.c_double()
    seed = int(np.random.randint(1, 1 << 63))

    rc = _lib.anneal_ising(
        n,
        offsets.ctypes.data_as(_pi32), neighbors.ctypes.data_as(_pi32),
        J.ctypes.data_as(_pf64), h.ctypes.data_as(_pf64),
        spins.ctypes.data_as(_pi8), best_spins.ctypes.data_as(_pi8),
        ctypes.c_double(beta_final), ctypes.c_int32(n_sweeps), ctypes.c_uint64(seed),
        ctypes.byref(final_E), ctypes.byref(best_E),
    )
    if rc != 0:
        raise RuntimeError(f"anneal_ising returned {rc}")
    return spins, final_E.value, best_spins, best_E.value

def parallel_anneal(G, beta_final, n_sweeps, K, weight="weight", field="weight"):
    with ThreadPoolExecutor(max_workers=K) as ex:
        runs = list(ex.map(
            lambda _: anneal(G, beta_final, n_sweeps, weight=weight, field=field),
            range(K)))
    # return min(runs, key=lambda r: r[3])
    return np.vstack([r[0] for r in runs])