// Simulated annealing for Ising-like models on a sparse graph (CSR).
// H = -sum_i h_i s_i - sum_{<ij>} J_ij s_i s_j, with each undirected edge
// stored twice in CSR so the inner-loop sum is one contiguous walk.
//
// Build (macOS):  cc -O3 -march=native -std=c99 -Wall -Wextra -fPIC -shared anneal.c -o libanneal.dylib
// Build (Linux):  cc -O3 -march=native -std=c99 -Wall -Wextra -fPIC -shared anneal.c -o libanneal.so

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// xoshiro256++ (Blackman/Vigna, public domain), seeded via splitmix64.
static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t xnext(uint64_t s[4]) {
    const uint64_t r = rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;    s[3] = rotl(s[3], 45);
    return r;
}
static inline double uniform01(uint64_t s[4]) {
    return (xnext(s) >> 11) * (1.0 / (double)(1ULL << 53));
}

int anneal_ising(
    int32_t n, const int32_t *offsets, const int32_t *neighbors,
    const double *J, const double *h,
    int8_t *spins, int8_t *best_spins,
    double beta_final, int32_t n_sweeps, uint64_t seed,
    double *final_energy, double *best_energy)
{
    if (n <= 0 || n_sweeps < 2) return 1;

    uint64_t sm = seed;
    uint64_t rng[4] = { splitmix64(&sm), splitmix64(&sm), splitmix64(&sm), splitmix64(&sm) };

    // Initial energy.
    double E = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double f = 0.0;
        for (int32_t k = offsets[i]; k < offsets[i + 1]; k++)
            f += J[k] * spins[neighbors[k]];
        E -= h[i] * spins[i] + 0.5 * spins[i] * f;
    }
    double bestE = E;
    memcpy(best_spins, spins, (size_t)n * sizeof(int8_t));

    int32_t *order = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!order) return 2;
    for (int32_t i = 0; i < n; i++) order[i] = i;

    for (int32_t sweep = 0; sweep < n_sweeps; sweep++) {
        double beta = beta_final * (double)sweep / (double)(n_sweeps - 1);

        for (int32_t i = n - 1; i > 0; i--) {       // Fisher-Yates
            int32_t j = (int32_t)(xnext(rng) % (uint64_t)(i + 1));
            int32_t t = order[i]; order[i] = order[j]; order[j] = t;
        }

        for (int32_t idx = 0; idx < n; idx++) {
            int32_t i = order[idx];
            double f = h[i];
            for (int32_t k = offsets[i]; k < offsets[i + 1]; k++)
                f += J[k] * spins[neighbors[k]];
            double dE = 2.0 * spins[i] * f;
            if (dE <= 0.0 || uniform01(rng) < exp(-beta * dE)) {
                spins[i] = (int8_t)(-spins[i]);
                E += dE;
                if (E < bestE) {
                    bestE = E;
                    memcpy(best_spins, spins, (size_t)n * sizeof(int8_t));
                }
            }
        }
    }

    free(order);
    *final_energy = E;
    *best_energy = bestE;
    return 0;
}