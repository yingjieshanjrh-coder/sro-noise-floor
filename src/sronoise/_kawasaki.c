/* Swap Monte Carlo for a multicomponent lattice gas with shell-resolved
 * pair interactions.  Compiled on demand by sronoise.mc. */
#include <math.h>
#include <stdint.h>

static uint64_t rng_state;

static inline uint64_t next_u64(void)
{
    /* splitmix64 */
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline double next_unit(void)
{
    return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

static inline double site_energy(int i, int a, const int *cfg, int nshell,
                                 const int *z, const int *off,
                                 const int *nbr, const double *V, int K)
{
    double e = 0.0;
    for (int s = 0; s < nshell; s++) {
        const int *nb = nbr + off[s] + (long)i * z[s];
        const double *Vs = V + (long)s * K * K + (long)a * K;
        for (int k = 0; k < z[s]; k++)
            e += Vs[cfg[nb[k]]];
    }
    return e;
}

/* nbr: concatenated (N x z[s]) neighbour tables, off[s] is the offset of
 * shell s.  V: nshell x K x K symmetric interactions (energy units).
 * beta: 1/kT in the same units.  Returns the acceptance ratio. */
double swap_mc(int N, int K, int nshell, const int *z, const int *off,
               const int *nbr, const double *V, double beta, int *cfg,
               long nsweeps, uint64_t seed)
{
    rng_state = seed;
    long acc = 0, tried = 0;
    for (long sweep = 0; sweep < nsweeps; sweep++) {
        for (int t = 0; t < N; t++) {
            int i = (int)(next_u64() % (uint64_t)N);
            int j = (int)(next_u64() % (uint64_t)N);
            int a = cfg[i], b = cfg[j];
            if (a == b) continue;
            tried++;
            double before = site_energy(i, a, cfg, nshell, z, off, nbr, V, K)
                          + site_energy(j, b, cfg, nshell, z, off, nbr, V, K);
            cfg[i] = b; cfg[j] = a;
            double after = site_energy(i, b, cfg, nshell, z, off, nbr, V, K)
                         + site_energy(j, a, cfg, nshell, z, off, nbr, V, K);
            double dE = after - before;
            if (dE <= 0.0 || next_unit() < exp(-beta * dE)) {
                acc++;
            } else {
                cfg[i] = a; cfg[j] = b;
            }
        }
    }
    return tried ? (double)acc / tried : 0.0;
}
