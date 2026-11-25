# Theory

This note derives the exact first and second moments of Warren–Cowley (WC)
short-range-order parameters for a random arrangement of atoms in a finite
periodic supercell, and the leading-order expressions quoted in the README.

## 1. Setting

A periodic supercell has $N$ sites. Species $a = 1,\dots,K$ occupies
$n_a$ sites, $\sum_a n_a = N$, and $c_a = n_a/N$. For coordination shell
$s$ let $A^s$ be the symmetric adjacency matrix of the periodic neighbour
graph: $A^s_{ij}$ is the number of lattice vectors of shell $s$ that join
site $i$ to a periodic image of site $j$. In large cells $A^s_{ij}\in\{0,1\}$;
in small cells a pair can be joined by several vectors and $A^s_{ij}>1$.
We require $A^s_{ii}=0$ (no site is its own neighbour), which only excludes
cells smaller than the shell radius. Write $d^s_i=\sum_j A^s_{ij}$; for a
single-sublattice crystal $d^s_i = z_s$.

With occupation indicators $x_i^a = [\sigma_i = a]$, the ordered pair count is

$$ M^s_{ab} = \sum_{i,j} A^s_{ij}\, x_i^a x_j^b , \qquad
   \sum_b M^s_{ab} = z_s n_a . $$

The conventional WC parameter is

$$ \alpha^s_{ab} = 1 - \frac{P_s(b\mid a)}{c_b}
   = 1 - \frac{N\,M^s_{ab}}{z_s\, n_a\, n_b}. \tag{1} $$

**Null hypothesis.** All $N!/\prod_a n_a!$ arrangements are equally likely
(ideal random solid solution at the composition actually simulated). The
$x_i^a$ are then multivariate hypergeometric: for $m$ *distinct* sites,

$$ \mathbb E\big[x_{i_1}^{a_1}\cdots x_{i_m}^{a_m}\big]
   = F_m(a_1,\dots,a_m) = \frac{\prod_a (n_a)_{m_a}}{(N)_m}, \tag{2} $$

where $m_a$ counts how often $a$ appears among $a_1..a_m$ and
$(n)_m = n(n-1)\cdots(n-m+1)$.

## 2. Mean

Because $A^s_{ii}=0$ every term in $M^s_{ab}$ involves two distinct sites, so

$$ \mathbb E[M^s_{ab}] = S_s\, F_2(a,b), \qquad S_s=\sum_{ij}A^s_{ij}=N z_s. $$

Inserting in (1):

$$ \mathbb E[\alpha^s_{ab}] = -\frac{1}{N-1}\quad (a\ne b), \qquad
   \mathbb E[\alpha^s_{aa}] = \frac{N-n_a}{n_a(N-1)}
   = \frac{1-c_a}{c_a\,(N-1)}. \tag{3} $$

The bias is the same for every lattice and every shell. It is positive for
like pairs, so a perfectly random cell already looks as if like atoms
avoid each other, and it grows as $1/(c_a N)$ for a dilute species.

The bias disappears if the reference probability is that of drawing a
neighbour *without replacement*:

$$ \tilde\alpha^s_{ab} = 1 - \frac{(N-1)\,M^s_{ab}}{z_s\, n_a\,(n_b-\delta_{ab})},
   \qquad \mathbb E[\tilde\alpha^s_{ab}] = 0 \ \text{exactly}. \tag{4} $$

## 3. Covariance

$$ \mathbb E[M^s_{ab}M^t_{cd}] = \sum_{ijkl} A^s_{ij}A^t_{kl}\,
   \mathbb E[x_i^a x_j^b x_k^c x_l^d] . $$

With $i\ne j$ and $k\ne l$, the only possible coincidences are $i=k$,
$i=l$, $j=k$, $j=l$, and the two double coincidences
$(i=k, j=l)$ and $(i=l, j=k)$. Summing the weights of each pattern
requires only three graph invariants,

$$ S_s=\sum_{ij}A^s_{ij},\qquad R_{st}=\sum_i d^s_i d^t_i,\qquad
   P_{st}=\sum_{ij}A^s_{ij}A^t_{ij}, $$

($R_{st}=N z_s z_t$; $P_{ss}$ equals $S_s$ for a simple graph;
$P_{st}=0$ for $s\ne t$ unless the cell is so small that one pair of sites
is a neighbour in two shells). By inclusion–exclusion:

| pattern | total weight | expectation |
|---|---|---|
| all four distinct | $S_sS_t-4R_{st}+2P_{st}$ | $F_4(a,b,c,d)$ |
| $i=k$ only | $R_{st}-P_{st}$ | $\delta_{ac}F_3(a,b,d)$ |
| $i=l$ only | $R_{st}-P_{st}$ | $\delta_{ad}F_3(a,b,c)$ |
| $j=k$ only | $R_{st}-P_{st}$ | $\delta_{bc}F_3(a,b,d)$ |
| $j=l$ only | $R_{st}-P_{st}$ | $\delta_{bd}F_3(a,b,c)$ |
| $i=k,\ j=l$ | $P_{st}$ | $\delta_{ac}\delta_{bd}F_2(a,b)$ |
| $i=l,\ j=k$ | $P_{st}$ | $\delta_{ad}\delta_{bc}F_2(a,b)$ |

so that

$$ \operatorname{Cov}(M^s_{ab},M^t_{cd}) =
   \sum_{\text{patterns}} (\text{weight})(\text{expectation})
   - S_sS_t F_2(a,b)F_2(c,d). \tag{5} $$

The WC parameters are affine in the counts, hence their exact covariance
follows from (5) by the coefficients in (1) or (4). The package evaluates
(5) in rational arithmetic, so no precision is lost in the cancellation
between the two terms.

These pair counts are the multicomponent "join counts" of spatial
statistics; the two-colour case under sampling without replacement is
classical (Cliff and Ord, *Spatial Processes*, 1981). Equation (5)
extends it to several colours, to two different neighbour graphs at once
(cross-shell covariance) and to periodic multigraphs.

## 4. Leading order

Expanding (2) in $1/N$ at fixed $c_a$ gives, with $B_s = N z_s/2$ bonds
in shell $s$,

$$ \operatorname{Var}(\tilde\alpha^s_{aa}) \simeq \frac{(1-c_a)^2}{B_s\, c_a^2},
   \qquad
   \operatorname{Var}(\tilde\alpha^s_{ab}) \simeq
   \frac{(1-c_a)(1-c_b)+c_ac_b}{2B_s\, c_a c_b}\quad(a\ne b), \tag{6} $$

and $\operatorname{Corr}(\alpha^s,\alpha^t)=O(1/N)$ for $s\ne t$. The
lattice enters only through the number of bonds. For a binary alloy the
unlike-pair variance is $1/B_s$ irrespective of composition. Equation (6)
is checked against the exact result to $2\times10^{-4}$ relative accuracy
at $N=2\times10^5$ in the test suite. At $N\lesssim 100$ it overestimates
the exact standard deviation by up to 30 % (Fig. 1b), so the exact form
should be used for small cells.

## 5. Tests

* **Single parameter.** $z = (\tilde\alpha-\mathbb E\tilde\alpha)/\sigma$ with
  the exact $\sigma$. Adequate when the expected number of $a$–$b$ bonds
  is $\gtrsim 10$.
* **Global.** Within a shell only $K(K-1)/2$ parameters are independent
  (row sums of $M$ are fixed). The statistic
  $\chi^2 = \delta^\top \Sigma^{+}\delta$ over the entries $a\le b$, with
  the Moore–Penrose inverse of the exact covariance, is referred to
  $\chi^2$ with $\operatorname{rank}\Sigma$ degrees of freedom. Its
  calibration is checked by simulation (tests, Fig. 4).
* **Dilute species.** When the expected number of bonds is small the
  count is discrete and skewed; use permutation p-values
  (`permutation_pvalues`). The smallest attainable p-value for
  *avoidance* is the null probability of zero $a$–$a$ bonds; if this
  exceeds the significance level, no single configuration of that cell
  can demonstrate avoidance (Fig. 2c).

## 6. Averaging over snapshots

If $\bar\alpha$ is averaged over $n$ configurations, divide the null
covariance by the *effective* number of independent configurations
$n_{\rm eff}=n/(2\tau)$, where $\tau$ is the integrated autocorrelation
time of the Monte Carlo chain measured in snapshot intervals. Using $n$
in place of $n_{\rm eff}$ overstates significance. Note also that
averaging reduces noise but not the bias (3) of the standard estimator,
which is a property of the cell, not of the sampling.
