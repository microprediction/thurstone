# Research ideas ledger

Parking lot for ideas not yet worked out, so they survive across machines
and sessions. Date-stamped; promote to a paper section only once actually
worked out.

## 1. Maximally symmetric cube-to-simplex diffeomorphism (2026-08-12)

Dropped from the mirror-map paper as speculative; recorded here.

- **Question.** Is there a canonical "most symmetric" diffeomorphism
  [0,1]^k → Δ^k?
- **Candidate answer.** The Brenier optimal-transport map from the uniform
  measure on the cube to the uniform measure on the simplex. It is the
  gradient of a convex potential (so itself a mirror map, on-theme), pushes
  volume to volume exactly, and — because the quadratic-cost transport map
  between measures invariant under a common group action is unique — it is
  automatically equivariant under the shared Sym(k): permutations of the k
  cube axes correspond to permutations of the k non-reference simplex
  vertices.
- **Where the Thurstone construction fits.** Mapping k coordinates onto a
  k-simplex requires *adding one reference horse* (cf. `CubeToSimplexMapping`
  and DIFFEOMORPHISMS.md). That gives free knobs beyond the homogeneous
  independent family: the reference horse's noise variance, and an
  exchangeable correlation among the k coordinate horses (machinery now in
  `thurstone/correlated.py`). Any treatment symmetric in the k axes
  preserves equivariance, so these knobs parametrize an equivariant family
  to optimize toward volume uniformity.
- **Technical note.** The Laplacian Jacobian structure survives correlation:
  for general ARUM the Williams–Daly–Zachary conditions still give a
  symmetric substitutes matrix with zero row sums (a graph Laplacian). Only
  the product-form weights and the O(nM) matvec trick require independence.
- **Concrete next steps.** (a) Compute the transport cost of the best
  Thurstone-family map against the Brenier optimum numerically (small k);
  (b) ask whether some noise law makes the Thurstone construction exactly
  optimal; (c) revisit the quality metrics in the diffeomorphism module
  (symmetry / volume preservation) as estimates of these quantities.

## 2. Sharp boundary asymptotics for the spectral gap (2026-08-12)

The mirror-map paper proves inverse conditioning ~ 1/λ₂ and observes
λ₂ → 0 at the simplex boundary. The overlap integrals w_ij decay at
explicit noise-dependent rates (Gaussian-tail in the Gaussian case), so
sharp constants should be attainable, giving a uniform conditioning theory
on boundary neighbourhoods. Listed as future work in the paper.

## 3. Wire joint Newton–CG into global calibration (2026-08-12)

`invert_outright_probabilities` currently inverts one race. The global and
multiray calibrators still use diagonal cached-curve Gauss–Newton. The
two-stage plan: per-race exact inversion (T⁻¹) then linear/bilinear
factorization across races — the multidimensional model
a_{ri} = v_r^T z_i separates cleanly after inversion.

## 4. Paper 2 skeleton: multidimensional Thurstone on contest hypergraphs

Known-direction identifiability is exactly
span{(e_i − e_j) ⊗ v_r : i,j ∈ H_r} = 1⊥ ⊗ R^d (necessary and
sufficient; identifies Z up to one common translation). Latent-direction
case has a genuine GL(d) gauge (z ↦ Az, v ↦ A⁻ᵀv); norm constraints
reduce it — the residual gauge should be theoremized, not hand-waved.
A race-common additive bias b_r is pure gauge (T(a+b1) = T(a)) and never
estimable from winner probabilities.

## 5. Observation: Newton rate equals Jacobian discretization error

Measured contraction 0.034/step matches the ~3% finite-difference mismatch
of the analytic Laplacian vs. the lattice map's derivative — a consistency
check now recorded in the mirror-map paper §7. Refining the lattice
steepens Newton convergence along with the quadrature.

## State of play (2026-08-12)

- Merged to main: Laplacian module + exact-repair matvec + LaplacianOperator
  + ratio-matching Newton–CG inversion + demo + docs (PRs #13, #15, #16,
  #17); correlated races module (PR #14, separate line of work).
- Open for review: PR #18 — papers/mirror-map (18pp draft; first review
  pass applied: linear-not-superlinear convergence fix, speculative remark
  dropped, margin overflows fixed).
- Local-only on the Mac (docs/book is gitignored): b_r gauge corrections in
  chapters 11/13 and a new "Exact joint Newton via the Laplacian Jacobian"
  subsection in chapter 11. These do NOT travel with the repo.
