# The Thurstone Mirror Map: Fast Inversion, Generalized Entropy and Multiway Choice Geometry

- **Status**: draft (for review)
- **Authors**: Peter Cotton
- **Started**: 2026-08-12

## Abstract

The map from latent abilities to winning probabilities in a multi-entrant
Thurstone contest is a convex gradient map whose Jacobian, for independent
translated noise, is a weighted complete-graph Laplacian. The paper (i)
assembles a global diffeomorphism theorem from known injectivity
(Hotz–Miller; Berry–Gandhi–Haile) and surjectivity (Norets–Takahashi)
results, contributing the quantitative side — explicit Laplacian weights,
spectral-gap conditioning, matrix-tree determinant, boundary degeneracy;
(ii) identifies the convex conjugate of the expected-winner surplus as a
generalized entropy whose gradient is the inverse map, making the ability
transform a non-logit mirror map; and (iii) gives an O(nM) Hessian-vector
product with an exact repair scheme for degenerate fields, a renormalization
lemma (matching probability ratios has Jacobian exactly −L because 1ᵀL = 0),
and the resulting near-linear Newton–CG joint inversion, described in full
detail in three appendices. Experiments use the `thurstone` package.

## Files

- `paper.tex` — manuscript (build with `../build.sh mirror-map`)
- `figures/` — figures used in this paper
- shared bibliography: `../refs.bib`

## Notes for revision

- Boundary asymptotics of λ₂ are stated qualitatively; sharp constants are
  listed as future work.
