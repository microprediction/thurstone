# Literature and Background

- Inferring Relative Ability from Winning Probability in Multientrant Contests — Peter Cotton (2021, SIAM Journal on Financial Mathematics)
  - Key idea: a fast lattice-based algorithm that links latent performance distributions to multi-entrant winning probabilities. Introduces the “winner-of-many” construction, a multiplicity-aware tie treatment (dead-heat sharing), and a monotone interpolation-based inverse mapping from prices to abilities. Scales to very large N by computing one field distribution and many marginal payoffs.
  - DOI: https://doi.org/10.1137/19M1276261

- Properties of the Mallows Model Depending on the Number of Alternatives: A Warning for an Experimentalist — Niclas Boehmer, Piotr Faliszewski, Sonja Kraiczy (2024)
  - Examines how the classical Mallows model’s behavior changes with the number of alternatives, showing empirical and theoretical divergences from real-world ranking data. Highlights pitfalls for experimental design and points to a recent variant (Boehmer et al., 2021) that better matches observed phenomena. Useful context when using Mallows- or Plackett–Luce–style components for ranking or score modeling alongside probabilistic winner models.
  - arXiv: https://arxiv.org/abs/2401.14562 · DOI: https://doi.org/10.48550/arXiv.2401.14562

- The Repeated Insertion Model for Rankings: Missing Link between Two Subset Choice Models — Jean‑Paul Doignon, Aleksandar Pekeč, Michel Regenwetter (2004, Psychometrika)
  - Introduces the Repeated Insertion Model (RIM), a probabilistic ranking model connecting subset choice frameworks. RIM is a special case of Marden’s orthogonal contrast family and subsumes the Mallows φ‑model as a special case. It provides a bridge between latent scale and size‑independent choice models and clarifies relationships among ranking and choice generative processes.
  - Journal: Psychometrika 69(1):33–54 · DOI: https://doi.org/10.1007/BF02295838 · Link: https://www.cambridge.org/core/journals/psychometrika/article/abs/repeated-insertion-model-for-rankings-missing-link-between-two-subset-choice-models/1E8685C7E25FC47BF4DA392801BAFC9D

- A Law of Comparative Judgment — L. L. Thurstone (1927)
  - Classical foundation for pairwise comparison models with latent normal utilities. Winning probabilities arise from differences of normal variables (probit link). Forms the conceptual basis for Thurstone–Mosteller models used in rating and ranking.

- Remarks on the Method of Paired Comparisons — Frederick Mosteller (1951)
  - Clarifies and extends Thurstone’s formulation; connects to practical estimation and inference for probit-style paired comparisons.

- Rank Analysis of Incomplete Block Designs: I — R. A. Bradley and M. E. Terry (1952)
  - The Bradley–Terry model (logistic/“logit” alternative to Thurstone’s probit). Provides a widely used parametric form for paired comparisons; basis for many modern rating methods.

- Individual Choice Behavior: A Theoretical Analysis — R. Duncan Luce (1959)
  - The Luce choice axiom (IIA) yields the multinomial logit model for multi-alternative selection. In the context of horse racing or multi-entrant contests, it motivates proportional allocation rules and softmax-like transforms.

- Assigning Probabilities to the Outcomes of Multi-Entry Competitions — D. A. Harville (1973)
  - Classical racing model for translating abilities into finish probabilities across multiple entrants. Provides a benchmark for multi-entrant probability assignment and ranking, often compared with paired-comparison approaches.

- The Analysis of Permutations — R. L. Plackett (1975)
  - Introduces the Plackett–Luce ranking model for full permutations based on Luce’s axiom. Important when modeling ordered finishes (not just the winner).

- Extending the Bradley–Terry Model to Accommodate Ties — R. R. Davidson (1970)
  - Allows explicit ties within Bradley–Terry-like frameworks. Relevant to dead-heat and tie handling in betting or competitive settings.

- Elo (1978) and Glicko (1999) rating systems
  - Practical rating systems derived from paired-comparison ideas. While not lattice-based, they inform how to update abilities online from outcomes.

## How these relate to this repository

- This repository implements a lattice-based forward-and-inverse framework for multi-entrant contests inspired by Cotton (2021). The forward model constructs the distribution of the minimum (winner) across entrants and accounts for multiplicity (dead-heat sharing). The inverse model uses monotone interpolation of an implicit ability→price map to recover abilities from prices.
- Thurstone–Mosteller, Bradley–Terry, and Luce models provide complementary probabilistic foundations for pairwise or multi-way choice. They inform simplified surrogates (e.g., relative LS stitching or Luce-style group allocation) used for diagnostics, initialization, and fast global calibration.
- Harville, Plackett–Luce, and Davidson connect to multi-entrant or ranking-oriented probability assignments, which are useful for cross-checks and extensions beyond “winner-only” pricing (e.g., place/show or full-order likelihoods).


https://arxiv.org/abs/2310.07712

https://arxiv.org/html/2312.14877v2

https://arxiv.org/html/2410.08851v1

https://arxiv.org/html/2402.01878v3



