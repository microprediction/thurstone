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

## Generative AI, Consistency, and Preference Optimization

- Found in the Middle: Permutation Self‑Consistency Improves Listwise Ranking in Large Language Models — Raphael Tang et al. (2023)
  - Reports strong positional/permutation biases in LLM listwise ranking and proposes “permutation self‑consistency” by marginalizing over input permutations to restore permutation‑invariant outputs. Connects to multi‑entrant models where probabilities should be invariant to listing order (e.g., Harville/Plackett–Luce).  
  - arXiv: https://arxiv.org/abs/2310.07712

- Robust Knowledge Extraction from Large Language Models using Social Choice Theory — Nico Potyka et al. (2023)
  - Uses social choice aggregation (e.g., Borda, Condorcet‑style ideas) over multiple LLM samples to reduce stochastic inconsistency in ranked outputs, echoing classical preference aggregation. Highlights how social‑choice‑inspired aggregation can stabilize noisy rankings before probabilistic calibration.  
  - arXiv: https://arxiv.org/abs/2312.14877

- LiPO: Listwise Preference Optimization through Learning‑to‑Rank — Tianqi Liu et al. (2024)
  - Frames preference alignment as listwise learning‑to‑rank (beyond pairwise DPO), often using Plackett–Luce‑like objectives. Reinforces the value of full‑list objectives when optimizing policies against ranked candidates—analogue to multi‑entrant probability fields.  
  - arXiv: https://arxiv.org/abs/2402.01878

- Measuring the Inconsistency of Large Language Models in Preferential Ranking — Xiutian Zhao, Ke Wang, Wei Peng (2024)
  - Empirically evaluates LLMs against axioms such as transitivity and Luce’s IIA, finding frequent violations. Indicates raw LLM‑derived preferences may not satisfy latent‑utility assumptions (Thurstone) without post‑processing or calibration—relevant when mapping model scores to lattice‑based abilities.  
  - arXiv: https://arxiv.org/abs/2410.08851




