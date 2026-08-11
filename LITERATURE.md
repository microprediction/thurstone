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

## Inversion, Convex Duality, and Identifiability

The discrete-choice econometrics literature contains, in pieces, a global inverse theory for the map from abilities (utilities) to winning (choice) probabilities. These references pin down the novelty boundary for any theoretical claims built on the ability transform: the qualitative diffeomorphism statement is essentially assembled from the entries below, whereas the explicit Laplacian structure of the Jacobian for independent translated noise (see `thurstone/laplacian.py`), quantitative conditioning via the spectral gap, boundary asymptotics, dead-heat/tie handling, and near-linear Newton–CG inversion are not treated there.

- Econometric Models of Probabilistic Choice — Daniel McFadden (1981, in *Structural Analysis of Discrete Data*, Manski & McFadden eds., MIT Press)
  - The social-surplus (expected maximum utility) function Ψ and the gradient representation ∇Ψ = p for additive random utility models. The symmetry and translation-invariance restrictions on ∂p_i/∂a_j are the Williams–Daly–Zachary conditions. This is the classical source for the "choice probabilities are a convex gradient map" fact.

- Conditional Choice Probabilities and the Estimation of Dynamic Models — V. Joseph Hotz and Robert A. Miller (1993, Review of Economic Studies 60(3):497–529)
  - The Hotz–Miller inversion: choice probabilities can be inverted back to utility differences. The starting point for the injectivity half of the global inverse theorem, and for the large CCP-estimation literature that consumes such inversions.

- Connected Substitutes and Invertibility of Demand — Steven Berry, Amit Gandhi, Philip Haile (2013, Econometrica 81(5):2087–2111)
  - General injectivity theorem: the demand/choice-probability map is invertible under a "connected substitutes" condition, covering ARUM as a special case. The definitive reference for uniqueness of the ability vector (up to translation) given interior probabilities.
  - DOI: https://doi.org/10.3982/ECTA10135

- On the Surjectivity of the Mapping between Utilities and Choice Probabilities — Andriy Norets and Satoru Takahashi (2013, Quantitative Economics 4(1):149–155)
  - Shows that when the additive utility shocks have full-support density, the map from utilities to choice probabilities is onto the interior of the simplex. This is exactly the boundary/properness argument needed to upgrade local invertibility to a global diffeomorphism T: 1⊥ → int Δ^{n−1}.
  - DOI: https://doi.org/10.3982/QE252

- Duality in Dynamic Discrete-Choice Models — Khai Xiang Chiong, Alfred Galichon, Matthew Shum (2016, Quantitative Economics 7(1):83–115)
  - Develops the convex-conjugate view: the inverse map is the gradient of the conjugate of the social surplus, T^{-1} = ∇Ω, computable via mass-transport/linear-programming methods. The natural reference for the "generalized entropy" interpretation of the ability transform.
  - DOI: https://doi.org/10.3982/QE436

- Discrete Choice and Rational Inattention: A General Equivalence Result — Mogens Fosgerau, Emerson Melo, André de Palma, Matthew Shum (2020, International Economic Review 61(4):1569–1589)
  - Uses generalized entropy functions (conjugates of surplus functions) to establish an equivalence between ARUM and rational-inattention models. Source of the "generalized entropy" terminology; for Gumbel noise the entropy is Shannon's, for other noise (including Thurstone/probit) it is the corresponding non-logit mirror map.
  - DOI: https://doi.org/10.1111/iere.12469

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




