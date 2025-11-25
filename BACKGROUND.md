# BACKGROUND

Why this repository exists, in plain steps.

---

## 1) The problem we actually care about
You have a contest with N entrants (horses, models, players). You want:

- Permutation‑invariant probabilities for any set of entrants (order doesn’t matter).  
- Coherence when the field changes (scratches/new entrants shouldn’t force a new “kind” of model).  
- A latent ability scale μ so that “how much better” matters (photo finish ≠ blowout).  
- Dead‑heat (tie) handling that matches real markets (fractional payouts).  
- Tractable forward (Ability → Prices) and inverse (Prices → Ability) computation.  

This sounds obvious. Historically, it hasn’t been available in one model.

---

## 2) The old trade‑off: Tractability vs Purity
There is a recurring tension:

- **Thurstone (probit)**: theoretically pure random‑utility; heavy integrals; hard to scale beyond pairs.  
- **Bradley–Terry / Luce (logit/IIA)**: efficient and simple; but IIA is often wrong, and “set effects” break it.  
- **Harville / Plackett–Luce**: fast listwise formulas; but still inherit IIA‑like issues and weak tie handling.  
- **Mallows / RIM**: beautiful distributions on permutations; no latent performance scale; not a contest generator.  

Result: we’ve had to choose between realism and runtime.

> A succinct summary of the landscape  
> – Thurstone: Theoretically “pure,” computationally heavy.  
> – Bradley–Terry/Luce: Computationally cheap, theoretically flawed (IIA).  
> – Harville: Handy shortcut, but assumes repeated IIA‑style selections.  
> – Mallows: Great as a prior on permutations, poor as a performance model.  

There hasn’t been a **consistent, tractable choice model** for multi‑entrant contests that keeps a believable generative story – until now.

---

## 3) The key idea: a lattice for the field, not for each pair
This repository implements the lattice‑based algorithm introduced by Cotton (2021) for multi‑entrant contests.

**Core principles**
- Model each entrant by an arbitrary base performance density (normal, skew‑normal, heavy‑tailed…).  
- Build the distribution of the minimum/maximum of the whole field once (“winner‑of‑many”).  
- Compute each entrant’s marginal win payoff against that fixed field (multiplicity‑aware dead‑heats).  
- Overall cost scales **O(N)** in the number of entrants for a fixed grid size.  

**Why this matters**
- You recover a **latent scale** μ with meaningful gaps.  
- Probabilities are **permutation‑invariant** by construction (depend only on the set).  
- You handle **dead‑heats exactly** (split mass as markets do).  
- You’re **coherent under field changes** (scratch/add entrants, recompute once).  

This is the computational middle road that previously didn’t exist: fast like logit, but with a real field‑based generative story.

---

## 4) Inverse and global calibration (consistent and practical)
Real life gives you prices. You want abilities. The forward map (μ → p̂) is monotone, so we invert it numerically:

- **1D global curve**: build p̂(g) for a shared grid of offsets; invert all prices at once with monotone interpolation.  
- **2D curves (loc, scale)**: precompute a few per‑scale curves; invert at each runner’s scale and interpolate in scale.  
- **Curve caching**: store these lookup curves and **reuse** them across many races (global calibration).  

For many races:

- **Gauss–Newton (curve‑based)**: rebuild curves around current (θ,b), run inner least‑squares/cross‑entropy steps with step‑size decay and update caps, then rebuild again.  
- **Relative‑then‑LS (fast baseline)**: invert each race, center (remove race shift), **slope‑weighted** LS stitch across overlaps.  

Both routes yield a **single coherent ability vector θ**, with optional per‑race intercepts b (capturing translation invariance). The former is fully consistent with the forward physics; the latter is a pragmatic, very fast stitch.

---

## 5) Why this is different from “yet another ranking model”
Other families either assume IIA, avoid ties, or have no notion of “by how much.” Here you get:

- A **latent performance scale** with arbitrary distributions (not fixed to normal/logit).  
- **Exact dead‑heat** allocation on a grid (what markets actually pay out).  
- **Set coherence**: add or remove entrants and stay in the same family with the same interpretation.  
- **O(N)** winner‑of‑many compute for a fixed grid; tractable inverse via monotone interpolation.  

This is the promised “consistent and tractable” choice model for contests.

---

## 6) Beyond sports: why LLM ranking keeps rediscovering these ideas
In LLM alignment, popular objectives (DPO, pairwise) are the **Bradley–Terry of the LLM world**: efficient but pairwise/logit‑based. Newer methods (LiPO) move toward **listwise/distributional objectives**, closer in spirit to a field distribution than to isolated pairs. At the same time, permutation self‑consistency and social‑choice aggregation attack **permutation bias** and **intransitivity** – the same invariance and consistency issues we solve here.

**Prediction**: preference learning for LLMs will continue shifting toward **listwise/field‑based objectives**, echoing what this repository already does for contests.

---

## 7) What you can do with this repository today
- **Forward pricing**: from assumed abilities (and scales), produce permutation‑invariant state prices with dead‑heat support.  
- **Inverse calibration**: from prices, recover implied abilities via monotone interpolation (1D/2D).  
- **Global fitting**: combine many races with cached curves; choose curve‑based GN or fast relative‑LS stitching.  
- **Diagnostics**: check tie mass, permutation invariance, sensitivity to field changes, slope‑based stability.  

Put simply: you can price, invert, and stitch **consistently** and **fast**, without pretending IIA holds or that “rank distance” is a performance model.

---

## 8) The punch line
The repository’s lattice implementation resolves the Tractability‑vs‑Purity dilemma:

- **Thurstone**: pure but heavy.  
- **BT/Luce**: cheap but brittle (IIA).  
- **This lattice**: **computationally efficient (≈ O(N))** and **theoretically flexible** (arbitrary distributions, exact ties, coherent fields).  

By solving the “winner‑of‑many” problem with arbitrary base distributions and dead‑heat sharing, it delivers fidelity that classical shortcuts can’t match – and it provides the inverse and global calibration machinery you need to use that fidelity in practice.

# BACKGROUND

This document summarizes the core background, literature, and research plan for modeling multi‑entrant contests and for connecting classical ranking/choice theory with modern machine‑learning (e.g., large language models, LLMs).

---


## Literature and Background

- Peter Cotton (2021). “Inferring Relative Ability from Winning Probability in Multientrant Contests.” SIAM Journal on Financial Mathematics.  
  Key idea: lattice‑based winner‑of‑many with multiplicity‑aware ties and a monotone inverse map from prices to abilities. Scales to large fields by computing one field distribution and many marginal payoffs.  
  DOI: https://doi.org/10.1137/19M1276261

- Niclas Boehmer, Piotr Faliszewski, Sonja Kraiczy (2024). “Properties of the Mallows Model Depending on the Number of Alternatives: A Warning for an Experimentalist.”  
  Shows that Mallows’ scaling with the number of alternatives can be misleading vs. real data; cautions for experimental design.  
  arXiv: https://arxiv.org/abs/2401.14562 · DOI: https://doi.org/10.48550/arXiv.2401.14562

- Jean‑Paul Doignon, Aleksandar Pekeč, Michel Regenwetter (2004). “The Repeated Insertion Model for Rankings: Missing Link between Two Subset Choice Models.” Psychometrika 69(1), 33–54.  
  Introduces RIM, a permutation generator connecting subset‑choice models; contains Mallows as a special case.  
  DOI: https://doi.org/10.1007/BF02295838 · Link: https://www.cambridge.org/core/journals/psychometrika/article/abs/repeated-insertion-model-for-rankings-missing-link-between-two-subset-choice-models/1E8685C7E25FC47BF4DA392801BAFC9D

- L. L. Thurstone (1927). “A Law of Comparative Judgment.” (Probit pairwise.)  
- F. Mosteller (1951). “Remarks on the Method of Paired Comparisons.”  
- R. A. Bradley, M. E. Terry (1952). “Rank Analysis of Incomplete Block Designs: I.” (Logit/BT.)  
- R. D. Luce (1959). “Individual Choice Behavior.” (IIA; Multinomial Logit.)  
- D. A. Harville (1973). “Assigning Probabilities to the Outcomes of Multi‑Entry Competitions.” (Plackett–Luce racing.)  
- R. L. Plackett (1975). “The Analysis of Permutations.” (Plackett–Luce rankings.)  
- R. R. Davidson (1970). “Extending the Bradley–Terry Model to Accommodate Ties.” (Tie handling.)  
- Elo (1978), Glicko (1999). Practical rating systems informed by BT‑style updates.

---

## Generative AI, Consistency, and Preference Optimization

- Raphael Tang et al. (2023). “Found in the Middle: Permutation Self‑Consistency Improves Listwise Ranking in Large Language Models.”  
  Finds strong positional/permutation biases; proposes marginalizing over input permutations and aggregating (e.g., Kemeny–Young).  
  arXiv: https://arxiv.org/abs/2310.07712

- Nico Potyka et al. (2023). “Robust Knowledge Extraction from Large Language Models using Social Choice Theory.”  
  Aggregates multiple stochastic outputs via social‑choice rules (e.g., Partial Borda) to stabilize rankings.  
  arXiv: https://arxiv.org/abs/2312.14877

- Tianqi Liu et al. (2024). “LiPO: Listwise Preference Optimization through Learning‑to‑Rank.”  
  Trains with listwise (often Plackett–Luce‑style) objectives; a list‑aware generalization of DPO.  
  arXiv: https://arxiv.org/abs/2402.01878

- Xiutian Zhao, Ke Wang, Wei Peng (2024). “Measuring the Inconsistency of Large Language Models in Preferential Ranking.”  
  Documents violations of transitivity and IIA in LLMs; motivates permutation‑invariant, robust aggregation.  
  arXiv: https://arxiv.org/abs/2410.08851

---

## Notes on Mallows (Why not for contests?)

See `MALLOWS_CRITIQUE.md` for a detailed argument. In brief: Mallows is an elegant prior on permutations but lacks a latent performance scale, behaves poorly under field changes, exhibits problematic scaling in the number of alternatives, and does not furnish a credible random‑utility/race‑time generative story for contests.

---

## Appendix: Tasks for Literature Curation (Optional)

1) For key arXiv items, capture title, authors, abstract, and year.  
2) Identify core mathematical frameworks (Plackett–Luce, BT/Thurstone, Mallows/RIM, LTR objectives).  
3) Compare to Cotton (2021) lattice approach; note where axioms (e.g., IIA) fail.  
4) Record tie handling, scalability with N, and any permutation‑invariance claims.  
5) Produce short annotated entries (Title; Authors; Key idea; Relation to contest‑ability inference).


