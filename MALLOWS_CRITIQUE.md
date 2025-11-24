# Mallows as a Contest Model: A Withering Rebuke

If you care about contests — races, tournaments, “which model is actually better?” — the Mallows model is the wrong tool. It is a beautifully symmetric probability distribution on permutations and a terrible model of how contests are generated.

Below is the “brutal” version of that case.

## 1) No latent performance scale

The canonical Mallows family puts probability on permutations via a central ranking \(\mu_0\), a dispersion parameter \(\phi\in(0,1)\), and a permutation distance \(d\):

\[
P_{\phi,\mu_0,d}(\mu)\ \propto\ \phi^{\,d(\mu_0,\mu)}.
\]

There are no contestant‑specific ability parameters \(w_i\) and no notion of “how much better” one contestant is than another. For a pair, the “better” item (earlier in \(\mu_0\)) is chosen with a fixed probability determined by \(\phi\), independent of the magnitude of any performance gap. A photo‑finish and a blowout mismatch are forced to have the same win probability for the “better” side. That’s not a model of performance; it’s an ordinal regularizer.

## 2) No credible generative story for contests

Contest models worthy of the name are random‑utility or race‑time models: each competitor draws a performance \(X_i=\mu_i+\varepsilon_i\); the winner is \(\arg\max_i X_i\) (or \(\arg\min\) for times). Mallows is nothing like that. It generates permutations by penalizing distance from a central order, or via the **Repeated Insertion Model (RIM)**: start with an empty list and insert each item at a random position with probabilities tied to distance from \(\mu_0\). RIM provides a clean permutation generator and includes Mallows as a special case, but it is not a model of produced performances or margins of victory. See the Psychometrika paper on RIM for details ([Doignon–Pekeč–Regenwetter, 2004](https://www.cambridge.org/core/journals/psychometrika/article/abs/repeated-insertion-model-for-rankings-missing-link-between-two-subset-choice-models/1E8685C7E25FC47BF4DA392801BAFC9D), DOI: https://doi.org/10.1007/BF02295838).

## 3) Not coherent under changing fields (scratches/entries)

In real contests, entrants come and go. If you remove an interior contestant from a Mallows distribution, the induced distribution on the survivors is generally **not** Mallows with the same \(\phi\) and a trivially truncated \(\mu_0\). Certain prefix truncations preserve form, but arbitrary scratches do not. That means there is no stable, global parameterization that remains coherent as the field composition changes — exactly the regime contests live in.

## 4) Pathologies as the number of entrants grows

When the number of alternatives \(n\) grows at fixed \(\phi\), Mallows exhibits scaling behaviors that **do not** match real data. A recent systematic study shows that rankings concentrate around the central order in ways that confound experimental interpretation and diverge from observed phenomena. The authors explicitly warn experimentalists about using Mallows as \(n\) varies ([Boehmer–Faliszewski–Kraiczy, 2024](https://arxiv.org/abs/2401.14562), DOI: https://doi.org/10.48550/arXiv.2401.14562). In contests, field size constantly changes; a dispersion parameter whose meaning drifts with \(n\) is not measuring a stable “uncertainty in performance.”

## 5) It isn’t even computationally nice (except in special cases)

Mallows looks tidy, but convenience is limited to specific distances (e.g., Kendall/Cayley/Hamming). Outside those, normalization and inference can be costly; even with simple distances, finding consensus (Kemeny‑style MLE) is NP‑hard, and computing subset choice probabilities \(P(i\text{ wins}\mid S)\) at scale is nontrivial. For contest analytics and assortment‑style queries, you often need specialized solvers or approximations. You end up with **both** an unrealistic behavioral story **and** nontrivial compute.

## 6) The ordinal shell problem

Some modern “Mallows‑flavored” objectives bolt on extra surrogates (e.g., sigmoids or listwise losses) to inject sensitivity to score differences. That can be pragmatically useful, but it is no longer Mallows as a generative model; it’s an ordinal shell wrapped around a different (often logit‑like) mechanism precisely because contest behavior needs **cardinal** performance gaps.

## 7) When Mallows is fine — and when it is not

Mallows is a lovely **prior on permutations**. It’s ideal when you want to regularize rankings around a nominal order; you don’t care about the generative mechanism; and ranks are just symbols. There’s a rich literature on distances, mixtures, and Bayesian variants.

For contests, where you really mean:

- each competitor has a latent ability;  
- they draw noisy performances;  
- **win probabilities reflect ability gaps**;  
- the field’s composition changes frequently;  

…Mallows is simply the wrong abstraction. It lacks ability parameters, ignores margins, misbehaves when contestants are added/removed, scales poorly in \(n\), and isn’t a universal computational panacea.

If you want a realistic contest model, use a **random‑utility / race‑time** framework (probit/logit/mixtures, or lattice‑based performance models). These are built on latent performance scales with noise, naturally yield permutation‑invariant probabilities for sets of entrants, and remain coherent as the field changes. They model contests as contests — not as decorative perturbations of a single ranking.

---

### References

- Doignon, J.‑P., Pekeč, A., Regenwetter, M. (2004). The Repeated Insertion Model for Rankings: Missing Link between Two Subset Choice Models. *Psychometrika*, 69(1), 33–54. Cambridge University Press. [Link](https://www.cambridge.org/core/journals/psychometrika/article/abs/repeated-insertion-model-for-rankings-missing-link-between-two-subset-choice-models/1E8685C7E25FC47BF4DA392801BAFC9D). DOI: https://doi.org/10.1007/BF02295838
- Boehmer, N., Faliszewski, P., Kraiczy, S. (2024). *Properties of the Mallows Model Depending on the Number of Alternatives: A Warning for an Experimentalist.* arXiv:2401.14562. [arXiv](https://arxiv.org/abs/2401.14562). DOI: https://doi.org/10.48550/arXiv.2401.14562


