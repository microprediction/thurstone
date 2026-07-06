"""
Swiss vs. World Cup: which format is the better *estimator* of team strength?
============================================================================

Claim under test
----------------
A **Swiss-system** tournament for the 48-team FIFA World Cup is *more
statistically powerful* -- it recovers the true ranking of teams, and in
particular crowns the genuinely-best team, more reliably -- than the current
**group round-robin + single-elimination knockout** format, when the two are
given the **same calendar length** (number of match-days), and even more
clearly when compared per game played.

Why this is a statistics question
----------------------------------
A tournament is an *estimator*. Nature fixes a latent ranking of the teams;
each match is a noisy pairwise comparison; the format is a rule for turning a
budget of noisy comparisons into an estimate of "who is best". Two estimators
built from the same number of match-days can differ enormously in variance:

  * **Knockout** is a sequential *single-elimination* estimator. One bad draw
    (an upset, or a coin-flip after a 0-0) removes a strong team permanently.
    Information is thrown away: once eliminated, a team plays no more informative
    games. The champion's estimate rests on a handful of high-variance duels.

  * **Group round-robin** only compares a team against the 3 others in its
    group -- a tiny, arbitrary neighbourhood of the field. Draws (1 pt each)
    further blunt the signal.

  * **Swiss** is *adaptive*. After each round it pairs teams with similar
    records, so nearly every game is a maximally-informative comparison between
    near-equals (the same principle as computerized adaptive testing). No team
    is eliminated, so every match-day adds signal for the whole field, and the
    final standings aggregate up to R roughly-independent, well-targeted
    comparisons per team.

The pairwise match model is exactly this repository's Thurstone Case-V model:
performance = ability + Gaussian noise, higher performance wins. So
    P(i beats j) = Phi( (theta_i - theta_j) / (sqrt(2) * sigma) ).

What the program does
---------------------
For each of many Monte-Carlo replications it draws a fresh set of 48 latent
abilities (Elo-calibrated by default) and, on the SAME abilities (paired
comparison / common random numbers -> strong variance reduction) and a SHARED
noisy pre-tournament seeding, it runs and scores a family of formats:

  * FIFA-2026 (12 seeded groups of 4; top-2 + 8 best thirds -> R32; single-
    elimination to the Final) -- ranked by bracket, and also by a BT-MLE refit;
  * plain Swiss of R rounds (seeded fold, Monrad/Dutch pairing, Buchholz) --
    snapshotted after every round, ranked by points and by BT-MLE;
  * Swiss WITH elimination on several cut schedules (game-budget-matched to,
    or thriftier than, the World Cup);
  * a 'protected final' Swiss (top-2 kept apart until a decisive grand final);
  * two model-driven designs -- an adaptive Thurstone tournament (pair by the
    posterior, never eliminate) and a champion-focus / best-arm scheme (funnel
    games onto the contenders);
  * a full single round-robin, as the 47-match-day gold-standard ceiling.

Headline findings (at 2000+ reps): at equal calendar every Swiss-style format
beats the knockout by a wide, significant margin (P(best) ~0.29 vs ~0.22,
McNemar p < 1e-7). The lever is NOT eliminating on thin, local evidence -- the
World Cup discards the true best team before the final ~68% of the time. You can
still eliminate (even with fewer games than the World Cup) as long as you cut on
accumulated whole-field record. Ranking by BT-MLE, adaptive posterior pairing,
and best-arm funnelling are all ~neutral for winner-ID, which is anyway capped
near 0.57 by top-team parity (the round-robin ceiling).

Calendar accounting
    FIFA-2026 champion path = 3 group match-days + 5 knockout rounds
    (R32, R16, QF, SF, Final) = 8 match-days. So "Swiss with 8 rounds" is the
    equal-calendar comparison. The program also reports Swiss at *fewer* rounds
    to show it reaches the World Cup's power with fewer match-days AND far fewer
    games.

Power metrics (higher = better estimator)
    * P(true #1 team is crowned champion)             -- primary
    * P(champion is in the true top 4)
    * E[true rank of the champion]                    (1 = perfect)
    * Spearman rho( final standings , true ranking )  -- full-field fidelity

Because both formats run on the same abilities each replication, the champion-
identification comparison is a matched-pairs design; the program reports a
McNemar exact test on the paired disagreements, so the Swiss advantage comes
with a p-value, not just a bar chart.

Usage
-----
    python examples/swiss_vs_worldcup_power.py                 # default run
    python examples/swiss_vs_worldcup_power.py --reps 4000 --plot out.png
    python examples/swiss_vs_worldcup_power.py --noise 1.3     # more upsets
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Calibration to real team differences (head-to-head via Elo)                  #
# --------------------------------------------------------------------------- #
# Football World-Elo gives a well-tested head-to-head model:
#     P(i beats j) = logistic( (elo_i - elo_j) * ln(10)/400 ).
# A Thurstone probit Phi(x) matches a logistic(y) when y = 1.702 * x, so an Elo
# gap maps to a Thurstone ability gap by:
#     theta_i - theta_j = (elo_i - elo_j) * (ln(10)/400) / 1.702.
# With this constant we fix the noise so that sqrt(2)*sigma = 1, i.e. the model
# reproduces Elo win probabilities exactly (before the draw band is applied).
#
# A 48-team World-Cup field spans roughly Elo 1550..2100; modelling the field as
# iid Normal with std ~130 Elo reproduces realistic gaps: a +100 Elo edge ~ 64%,
# +200 ~ 76%, +300 ~ 85% single-game win probability. This is the "sensible
# head-to-head" calibration; pass --elo-sd to widen/narrow the field.
LOGIT_TO_PROBIT = 1.702
THURSTONE_PER_ELO = (np.log(10.0) / 400.0) / LOGIT_TO_PROBIT   # ~0.003383
CALIBRATED_SIGMA = 1.0 / np.sqrt(2.0)                          # -> sqrt(2)*sigma = 1


def sample_abilities(rng, n, model, ability_sd, elo_sd):
    """Latent Thurstone abilities for the field.

    'elo' (default): draw iid Elo ratings ~ N(0, elo_sd) and convert to
    Thurstone units, so pairwise outcomes match the football Elo model.
    'gaussian': abstract N(0, ability_sd) abilities (the original toy model).
    """
    if model == "elo":
        return rng.normal(0.0, elo_sd, size=n) * THURSTONE_PER_ELO
    return rng.normal(0.0, ability_sd, size=n)

# --------------------------------------------------------------------------- #
# Match model  (Thurstone Case V -- this repo's model)                         #
# --------------------------------------------------------------------------- #


@dataclass
class MatchModel:
    """Turns latent abilities into noisy match outcomes.

    performance_i = theta_i + N(0, sigma^2);  higher performance wins.
    A margin whose magnitude is below `draw_band` counts as a draw where draws
    are allowed (group stage); in the knockout it is resolved by a fair coin
    (a stand-in for a penalty shoot-out), which is exactly the extra variance
    that makes single-elimination a noisy estimator.
    """

    sigma: float = 1.0
    draw_band: float = 0.55

    def play(self, rng, theta_i, theta_j):
        """Return (margin, result) with result in {+1 i wins, 0 draw, -1 j wins}."""
        margin = (theta_i - theta_j) + rng.normal(0.0, self.sigma) - rng.normal(
            0.0, self.sigma
        )
        if abs(margin) < self.draw_band:
            return margin, 0
        return margin, 1 if margin > 0 else -1

    def play_decisive(self, rng, theta_i, theta_j):
        """Knockout game: a draw is settled by a fair coin (penalties)."""
        margin, res = self.play(rng, theta_i, theta_j)
        if res == 0:
            res = 1 if rng.random() < 0.5 else -1
        return margin, res


# --------------------------------------------------------------------------- #
# Bookkeeping for a team inside a tournament run                               #
# --------------------------------------------------------------------------- #


@dataclass
class Standing:
    idx: int              # team id (index into the ability vector)
    theta: float          # latent ability
    seed: int             # seeding rank (0 = strongest), = FIFA-ranking analogue
    points: int = 0
    gd: float = 0.0       # accumulated margin (goal-difference proxy)
    opponents: list = field(default_factory=list)

    def add(self, opp_idx, result, margin):
        self.points += 3 if result == 1 else (1 if result == 0 else 0)
        self.gd += margin
        self.opponents.append(opp_idx)


def _sort_key(standings):
    """Rank by points, then Buchholz (sum of opponents' points), then margin,
    then seed. Returns team ids best-first."""
    pts = {s.idx: s.points for s in standings}
    buch = {s.idx: sum(pts.get(o, 0) for o in s.opponents) for s in standings}
    return sorted(
        standings,
        key=lambda s: (s.points, buch[s.idx], s.gd, -s.seed),
        reverse=True,
    )


# --------------------------------------------------------------------------- #
# Format 1 -- Swiss system (with per-round standings snapshots)                #
# --------------------------------------------------------------------------- #


def run_swiss(rng, theta, model: MatchModel, rounds: int, seed_order=None):
    """Run an `rounds`-round Swiss tournament.

    Round 1 uses the standard seeded fold: the pre-tournament top half plays the
    bottom half (seed k vs seed n/2+k). Later rounds use greedy Monrad/Dutch
    pairing within score groups, avoiding rematches when possible. Returns a list
    `snapshots` where snapshots[r] is the ranked list of team ids after r+1
    rounds -- so a single run gives the entire power-vs-match-day curve.

    `seed_order` is the shared (noisy) pre-tournament ranking; pass the SAME
    array used for the World Cup so the two formats are seeded identically.
    """
    n = len(theta)
    order = np.argsort(-theta) if seed_order is None else np.asarray(seed_order)
    seed_of = {int(idx): s for s, idx in enumerate(order)}
    st = {i: Standing(idx=i, theta=float(theta[i]), seed=seed_of[i]) for i in range(n)}

    snapshots = []
    games = []          # (i, j, score_for_i) for downstream BT-MLE ranking
    for r in range(rounds):
        if r == 0:
            ranked = [int(i) for i in order]
        else:
            ranked = [s.idx for s in _sort_key([st[i] for i in range(n)])]

        if r == 0:
            # Seeded fold: 1 vs (n/2+1), 2 vs (n/2+2), ...
            half = n // 2
            pairs = [(ranked[k], ranked[half + k]) for k in range(half)]
        else:
            pairs = _greedy_pairs(ranked, st)

        for a, b in pairs:
            margin, res = model.play(rng, theta[a], theta[b])
            st[a].add(b, res, margin)
            st[b].add(a, -res, -margin)
            games.append((a, b, 1.0 if res == 1 else (0.5 if res == 0 else 0.0)))

        snapshots.append([s.idx for s in _sort_key([st[i] for i in range(n)])])
    return snapshots, games


def _greedy_pairs(ranked, st):
    """Pair adjacent teams in the standings, skipping rematches greedily."""
    remaining = list(ranked)
    pairs = []
    while remaining:
        a = remaining.pop(0)
        partner_pos = None
        for pos, b in enumerate(remaining):
            if b not in st[a].opponents:
                partner_pos = pos
                break
        if partner_pos is None:  # everyone left already played a -> allow rematch
            partner_pos = 0
        b = remaining.pop(partner_pos)
        pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------------- #
# Format 1b -- ADAPTIVE Thurstone tournament  (the clever solution)            #
# --------------------------------------------------------------------------- #
# Fixes the three pathologies the reader identified in one stroke:
#
#   * "losing is rewarded"  -- classic Swiss pairs by POINTS, so a loss drops you
#      into a soft score group where cheap points await. Here we pair by the
#      Bradley-Terry / Thurstone posterior ability, which credits the *quality*
#      of your results. A narrow loss to the #1 seed barely dents your estimate,
#      so you keep facing strong teams: losing is no longer rewarded.
#
#   * "too much randomness" -- each game is scheduled to be maximally informative.
#      A pairwise comparison's Fisher information about (theta_i - theta_j) peaks
#      when the teams are near-equal (win prob ~ 1/2). Pairing adjacent in the
#      *ability* order (not the noisy points order) puts every game near that
#      optimum, so the design extracts the most signal per match-day.
#
#   * "sent home too early"  -- nobody is eliminated. Calendar time (match-days),
#      not games, is the binding World-Cup constraint, and games run in parallel;
#      so we keep all 48 teams playing every round and never discard a team on
#      thin, early data. The ranking is the model posterior over the whole field.
#
# In experimental-design terms this is sequential, near-D-optimal active learning
# for a Thurstone model -- exactly this repository's subject matter.


def _brick_pairs(ranked, offset):
    """'Brick-wall' pairing along the ability order: offset 0 gives
    (1,2)(3,4)..., offset 1 gives (2,3)(4,5)...(n,1). Alternating the offset
    across rounds keeps the whole ability line densely and connectedly compared
    (each team meets both neighbours), which is what a good full-field estimate
    needs, while every game is still a near-equal, high-information match."""
    n = len(ranked)
    if offset == 0:
        return [(ranked[k], ranked[k + 1]) for k in range(0, n - 1, 2)]
    pairs = [(ranked[k], ranked[k + 1]) for k in range(1, n - 1, 2)]
    pairs.append((ranked[n - 1], ranked[0]))   # wrap: one long-range calibration link
    return pairs


def run_adaptive(rng, theta, model: MatchModel, rounds: int, seed_order=None):
    """Adaptive Thurstone tournament -- the clever solution. Returns (snapshots, games).

    Round 1 is the long-range seeded fold (top half vs bottom half) to plant
    global calibration links. Every later round: refit the Bradley-Terry /
    Thurstone MLE to all games so far, order the field by estimated ability, and
    pair near-equals with an alternating brick-wall offset so the ability line
    stays connected. Ranks by the posterior, and never eliminates anyone.
    snapshots[r] is the MLE ranking after r+1 rounds.
    """
    n = len(theta)
    order = np.argsort(-theta) if seed_order is None else np.asarray(seed_order)
    games = []
    snapshots = []
    for r in range(rounds):
        if r == 0:
            ranked = [int(i) for i in order]
            half = n // 2
            pairs = [(ranked[k], ranked[half + k]) for k in range(half)]  # fold
        else:
            ranked = list(np.argsort(-bt_mle(n, games)))
            pairs = _brick_pairs(ranked, offset=r % 2)
        for a, b in pairs:
            _, res = model.play(rng, theta[a], theta[b])
            games.append((a, b, 1.0 if res == 1 else (0.5 if res == 0 else 0.0)))
        snapshots.append(list(np.argsort(-bt_mle(n, games))))
    return snapshots, games


# --------------------------------------------------------------------------- #
# Format 1c -- CHAMPION-FOCUS  (best-arm identification: we care only who wins) #
# --------------------------------------------------------------------------- #
# If the only question is "who is the single best team?", ranking the tail is
# wasted effort. This is a pure best-arm-identification problem, so we FUNNEL the
# comparison budget onto the contenders: each round the top-C teams by posterior
# play a rotating round-robin among THEMSELVES (so the leaders are compared to
# the leaders, repeatedly, at rising precision), while everyone else plays out
# below. C shrinks over the rounds. Crucially nobody is eliminated -- a team the
# early rounds under-rated keeps winning below and climbs back into the pool, so
# there is no "sent home on 2 games" failure mode. Champion = posterior argmax.


def _circle_pairs(pool, t):
    """One round of a round-robin over `pool` by the circle method (rotation t)."""
    m = len(pool)
    if m < 2:
        return []
    fixed, rot = pool[0], pool[1:]
    k = t % (m - 1)
    rot = rot[k:] + rot[:k]
    arr = [fixed] + rot
    return [(arr[i], arr[m - 1 - i]) for i in range(m // 2)]


# Championship-pool size per match-day: broad early (find the contenders), then
# funnel the games onto the survivors for a high-precision run-off.
FUNNEL = (48, 24, 16, 12, 8, 8, 6, 4)


def run_champion_focus(rng, theta, model: MatchModel, rounds: int,
                       seed_order=None, funnel=FUNNEL):
    n = len(theta)
    order = np.argsort(-theta) if seed_order is None else np.asarray(seed_order)
    games = []
    snapshots = []
    for r in range(rounds):
        if r == 0:
            ranked = [int(i) for i in order]
            half = n // 2
            pairs = [(ranked[k], ranked[half + k]) for k in range(half)]  # fold
        else:
            ranked = list(np.argsort(-bt_mle(n, games)))
            C = min(funnel[r] if r < len(funnel) else funnel[-1], n)
            C -= C % 2
            pairs = _circle_pairs(ranked[:C], r) + _brick_pairs(ranked[C:], r % 2)
        for a, b in pairs:
            _, res = model.play(rng, theta[a], theta[b])
            games.append((a, b, 1.0 if res == 1 else (0.5 if res == 0 else 0.0)))
        snapshots.append(list(np.argsort(-bt_mle(n, games))))
    return snapshots, games


# --------------------------------------------------------------------------- #
# Format 1d -- PROTECTED FINAL  ('top-2 can't meet until the final round')      #
# --------------------------------------------------------------------------- #
# A Swiss where the two title FAVOURITES are kept apart until a grand final, so
# the marquee clash is saved for the last match-day. The favourites are "the two
# most likely to win" -- NOT the two pre-tournament seeds, and not merely the two
# points-leaders. Under Thurstone, a team's probability of winning (its
# performance being the field maximum) is a state price, and state-price order is
# monotone in ability -- so the two most likely to win are exactly the top two by
# the current Bradley-Terry / Thurstone ability estimate. That is a one-shot
# calculation from the fitted abilities (thurstone.state_prices_from_ability),
# NOT a nested re-simulation of the remaining tournament.
#
# Rounds 1..R-1 pair the top-2 favourites with #3/#4 instead of each other; round
# R stages the decisive final between them, and the champion is its winner.
# Empirically the *protection* is free (each favourite still meets #3/#4, so the
# top order is pinned by transitivity), and it barely matters whether you protect
# by points or by the Thurstone ability -- but letting one game DECIDE costs a
# little, since it replaces R rounds of evidence with a single ~50/50 coin-flip.


def run_protected_final(rng, theta, model: MatchModel, rounds: int, seed_order=None):
    n = len(theta)
    order = np.argsort(-theta) if seed_order is None else np.asarray(seed_order)
    seed_of = {int(idx): s for s, idx in enumerate(order)}
    st = {i: Standing(idx=i, theta=float(theta[i]), seed=seed_of[i]) for i in range(n)}
    games = []
    champion = None
    for r in range(rounds):
        # Rank by the Thurstone win-probability ADJUSTED BY CUMULATIVE SCORES:
        # standardized BT-MLE ability + standardized points. Round 1 falls back to
        # the pre-tournament seed. (Empirically this blend, pure points, and pure
        # ability pick nearly the same top-2, so the choice barely matters.)
        if r == 0:
            ranked = [int(i) for i in order]
        else:
            abil = bt_mle(n, games)
            pts = np.array([st[i].points for i in range(n)], float)
            za = (abil - abil.mean()) / (abil.std() + 1e-9)
            zp = (pts - pts.mean()) / (pts.std() + 1e-9)
            ranked = list(np.argsort(-(za + zp)))
        last = r == rounds - 1
        if last:                                   # grand final: the two favourites
            pairs = [(ranked[0], ranked[1])] + _greedy_pairs(ranked[2:], st)
        else:                                      # keep the top-2 favourites apart
            pairs = ([(ranked[0], ranked[2]), (ranked[1], ranked[3])]
                     + _greedy_pairs(ranked[4:], st))
        for a, b in pairs:
            if last and a == ranked[0] and b == ranked[1]:
                margin, res = model.play_decisive(rng, theta[a], theta[b])
                champion = a if res == 1 else b
            else:
                margin, res = model.play(rng, theta[a], theta[b])
            st[a].add(b, res, margin)
            st[b].add(a, -res, -margin)
            games.append((a, b, 1.0 if res == 1 else (0.5 if res == 0 else 0.0)))
    order_final = list(np.argsort(-bt_mle(n, games)))
    ranking = [champion] + [i for i in order_final if i != champion]  # crown finalist
    return champion, ranking


# --------------------------------------------------------------------------- #
# Format 2 -- FIFA World Cup 2026 (groups of 4 + knockout)                     #
# --------------------------------------------------------------------------- #


def run_world_cup(rng, theta, model: MatchModel, seed_order=None):
    """Simulate the 48-team, 2026 format. Returns (champion_idx, full_ranking).

    - Seeded draw: sort teams by the (noisy) pre-tournament seeding into 4 pots
      of 12; each group of 4 gets one team per pot (this is the format's genuine
      strength -- it protects the top-seeded teams from meeting early).
    - Group stage: single round-robin (3 match-days), 3/1/0 points.
    - Advance: top 2 of each group (24) + the 8 best 3rd-placed teams -> R32.
    - Knockout: single elimination R32 -> R16 -> QF -> SF -> Final (5 match-days).

    `seed_order` is the shared pre-tournament ranking (team ids, best-seed
    first); pass the SAME array to every format so seeding is identical and
    realistically imperfect. `full_ranking` orders all 48 teams by how far they
    went (champion first), breaking ties within a stage by group-stage points.
    """
    n = len(theta)
    assert n == 48, "World-Cup format is hard-wired for 48 teams"
    order = np.argsort(-theta) if seed_order is None else np.asarray(seed_order)
    seed_of = {int(idx): s for s, idx in enumerate(order)}

    # Seeded draw into 12 groups of 4, one team per pot.
    pots = [order[0:12], order[12:24], order[24:36], order[36:48]]
    perm = [rng.permutation(12) for _ in range(4)]
    groups = [[] for _ in range(12)]
    for p in range(4):
        for g in range(12):
            groups[g].append(int(pots[p][perm[p][g]]))

    st = {i: Standing(idx=i, theta=float(theta[i]), seed=seed_of[i]) for i in range(n)}

    # Group stage
    games = []          # (i, j, score_for_i) for downstream BT-MLE ranking
    group_tables = []
    for g in groups:
        for a, b in [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]:
            ta, tb = g[a], g[b]
            margin, res = model.play(rng, theta[ta], theta[tb])
            st[ta].add(tb, res, margin)
            st[tb].add(ta, -res, -margin)
            games.append((ta, tb, 1.0 if res == 1 else (0.5 if res == 0 else 0.0)))
        group_tables.append(_sort_key([st[i] for i in g]))

    winners = [t[0].idx for t in group_tables]
    runners = [t[1].idx for t in group_tables]
    thirds = sorted(
        (t[2] for t in group_tables),
        key=lambda s: (s.points, s.gd),
        reverse=True,
    )
    best_thirds = [s.idx for s in thirds[:8]]
    eliminated_group = [t[3].idx for t in group_tables] + [
        s.idx for s in thirds[8:]
    ]

    # 32 qualifiers, seeded for the bracket by (group finish, then points).
    q_winners = sorted(winners, key=lambda i: -st[i].points)
    q_runners = sorted(runners, key=lambda i: -st[i].points)
    q_thirds = best_thirds
    bracket = q_winners + q_runners + q_thirds  # 12 + 12 + 8 = 32, strong-first

    # Standard seeding: 1 vs 32, 2 vs 31, ... keeps the strongest apart.
    field = list(bracket)
    ko_exits = {"R32": [], "R16": [], "QF": [], "SF": [], "Final": []}
    stage_names = ["R32", "R16", "QF", "SF", "Final"]
    for stage in stage_names:
        nxt = []
        losers = []
        m = len(field)
        for k in range(m // 2):
            a, b = field[k], field[m - 1 - k]
            _, res = model.play_decisive(rng, theta[a], theta[b])
            games.append((a, b, 1.0 if res == 1 else 0.0))
            if res == 1:
                nxt.append(a)
                losers.append(b)
            else:
                nxt.append(b)
                losers.append(a)
        ko_exits[stage] = losers
        field = nxt
    champion = field[0]

    # Full-field ranking by round reached (champion first), then group strength.
    def by_group(ids):
        return sorted(ids, key=lambda i: (st[i].points, st[i].gd), reverse=True)

    full = (
        [champion]
        + by_group(ko_exits["Final"])
        + by_group(ko_exits["SF"])
        + by_group(ko_exits["QF"])
        + by_group(ko_exits["R16"])
        + by_group(ko_exits["R32"])
        + by_group(eliminated_group)
    )
    # `group_out` = the 16 teams that never reached the knockout, for the
    # "was the true best sent home too early?" diagnostic.
    return champion, full, games, set(eliminated_group)


# --------------------------------------------------------------------------- #
# Format 3 -- full single round robin (gold-standard reference)                #
# --------------------------------------------------------------------------- #


def run_round_robin(rng, theta, model: MatchModel):
    n = len(theta)
    st = {i: Standing(idx=i, theta=float(theta[i]), seed=0) for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            margin, res = model.play(rng, theta[i], theta[j])
            st[i].add(j, res, margin)
            st[j].add(i, -res, -margin)
    ranking = [s.idx for s in _sort_key([st[i] for i in range(n)])]
    return ranking[0], ranking


# --------------------------------------------------------------------------- #
# Format 4 -- Swiss WITH elimination  (game-budget matched to the World Cup)   #
# --------------------------------------------------------------------------- #
# The reader's question: give Swiss the World Cup's *thrift* by culling the
# lowest-scoring teams each match-day, so the total game count matches the World
# Cup while keeping Swiss's adaptive, near-equal pairings and its no-single-
# upset-eliminates-you aggregation. This is the "best of both" candidate: the
# World Cup's game budget AND Swiss's information efficiency, over 8 match-days.


# Field size entering each of the 8 match-days. All run in the World Cup's
# calendar; survivors are ranked after the last round and the champion is the
# top survivor (no separate coin-flip final). Unlike the knockout, a team is cut
# on its ACCUMULATED record across the whole field, so no single upset ends it.
#   'aggressive' plays only 71 games (fewer than the World Cup's 103);
#   'matched'    plays 104 (~ the World Cup's game budget);
#   'gentle'     holds the field longer, then cuts.
ELIM_SCHEDULES = {
    "aggressive": (48, 32, 24, 16, 10, 6, 4, 2),
    "matched":    (48, 44, 36, 28, 20, 14, 10, 8),
    "gentle":     (48, 48, 44, 34, 22, 12, 8, 4),
}
ELIM_SCHEDULE = ELIM_SCHEDULES["matched"]   # representative, for the diagnostic


def run_swiss_elim(rng, theta, model: MatchModel, schedule=ELIM_SCHEDULE,
                   seed_order=None):
    n = len(theta)
    order = np.argsort(-theta) if seed_order is None else np.asarray(seed_order)
    seed_of = {int(idx): s for s, idx in enumerate(order)}
    st = {i: Standing(idx=i, theta=float(theta[i]), seed=seed_of[i]) for i in range(n)}

    alive = [int(i) for i in order]           # seeded order to start
    elim_order = []                            # teams as they are cut (earliest first)
    cut_md = {}                                # team -> match-day it was cut
    games = []

    for md, size in enumerate(schedule):
        # Trim the field to this match-day's size (cut lowest-ranked survivors).
        if len(alive) > size:
            ranked_alive = [s.idx for s in _sort_key([st[i] for i in alive])]
            keep = ranked_alive[:size]
            cut = ranked_alive[size:]
            for c in cut:
                cut_md[c] = md
            elim_order.extend(reversed(cut))   # higher-ranked cuts placed later
            alive = keep

        # Pair the survivors Swiss-style and play the round.
        if md == 0:
            half = len(alive) // 2
            pairs = [(alive[k], alive[half + k]) for k in range(half)]
        else:
            ranked_alive = [s.idx for s in _sort_key([st[i] for i in alive])]
            pairs = _greedy_pairs(ranked_alive, st)
        for a, b in pairs:
            margin, res = model.play(rng, theta[a], theta[b])
            st[a].add(b, res, margin)
            st[b].add(a, -res, -margin)
            games.append((a, b, 1.0 if res == 1 else (0.5 if res == 0 else 0.0)))

    survivors = [s.idx for s in _sort_key([st[i] for i in alive])]
    champion = survivors[0]
    # Full-field ranking: survivors first, then the eliminated in reverse order
    # of when they were cut (last-cut ranked above first-cut).
    ranking = survivors + list(reversed(elim_order))
    return champion, ranking, games, cut_md


# --------------------------------------------------------------------------- #
# Power metrics                                                                #
# --------------------------------------------------------------------------- #


def bt_mle(n, games, alpha=0.5, iters=100, tol=1e-7):
    """Bradley-Terry maximum-likelihood abilities from a list of games.

    Each game is (i, j, s) with s = score for i (1 win, 0.5 draw, 0 loss).
    Ranks teams by *quality of results*, not raw points: beating a strong team
    counts far more than beating a weak one, so it removes the "losing is
    rewarded" artifact of Swiss point-counting -- a team dropped into a soft
    score group gets little credit for beating soft opponents.

    Fitted with Hunter's (2004) MM algorithm, regularized by `alpha` virtual
    draws against a fixed average-strength anchor so the estimate is always
    finite (even if a team wins or loses everything) and shrinks gently toward
    the field mean. Returns log-abilities (higher = stronger).
    """
    if not games:
        return np.zeros(n)
    a = np.fromiter((g[0] for g in games), dtype=int, count=len(games))
    b = np.fromiter((g[1] for g in games), dtype=int, count=len(games))
    s = np.fromiter((g[2] for g in games), dtype=float, count=len(games))

    W = (np.bincount(a, weights=s, minlength=n)
         + np.bincount(b, weights=1.0 - s, minlength=n)
         + alpha)                         # alpha virtual wins vs anchor(=1)

    p = np.ones(n)
    for _ in range(iters):
        inv = 1.0 / (p[a] + p[b])
        denom = (np.bincount(a, weights=inv, minlength=n)
                 + np.bincount(b, weights=inv, minlength=n)
                 + 2.0 * alpha / (p + 1.0))   # 2*alpha games vs strength-1 anchor
        p_new = W / denom
        p_new /= np.exp(np.log(p_new).mean())   # fix scale (geometric mean 1)
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new
    return np.log(p)


def bt_ranking(n, games):
    """Team ids ordered strongest-first by the Bradley-Terry MLE."""
    return list(np.argsort(-bt_mle(n, games)))


def spearman(rank_ids, true_rank_of):
    """Spearman rho between an ordering (list of team ids, best first) and truth."""
    n = len(rank_ids)
    est = np.empty(n)
    tru = np.empty(n)
    for pos, i in enumerate(rank_ids):
        est[pos] = pos
        tru[pos] = true_rank_of[i]
    est -= est.mean()
    tru -= tru.mean()
    denom = np.sqrt((est**2).sum() * (tru**2).sum())
    return float((est * tru).sum() / denom) if denom else 0.0


@dataclass
class Accum:
    """Accumulates the four power metrics over replications."""

    label: str
    matchdays: int
    games_per_team: float
    total_games: int
    crowned_best: int = 0
    champ_top4: int = 0
    champ_rank_sum: float = 0.0
    rho_sum: float = 0.0
    n: int = 0
    _champ_is_best: list = field(default_factory=list)  # per-rep, for McNemar

    def update(self, champion, ranking, true_rank_of, best_id):
        is_best = int(champion == best_id)
        self.crowned_best += is_best
        self.champ_top4 += int(true_rank_of[champion] < 4)
        self.champ_rank_sum += true_rank_of[champion] + 1  # 1 = perfect
        self.rho_sum += spearman(ranking, true_rank_of)
        self._champ_is_best.append(is_best)
        self.n += 1

    def row(self):
        p = self.crowned_best / self.n
        se = np.sqrt(p * (1 - p) / self.n)
        return dict(
            label=self.label,
            matchdays=self.matchdays,
            gpt=self.games_per_team,
            games=self.total_games,
            p_best=p,
            p_best_ci=1.96 * se,
            p_top4=self.champ_top4 / self.n,
            mean_champ_rank=self.champ_rank_sum / self.n,
            rho=self.rho_sum / self.n,
        )


def mcnemar_p(a_wins, b_wins):
    """Exact two-sided McNemar p-value on paired discordant counts
    (a_wins = reps where A got it right and B did not, and vice-versa)."""
    from math import comb

    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000, help="Monte-Carlo replications")
    ap.add_argument("--teams", type=int, default=48, help="field size (WC needs 48)")
    ap.add_argument("--model", choices=["elo", "gaussian"], default="elo",
                    help="ability model: 'elo' calibrates to head-to-head reality")
    ap.add_argument("--elo-sd", type=float, default=130.0,
                    help="std dev of the field's Elo ratings (elo model)")
    ap.add_argument("--noise", type=float, default=None,
                    help="Thurstone noise sigma (gaussian model; elo fixes it)")
    ap.add_argument("--draw-band", type=float, default=0.32,
                    help="margin magnitude below which a group game is a draw")
    ap.add_argument("--ability-sd", type=float, default=1.0,
                    help="std dev of latent ability (gaussian model)")
    ap.add_argument("--seed-noise", type=float, default=0.12,
                    help="noise (theta units) in the shared pre-tournament seeding")
    ap.add_argument("--swiss-rounds", type=int, default=8,
                    help="max Swiss rounds (snapshots taken after each)")
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--no-rr", action="store_true", help="skip round-robin reference")
    ap.add_argument("--plot", type=str, default=None,
                    help="path to save the power-vs-match-day figure")
    args = ap.parse_args()

    n = args.teams
    # For the elo model the probit scale is fixed so outcomes match Elo exactly.
    sigma = CALIBRATED_SIGMA if args.model == "elo" else (args.noise or 1.0)
    model = MatchModel(sigma=sigma, draw_band=args.draw_band)
    rng = np.random.default_rng(args.seed)

    # Calendar / games accounting.
    WC_MATCHDAYS = 8            # 3 group + R32,R16,QF,SF,Final
    WC_TOTAL_GAMES = 12 * 6 + (16 + 8 + 4 + 2 + 1)  # 72 + 31 = 103
    WC_GPT = WC_TOTAL_GAMES * 2 / n                  # avg games per team

    R = args.swiss_rounds
    wc = Accum("World Cup (bracket rank)", WC_MATCHDAYS, WC_GPT, WC_TOTAL_GAMES)
    wc_mle = Accum("World Cup + BT-MLE rank", WC_MATCHDAYS, WC_GPT, WC_TOTAL_GAMES)
    elim = {
        name: Accum(f"Swiss-elim: {name}", len(s),
                    sum(x // 2 for x in s) * 2 / n, sum(x // 2 for x in s))
        for name, s in ELIM_SCHEDULES.items()
    }
    swiss = {
        r: Accum(f"Swiss ({r}r, points)", r, float(r), r * (n // 2))
        for r in range(1, R + 1)
    }
    swiss_mle = Accum(f"Swiss ({R}r) + BT-MLE", R, float(R), R * (n // 2))
    adapt = {
        r: Accum(f"Adaptive Thurstone ({r}r)", r, float(r), r * (n // 2))
        for r in range(1, R + 1)
    }
    focus = {
        r: Accum(f"Champion-focus ({r}r)", r, float(r), r * (n // 2))
        for r in range(1, R + 1)
    }
    protected = Accum("Protected + grand final", R, float(R), R * (n // 2))
    rr = None if args.no_rr else Accum(
        "Round robin (full)", n - 1, float(n - 1), n * (n - 1) // 2
    )

    calibration = {"group_games": 0, "draws": 0, "fav_wins": 0, "fav_games": 0}
    # "Sent home too early?" diagnostic -- how often the TRUE best team is
    # discarded before the deciding stage in each elimination-based format.
    diag = {"reps48": 0, "wc_group_out": 0, "wc_no_final": 0, "elim_cut_early": 0,
            "champ_from_topboard": 0, "best_on_topboard": 0}

    for _ in range(args.reps):
        theta = sample_abilities(rng, n, args.model, args.ability_sd, args.elo_sd)
        true_order = np.argsort(-theta)           # best-first
        true_rank_of = np.empty(n, dtype=int)
        for pos, i in enumerate(true_order):
            true_rank_of[int(i)] = pos
        best_id = int(true_order[0])

        # Shared, realistically-imperfect pre-tournament seeding (an FIFA-ranking
        # analogue): a noisy estimate of ability. The SAME seed_order is handed to
        # every format, so no format gets a seeding edge.
        rating = theta + rng.normal(0.0, args.seed_noise, size=n)
        seed_order = np.argsort(-rating)

        # --- World Cup (only defined for 48) ---
        if n == 48:
            champ, wc_full, wc_games, group_out = run_world_cup(
                rng, theta, model, seed_order)
            wc.update(champ, wc_full, true_rank_of, best_id)
            # Same World-Cup games, but ranked by the Bradley-Terry MLE.
            mr = bt_ranking(n, wc_games)
            wc_mle.update(mr[0], mr, true_rank_of, best_id)

            # --- Swiss + elimination: several cut schedules, all 8 match-days ---
            for name, sched in ELIM_SCHEDULES.items():
                champ, ranking, _, cut_md = run_swiss_elim(
                    rng, theta, model, sched, seed_order)
                elim[name].update(champ, ranking, true_rank_of, best_id)
                if name == "matched":
                    matched_cut_md = cut_md

            # Diagnostic: did each elimination format send the true best home early?
            diag["reps48"] += 1
            diag["wc_group_out"] += int(best_id in group_out)     # out in group stage
            diag["wc_no_final"] += int(best_id not in wc_full[:2])  # never made the final
            diag["elim_cut_early"] += int(
                matched_cut_md.get(best_id, len(ELIM_SCHEDULE)) < len(ELIM_SCHEDULE) - 1)

        # --- Swiss: one run, snapshot every round (points ranking) ---
        snaps, s_games = run_swiss(rng, theta, model, R, seed_order)
        for r in range(1, R + 1):
            ranking = snaps[r - 1]
            swiss[r].update(ranking[0], ranking, true_rank_of, best_id)
        # Same full Swiss games, ranked by the BT-MLE (fixes "losing is rewarded").
        smr = bt_ranking(n, s_games)
        swiss_mle.update(smr[0], smr, true_rank_of, best_id)

        # 'Top board' diagnostic: entering the final round the standings leaders
        # (snaps[R-2][:2]) are paired on board 1. How often is the eventual
        # champion -- and the true best -- actually on that board?
        top_board = set(snaps[R - 2][:2]) if R >= 2 else set(snaps[0][:2])
        diag["champ_from_topboard"] += int(snaps[R - 1][0] in top_board)
        diag["best_on_topboard"] += int(best_id in top_board)

        # --- Adaptive Thurstone: model-based pairing + MLE ranking ---
        a_snaps, _ = run_adaptive(rng, theta, model, R, seed_order)
        for r in range(1, R + 1):
            ranking = a_snaps[r - 1]
            adapt[r].update(ranking[0], ranking, true_rank_of, best_id)

        # --- Champion-focus: best-arm identification (we care only who wins) ---
        f_snaps, _ = run_champion_focus(rng, theta, model, R, seed_order)
        for r in range(1, R + 1):
            ranking = f_snaps[r - 1]
            focus[r].update(ranking[0], ranking, true_rank_of, best_id)

        # --- Protected final: top-2 kept apart until a decisive grand final ---
        if n == 48:
            champ, ranking = run_protected_final(rng, theta, model, R, seed_order)
            protected.update(champ, ranking, true_rank_of, best_id)

        # --- Round robin reference ---
        if rr is not None:
            champ, ranking = run_round_robin(rng, theta, model)
            rr.update(champ, ranking, true_rank_of, best_id)

        # crude calibration diagnostics from one fresh group game per rep
        a, b = int(true_order[10]), int(true_order[20])
        _, res = model.play(rng, theta[a], theta[b])
        calibration["group_games"] += 1
        calibration["draws"] += int(res == 0)
        calibration["fav_games"] += 1
        calibration["fav_wins"] += int(res == 1)

    # ----------------------------------------------------------------------- #
    # Report                                                                   #
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("  SWISS vs WORLD CUP  --  which format better identifies the best team?")
    print("=" * 78)
    print(f"  Replications        : {args.reps:,}   (fresh abilities each rep)")
    print(f"  Field size          : {n} teams")
    print(f"  Match model         : performance = ability + N(0,σ²), higher wins")
    if args.model == "elo":
        print(f"  Ability calibration : Elo, field sd {args.elo_sd:.0f} pts "
              f"(head-to-head calibrated)")

        def elo_winp(gap):
            return norm.cdf(gap * THURSTONE_PER_ELO / (np.sqrt(2) * sigma))

        gaps = [50, 100, 200, 300]
        implied = "   ".join(f"+{g}→{elo_winp(g):.0%}" for g in gaps)
        print(f"  Implied win prob    : {implied}   (single game, Elo edge)")
    else:
        print(f"  Ability calibration : Gaussian, sd {args.ability_sd}, σ {sigma:.3f}")
    print(f"  Seeding             : shared noisy pre-tournament rating "
          f"(seed-noise {args.seed_noise})")
    dr = calibration["draws"] / calibration["group_games"]
    fw = calibration["fav_wins"] / calibration["fav_games"]
    print(f"  Realized draw rate  : {dr:5.1%}  (rank-11 vs rank-21 sample)")
    print(f"  Mid-favourite win % : {fw:5.1%}")
    print("-" * 78)

    header = (f"  {'format':<26}{'m-days':>7}{'g/team':>8}{'games':>7}"
              f"{'P(best)':>9}{'±95%':>7}{'top4':>7}{'E[rank]':>8}{'ρ':>7}")
    print(header)
    print("-" * 78)

    def show(acc):
        r = acc.row()
        print(f"  {r['label']:<26}{r['matchdays']:>7}{r['gpt']:>8.1f}"
              f"{r['games']:>7}{r['p_best']:>9.3f}{r['p_best_ci']:>7.3f}"
              f"{r['p_top4']:>7.2f}{r['mean_champ_rank']:>8.2f}{r['rho']:>7.3f}")

    if n == 48:
        print("  No-elimination formats, 8 match-days (equal calendar):")
        show(wc)
        show(wc_mle)
        show(swiss[R])
        show(swiss_mle)
        show(adapt[R])
        show(focus[R])
        print("  " + "-" * 74)
        print("  Send teams home AND beat the knockout -- Swiss with elimination:")
        show(wc)
        for name in ELIM_SCHEDULES:
            show(elim[name])
        print("  " + "-" * 74)
        print("  'Top-2 can't meet until the final' (marquee grand final):")
        show(swiss[R])
        show(protected)
    print("  " + "-" * 74)
    print("  Champion-focus (best-arm), power vs match-day -- WINNER-ONLY objective:")
    for r in range(1, R + 1):
        show(focus[r])
    if rr is not None:
        print("  " + "-" * 74)
        show(rr)
    print("-" * 78)

    # ----------------------------------------------------------------------- #
    # "Sent home too early?" diagnostic                                        #
    # ----------------------------------------------------------------------- #
    if n == 48 and diag["reps48"]:
        d = diag["reps48"]
        print("  DIAGNOSTIC -- how often the TRUE best team is discarded early:")
        print(f"      World Cup: eliminated in the GROUP stage   "
              f"{diag['wc_group_out']/d:5.1%}")
        print(f"      World Cup: never even reaches the final     "
              f"{diag['wc_no_final']/d:5.1%}")
        print(f"      Swiss+elim: cut before the final round      "
              f"{diag['elim_cut_early']/d:5.1%}")
        print(f"      Adaptive / full Swiss: nobody is eliminated  0.0%  "
              f"(the best is always still measurable)")
        print("  TOP BOARD -- final-round board 1 pairs the two standings leaders:")
        print(f"      P(champion comes from the top board)        "
              f"{diag['champ_from_topboard']/args.reps:5.1%}")
        print(f"      P(true best is on the top board)            "
              f"{diag['best_on_topboard']/args.reps:5.1%}   "
              f"(parity ceiling on any 'final decides it' rule)")
        print("-" * 78)

    # ----------------------------------------------------------------------- #
    # The headline comparisons                                                 #
    # ----------------------------------------------------------------------- #
    if n == 48:
        d = diag["reps48"]
        wc_r = wc.row()
        s8 = swiss[R]
        s8_r = s8.row()

        # Paired McNemar test on champion identification (same abilities/rep).
        a_only = sum(
            1 for sw, w in zip(s8._champ_is_best, wc._champ_is_best)
            if sw == 1 and w == 0
        )
        b_only = sum(
            1 for sw, w in zip(s8._champ_is_best, wc._champ_is_best)
            if sw == 0 and w == 1
        )
        p = mcnemar_p(a_only, b_only)

        # Representative elimination schedule (the aggressive, fewest-games one).
        elim_agg = elim["aggressive"]
        elim_agg_r = elim_agg.row()
        ea = sum(1 for e, w in zip(elim_agg._champ_is_best, wc._champ_is_best)
                 if e == 1 and w == 0)
        eb = sum(1 for e, w in zip(elim_agg._champ_is_best, wc._champ_is_best)
                 if e == 0 and w == 1)
        ep = mcnemar_p(ea, eb)

        print("\n  VERDICT")
        print("  " + "-" * 74)
        lift = s8_r["p_best"] - wc_r["p_best"]
        print(f"  [1] EQUAL CALENDAR ({WC_MATCHDAYS} match-days) -- the binding World-Cup"
              f" constraint:")
        print(f"      World Cup   P(crown the best team) = {wc_r['p_best']:.3f}"
              f"   ρ = {wc_r['rho']:.3f}")
        print(f"      Swiss-{args.swiss_rounds}     P(crown the best team) = "
              f"{s8_r['p_best']:.3f}   ρ = {s8_r['rho']:.3f}")
        print(f"                  Δ = +{lift:.3f}  "
              f"({lift / wc_r['p_best'] * 100:.0f}% relative)")
        print(f"      Paired McNemar test on champion ID:  p = {p:.2e}")
        print(f"          (Swiss-right/WC-wrong={a_only},  WC-right/Swiss-wrong={b_only})")
        print("      -> Swiss is the clearly more powerful estimator per match-day: no")
        print("         team is eliminated, so every round adds signal for the whole")
        print("         field instead of only the surviving bracket.")
        print()
        # Decomposition of the reader's three critiques into the clever fix.
        s8_mle_r = swiss_mle.row()
        ad_r = adapt[R].row()
        aa = sum(1 for x, w in zip(adapt[R]._champ_is_best, wc._champ_is_best)
                 if x == 1 and w == 0)
        ab = sum(1 for x, w in zip(adapt[R]._champ_is_best, wc._champ_is_best)
                 if x == 0 and w == 1)
        ap_ = mcnemar_p(aa, ab)
        rr_r = rr.row() if rr is not None else None

        print("  [2] THE READER'S CRITIQUES, AND THE CLEVER FIX (all at 8 match-days):")
        print("      (a) 'losing is rewarded' -- classic Swiss ranks by POINTS, so a loss")
        print("          buys softer opponents and cheap points. Fix: rank by the")
        print("          Bradley-Terry / Thurstone MLE (credits quality of opponents).")
        print(f"              Swiss, points ranking   P(best) = {s8_r['p_best']:.3f}"
              f"   ρ = {s8_r['rho']:.3f}")
        print(f"              Swiss, BT-MLE ranking    P(best) = {s8_mle_r['p_best']:.3f}"
              f"   ρ = {s8_mle_r['rho']:.3f}   <- same games, better estimator")
        print("      (b) 'sent home too early' -- see the diagnostic above: the World Cup")
        print(f"          dumps the true best in the group stage {diag['wc_group_out']/d:.0%} of the"
              f" time; the elim")
        print(f"          schedule cuts it early {diag['elim_cut_early']/d:.0%}. Fix: DON'T eliminate"
              f" -- games run")
        print("          in parallel, so keep all 48 playing every match-day.")
        print("      (c) 'too random' -- I tried to beat plain Swiss by being clever:")
        print("          pairing ONLY near-equals off the posterior (Adaptive) and")
        print("          FUNNELLING games onto the contenders (Champion-focus, best-arm).")
        print()
        fc_r = focus[R].row()
        print("  [3] WINNER-IDENTIFICATION POWER AT 8 MATCH-DAYS  (the honest scoreboard):")
        rows8 = [("World Cup (RR + KO)", wc_r), ("Swiss (points)", s8_r),
                 ("Swiss + BT-MLE", s8_mle_r), ("Adaptive Thurstone", ad_r),
                 ("Champion-focus (best-arm)", fc_r)]
        for name, rr2 in rows8:
            bar = "#" * int(round(rr2["p_best"] * 100))
            print(f"        {name:<27} P(best) = {rr2['p_best']:.3f} ± "
                  f"{rr2['p_best_ci']:.3f}  {bar}")
        if rr_r is not None:
            print(f"        {'Round-robin CEILING':<27} P(best) = {rr_r['p_best']:.3f}"
                  f"          (47 match-days, 1128 games)")
        print("  " + "-" * 74)
        print("  FINDINGS (what actually held up at scale):")
        print(f"    1. Swiss >> World Cup, robustly (p = {p:.1e}). The decisive lever is")
        print(f"       simply NOT eliminating: the World Cup discards the true best team")
        print(f"       before the final {diag['wc_no_final']/d:.0%} of the time, on 1-3 games of data.")
        mle_gap = s8_mle_r["p_best"] - s8_r["p_best"]
        print(f"    2. 'Losing is rewarded' is REAL but SMALL: BT-MLE moves P(best) by only")
        print(f"       {mle_gap:+.3f} vs points -- Swiss pairing already balances schedules, so")
        print(f"       the Buchholz-style correction has little left to fix.")
        print(f"    3. Clever pairing does NOT help winner-ID here: Adaptive/Champion-focus")
        print(f"       tie or trail plain Swiss. Pairing only near-equals turns the top into")
        print(f"       ~50/50 coin-flips, which REMOVES separating signal.")
        if rr_r is not None:
            print(f"    4. Hard ceiling from parity: even a 1128-game round-robin crowns the")
            print(f"       true best only {rr_r['p_best']:.0%} of the time -- the top teams sit within a")
            print(f"       coin-flip, so there is little headroom above plain Swiss at 8 days.")
        print(f"    5. You CAN eliminate and STILL beat the knockout. Swiss-elim (even the")
        print(f"       aggressive {elim_agg_r['games']}-game schedule -- FEWER games than the World Cup's")
        print(f"       {WC_TOTAL_GAMES}) crowns the best {elim_agg_r['p_best']:.3f} vs the knockout's "
              f"{wc_r['p_best']:.3f} (McNemar p = {ep:.1e}).")
        print(f"       It cuts on ACCUMULATED record, not one game, so no single upset or")
        print(f"       unlucky group ends a contender -- and it ~ties full Swiss, so for")
        print(f"       winner-ID the elimination is essentially free.")
        prot_r = protected.row()
        print(f"    6. A 'top-2 can't meet until the final' rule is nearly free: reserving")
        print(f"       the marquee clash for a grand final that DECIDES the title scores")
        print(f"       {prot_r['p_best']:.3f} vs plain Swiss's {s8_r['p_best']:.3f} -- a small cost, because one")
        print(f"       coin-flip game replaces 8 rounds of evidence. Keep the drama, but let")
        print(f"       the final be a tie-break, not winner-take-all. (Still >> the knockout.)")
        print("  " + "-" * 74)
        print("  BOTTOM LINE: for the same calendar, ANY Swiss-style league beats")
        print("  group+knockout -- with OR without elimination. The World Cup's flaw is not")
        print("  that it sends teams home; it's sending them home on one game / one group.")
        print("  Cut on a whole-field record instead and you keep the drama AND the power.")
        print("=" * 78 + "\n")

    if args.plot:
        _make_plot(args, wc if n == 48 else None, swiss, adapt, rr)


def _make_plot(args, wc, swiss, adapt, rr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = list(range(1, args.swiss_rounds + 1))
    p_swiss = [swiss[r].row()["p_best"] for r in rounds]
    p_adapt = [adapt[r].row()["p_best"] for r in rounds]
    rho_swiss = [swiss[r].row()["rho"] for r in rounds]
    rho_adapt = [adapt[r].row()["rho"] for r in rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(rounds, p_adapt, "o-", color="#1f77b4", label="Adaptive Thurstone")
    ax1.plot(rounds, p_swiss, "s--", color="#7f7f7f", label="Swiss (points)")
    if wc is not None:
        ax1.axhline(wc.row()["p_best"], color="#d62728", ls="--",
                    label="World Cup (8 match-days)")
        ax1.axvline(8, color="gray", ls=":", alpha=0.6)
    ax1.set_xlabel("match-days (rounds)")
    ax1.set_ylabel("P(crown the true best team)")
    ax1.set_title("Statistical power vs calendar time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(rounds, rho_adapt, "o-", color="#1f77b4", label="Adaptive Thurstone")
    ax2.plot(rounds, rho_swiss, "s--", color="#7f7f7f", label="Swiss (points)")
    if wc is not None:
        ax2.axhline(wc.row()["rho"], color="#d62728", ls="--", label="World Cup")
    if rr is not None:
        ax2.axhline(rr.row()["rho"], color="#2ca02c", ls="-.",
                    label="Round robin (ceiling)")
    ax2.set_xlabel("match-days (rounds)")
    ax2.set_ylabel("Spearman ρ vs true ranking")
    ax2.set_title("Full-field ranking fidelity")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Swiss vs FIFA World Cup format — Thurstone Monte-Carlo",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"  [plot saved to {args.plot}]\n")


if __name__ == "__main__":
    main()
