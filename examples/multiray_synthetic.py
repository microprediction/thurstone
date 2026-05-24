from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from thurstone import (AbilityCalibrator, Density, MultiRayGlobalCalibrator,
                       UniformLattice)
from thurstone.multiray import (_interp_price_and_slope_1d,
                                _interp_price_and_slope_2d)
from thurstone.pricing import Race


def main():
    rng = np.random.default_rng(12345)
    dim = 2
    n_items = 25
    n_conds = 6

    grid = UniformLattice(L=500, unit=0.1)
    base = Density.skew_normal(grid, loc=0.0, scale=1.0, a=0.0)

    item_ids = [f"h{i}" for i in range(n_items)]

    # True parameters
    Z_true = rng.standard_normal((n_items, dim))
    V_true = []
    beta_true = []
    for _ in range(n_conds):
        v = rng.standard_normal(dim)
        v = v / np.linalg.norm(v)
        V_true.append(v)
        beta_true.append(0.1 * rng.standard_normal())
    V_true = np.array(V_true, dtype=float)
    beta_true = np.array(beta_true, dtype=float)

    # Observations
    calibrators = []
    prices_obs = []
    cond_ids = []
    for j in range(n_conds):
        a = beta_true[j] + Z_true @ V_true[j]
        dens = [base.shift_fractional(float(ai / base.lattice.unit)) for ai in a]
        p_obs = np.array(Race(dens).state_prices(), dtype=float)
        calibrators.append(AbilityCalibrator(base))
        prices_obs.append(p_obs)
        cond_ids.append(f"c{j}")

    # Split conditions into train/hold-out
    idx = np.arange(n_conds)
    rng.shuffle(idx)
    split = max(1, n_conds // 2)
    train_idx = idx[:split]
    hold_idx = idx[split:]
    train_cond_ids = [cond_ids[i] for i in train_idx]
    hold_cond_ids = [cond_ids[i] for i in hold_idx]

    fit = MultiRayGlobalCalibrator(item_ids=item_ids, dim=dim, random_state=999)
    for j in train_idx:
        fit.add_condition(
            cond_id=cond_ids[j],
            calibrator=calibrators[j],
            item_ids=item_ids,
            prices=prices_obs[j],
        )

    # Before fit
    fit.rebuild_all_curves()
    preds0_train = np.stack(
        [fit.predict_condition(cid) for cid in train_cond_ids], axis=0
    )
    obs_train = np.stack(
        [prices_obs[cond_ids.index(cid)] for cid in train_cond_ids], axis=0
    )
    mse0_train = float(np.mean((preds0_train - obs_train) ** 2))
    print(f"Train MSE before: {mse0_train:.6e}")

    # Fit
    fit.fit_with_rebuild(num_outer_iters=5, num_inner_iters=10)

    # After fit
    fit.rebuild_all_curves()
    preds1_train = np.stack(
        [fit.predict_condition(cid) for cid in train_cond_ids], axis=0
    )
    mse1_train = float(np.mean((preds1_train - obs_train) ** 2))
    max_abs_err_train = float(np.max(np.abs(preds1_train - obs_train)))
    print(f"Train MSE after:  {mse1_train:.6e}")
    print(f"Train max abs err: {max_abs_err_train:.6e}")

    # ---- Hold-out evaluation: fit (beta, V) per held-out condition with Z frozen ----
    def fit_holdout_condition(
        cond_j: int,
        Z: np.ndarray,
        step_beta=0.3,
        step_v=0.3,
        l2_v=1e-6,
        slope_floor=1e-10,
        iters=15,
    ):
        cid = cond_ids[cond_j]
        cal = calibrators[cond_j]
        # Initialize (beta, v) using ability-space LS from local inversion
        try:
            a_local = np.asarray(cal.solve_from_prices(prices_obs[cond_j]), dtype=float)
        except Exception:
            a_local = np.zeros(len(item_ids), dtype=float)
        X0 = np.zeros((len(item_ids), 1 + dim), dtype=float)
        X0[:, 0] = 1.0
        X0[:, 1:] = Z
        XtX0 = X0.T @ X0
        XtX0[1:, 1:] += max(l2_v, 1e-8) * np.eye(dim, dtype=float)
        Xty0 = X0.T @ a_local
        try:
            w0 = np.linalg.solve(XtX0, Xty0)
        except np.linalg.LinAlgError:
            w0 = np.linalg.lstsq(XtX0, Xty0, rcond=None)[0]
        beta_loc = float(w0[0])
        v_loc = np.asarray(w0[1:], dtype=float)
        for _ in range(iters):
            # rebuild curves around current field for this condition
            mu_r = (Z @ v_loc + beta_loc).tolist()
            cal.rebuild_curves_from_field_1d(mu_r)
            # predict p and slopes
            p_hat = np.zeros(len(item_ids), dtype=float)
            slopes = np.zeros(len(item_ids), dtype=float)
            for k in range(len(item_ids)):
                mu = float(beta_loc + float(np.dot(v_loc, Z[k])))
                p, dp = _interp_price_and_slope_1d(cal, mu)
                p_hat[k] = p
                slopes[k] = dp
            e = p_hat - prices_obs[cond_j]
            slopes_safe = slopes.copy()
            mask = np.abs(slopes_safe) < slope_floor
            slopes_safe[mask] = slope_floor
            y = -e / slopes_safe
            # regress for delta beta and delta v
            X = np.zeros((len(item_ids), 1 + dim), dtype=float)
            X[:, 0] = 1.0
            X[:, 1:] = Z
            XtX = X.T @ X
            XtX[1:, 1:] += l2_v * np.eye(dim, dtype=float)
            Xty = X.T @ y
            try:
                w = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
            beta_loc += step_beta * float(w[0])
            v_loc = v_loc + step_v * np.asarray(w[1:], dtype=float)
        # final prediction without another rebuild (safe to reuse last curves)
        p_final = np.zeros(len(item_ids), dtype=float)
        for k in range(len(item_ids)):
            mu = float(beta_loc + float(np.dot(v_loc, Z[k])))
            p, _ = _interp_price_and_slope_1d(cal, mu)
            p_final[k] = p
        return beta_loc, v_loc, p_final

    Z_fit = np.stack([fit.Z[hid] for hid in item_ids], axis=0)
    preds_hold = []
    for j in hold_idx:
        _, _, p_final = fit_holdout_condition(
            j,
            Z_fit,
            step_beta=fit.step_beta,
            step_v=fit.step_v,
            l2_v=fit.l2_v,
            slope_floor=fit.slope_floor,
            iters=15,
        )
        preds_hold.append(p_final)
    if preds_hold:
        preds_hold = np.stack(preds_hold, axis=0)
        obs_hold = np.stack([prices_obs[j] for j in hold_idx], axis=0)
        mse_hold = float(np.mean((preds_hold - obs_hold) ** 2))
        max_abs_err_hold = float(np.max(np.abs(preds_hold - obs_hold)))
        print(f"Hold-out MSE: {mse_hold:.6e}")
        print(f"Hold-out max abs err: {max_abs_err_hold:.6e}")
    else:
        preds_hold = np.empty((0, len(item_ids)), dtype=float)
        obs_hold = np.empty((0, len(item_ids)), dtype=float)

    # Show a few embeddings and rays
    some_items = item_ids[:3]
    print("Sample item embeddings:")
    for hid in some_items:
        print(hid, fit.Z[hid])
    print("Condition rays (first 3, train only):")
    for cid in train_cond_ids[:3]:
        print(cid, fit.V[cid], "beta=", fit.beta[cid])

    # ---- Visualization: embeddings and rays in 2D ----
    Z = np.stack([fit.Z[hid] for hid in item_ids], axis=0)
    # color items by the most likely condition (argmax predicted prob)
    # use train predictions for coloring (avoid leakage from hold-out fitting)
    pred_mat = preds1_train  # shape (n_train_conds, n_items)
    assign = np.argmax(pred_mat, axis=0)  # per item -> condition index within train set
    cmap = plt.get_cmap("tab10")

    def cid_color(idx: int):
        return cmap(idx % cmap.N)

    plt.figure(figsize=(6, 6))
    item_colors = [cid_color(int(assign[i])) for i in range(len(item_ids))]
    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        s=20,
        alpha=0.85,
        c=item_colors,
        label="items (colored by argmax cond)",
    )
    # draw rays from the origin (V are normalized), color by condition
    scale = np.percentile(np.linalg.norm(Z, axis=1), 95)
    # Only plot rays for conditions actually present in the fitted model
    trained_cids = list(fit.V.keys())
    for j, cid in enumerate(trained_cids):
        v = fit.V[cid]
        # color index consistent with train ordering if possible
        color_idx = train_cond_ids.index(cid) if cid in train_cond_ids else j
        color = cid_color(color_idx)
        plt.arrow(
            0,
            0,
            v[0] * scale,
            v[1] * scale,
            length_includes_head=True,
            head_width=0.05 * scale,
            color=color,
            alpha=0.95,
        )
        plt.text(v[0] * scale * 1.05, v[1] * scale * 1.05, cid, color=color)
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.title("Item embeddings Z and condition rays V")
    plt.legend()
    plt.tight_layout()

    # ---- Visualization: predicted vs observed probabilities per condition ----
    fig, axes = plt.subplots(
        len(cond_ids), 1, figsize=(7, 2.2 * len(cond_ids)), sharex=True
    )
    if len(cond_ids) == 1:
        axes = [axes]
    for j, cid in enumerate(cond_ids):
        p_obs = prices_obs[j]
        if j in train_idx:
            tpos = train_cond_ids.index(cid)
            p_pred = preds1_train[tpos]
            color = cid_color(tpos)
        else:
            hpos = list(hold_idx).index(j)
            p_pred = preds_hold[hpos]
            color = cid_color(hpos + 5)  # shift color index for hold-out
        ax = axes[j]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.scatter(p_obs, p_pred, s=20, alpha=0.85, color=color)
        ax.set_ylabel(f"{cid} {'[train]' if j in train_idx else '[hold-out]'}")
    axes[-1].set_xlabel("Observed probability")
    axes[0].set_title("Predicted vs observed")
    plt.tight_layout()

    # ---- Visualization: ability vs probability along each ray ----
    fig, axes = plt.subplots(
        len(cond_ids), 1, figsize=(7, 2.2 * len(cond_ids)), sharex=False
    )
    if len(cond_ids) == 1:
        axes = [axes]
    for j, cid in enumerate(cond_ids):
        p_obs = prices_obs[j]
        if j in train_idx:
            v = fit.V[cid]
            b = fit.beta[cid]
            a = Z @ v + b
            tpos = train_cond_ids.index(cid)
            p_pred = preds1_train[tpos]
            color = cid_color(tpos)
        else:
            # refit to get parameters and predicted probs for hold-out without changing Z
            beta_loc, v_loc, p_final = fit_holdout_condition(
                j,
                Z,
                step_beta=fit.step_beta,
                step_v=fit.step_v,
                l2_v=fit.l2_v,
                slope_floor=fit.slope_floor,
                iters=5,
            )
            a = Z @ v_loc + beta_loc
            p_pred = p_final
            hpos = list(hold_idx).index(j)
            color = cid_color(hpos + 5)
        ax = axes[j]
        ax.scatter(
            a,
            p_obs,
            s=30,
            facecolors="none",
            edgecolors=color,
            marker="o",
            alpha=0.95,
            label=f"{cid} obs",
        )
        ax.scatter(
            a, p_pred, s=30, color=color, marker="x", alpha=0.95, label=f"{cid} pred"
        )
        ax.set_ylabel(cid)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("a_ij = beta_j + v_j^T z_i")
    axes[0].set_title("Per-condition: ability vs probability")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
