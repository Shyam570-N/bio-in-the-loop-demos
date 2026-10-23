"""
Session 2: multi-objective Bayesian optimization (Pareto front) + a biologist rules layer.
SIMULATED data only. Builds on bayesopt_session1.py (same hidden TEER landscape).

Three objectives for each condition (dose, timing):
  1. TEER recovery at 72 h          (higher is better)
  2. LDH release (cytotoxicity)     (lower is better)
  3. Dose                           (lower is better; wider safety margin)

Method: ParEGO-style scalarization. Each pick uses a random weighting of the three objectives,
a Gaussian process on that weighted score, and expected improvement. Different weightings push
the search toward different parts of the trade-off front.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bayesopt_session1 as b

GRID, GRID_N = b.GRID, b.GRID_N
LDH_SD = 2.0
RULE_LDH_MAX = 20.0     # example biologist rule: reject if LDH release > 20 %
RULE_TEER_MIN = 60.0    # example acceptance floor: TEER recovery >= 60 %

def _sig(z): return 1.0 / (1.0 + np.exp(-z))

def true_ldh(u, v):
    x = b.DOSE_RANGE[0] + (b.DOSE_RANGE[1] - b.DOSE_RANGE[0]) * np.asarray(u)
    teer = b.true_teer(u, v)
    # background + mild dose-dependent stress + steep toxicity at high dose + injury-related death (less if protected)
    return 5 + 8 * _sig((x - 0.2) / 0.4) + 50 * _sig((x - 1.2) / 0.25) + 10 * (1 - teer / 100)

def f_true(X):
    """All objectives oriented so that HIGHER is better, each scaled to about 0..1."""
    return np.column_stack([b.true_teer(X[:, 0], X[:, 1]) / 100,
                            1 - true_ldh(X[:, 0], X[:, 1]) / 100,
                            1 - X[:, 0]])

def f_measured(X, rng):
    teer = b.true_teer(X[:, 0], X[:, 1]) + rng.normal(0, b.NOISE_SD, len(X))
    ldh = true_ldh(X[:, 0], X[:, 1]) + rng.normal(0, LDH_SD, len(X))
    return np.column_stack([teer / 100, 1 - ldh / 100, 1 - X[:, 0]])

def nondominated(F):
    """Boolean mask of non-dominated rows (maximize all columns)."""
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        dom = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep

_MC = np.random.default_rng(0).uniform(0, 1, (40000, 3))
def hypervolume(F):
    """Monte Carlo share of the unit cube dominated by F (reference point = worst corner)."""
    F = F[nondominated(F)]
    covered = np.zeros(len(_MC), bool)
    for f in F:
        covered |= np.all(_MC <= f, axis=1)
    return covered.mean()

def scalar(F, w):
    t = F * w
    return t.min(axis=1) + 0.05 * t.sum(axis=1)        # augmented Chebyshev (to maximize)

def propose_one(X, s_raw, hp, exclude):
    m, sdv = s_raw.mean(), s_raw.std() + 1e-9
    y = (s_raw - m) / sdv
    mu, sd = b.gp_posterior(X, y, GRID, hp)
    mu_obs, _ = b.gp_posterior(X, y, X, hp)
    ei = b.expected_improvement(mu, sd, mu_obs.max())
    for p in exclude:
        ei[np.all(np.isclose(GRID, p), axis=1)] = -1
    return GRID[int(np.argmax(ei))]

def campaign_mo(seed, first=8, rounds=6, batch=6):
    rng = np.random.default_rng(seed)
    X = b.lhs(first, rng)
    F = f_measured(X, rng)
    for _ in range(rounds):
        s_eq = scalar(F, np.ones(3) / 3)
        m, sdv = s_eq.mean(), s_eq.std() + 1e-9
        hp = b.fit_hyperparams(X, (s_eq - m) / sdv)
        picks = []
        for _ in range(batch):
            w = rng.dirichlet(np.ones(3))
            picks.append(propose_one(X, scalar(F, w), hp, picks))
        picks = np.array(picks)
        X = np.vstack([X, picks])
        F = np.vstack([F, f_measured(picks, rng)])
    return X

def campaign_random_mo(seed, n):
    rng = np.random.default_rng(20_000 + seed)
    return rng.uniform(0, 1, (n, 2))

def to_units(X):
    x = b.DOSE_RANGE[0] + (b.DOSE_RANGE[1] - b.DOSE_RANGE[0]) * X[:, 0]
    t = b.TIME_RANGE[0] + (b.TIME_RANGE[1] - b.TIME_RANGE[0]) * X[:, 1]
    return 10 ** x, t

if __name__ == "__main__":
    n_total = 44
    checks = [8, 14, 20, 26, 32, 38, 44]
    Ftrue_grid = f_true(GRID)
    front_mask = nondominated(Ftrue_grid)
    hv_full = hypervolume(Ftrue_grid)
    print(f"Grid conditions: {len(GRID)}, true Pareto-front conditions: {front_mask.sum()}, hypervolume {hv_full:.3f}")

    bo_hv, rn_hv, bo_X = [], [], None
    for s in range(30):
        Xb = campaign_mo(s)
        Xr = campaign_random_mo(s, n_total)
        if s == 1:
            bo_X, rn_X = Xb, Xr
        bo_hv.append([hypervolume(f_true(Xb[:k])) / hv_full for k in checks])
        rn_hv.append([hypervolume(f_true(Xr[:k])) / hv_full for k in checks])
    bo_hv, rn_hv = np.array(bo_hv), np.array(rn_hv)
    for i, k in enumerate(checks):
        print(f"{k:>3} wells | BO median {np.median(bo_hv[:, i]):.2f} | random median {np.median(rn_hv[:, i]):.2f}  (share of best possible trade-off coverage)")

    # ---------- figure
    teer_g = Ftrue_grid[:, 0] * 100
    ldh_g = (1 - Ftrue_grid[:, 1]) * 100
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 17))
    ax = axes[0]
    ax.scatter(teer_g, ldh_g, s=6, c="lightgray", label="all possible conditions")
    sc = ax.scatter(teer_g[front_mask], ldh_g[front_mask], s=22, c=Ftrue_grid[front_mask, 2] * 0 + (b.DOSE_RANGE[0] + 4 * GRID[front_mask, 0]),
                    cmap="viridis", edgecolor="k", linewidth=0.3, label="true Pareto front")
    ax.axhspan(RULE_LDH_MAX, ldh_g.max() + 5, color="red", alpha=0.08)
    ax.axvline(RULE_TEER_MIN, color="gray", ls=":", lw=1)
    ax.text(2, RULE_LDH_MAX + 1.5, f"rule layer rejects LDH > {RULE_LDH_MAX:.0f}%", color="darkred", fontsize=8)
    ax.set_xlabel("TEER recovery (%)  [higher is better]"); ax.set_ylabel("LDH release (%)  [lower is better]")
    ax.set_title("A. Hidden truth: all conditions and the Pareto front"); fig.colorbar(sc, ax=ax, label="log10 dose (uM) on the front")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    Fr = f_true(rn_X); Fb = f_true(bo_X)
    ax.scatter(teer_g, ldh_g, s=5, c="lightgray")
    ax.scatter(Fr[:, 0] * 100, (1 - Fr[:, 1]) * 100, s=26, c="tab:gray", label="44 random wells")
    ax.scatter(Fb[:, 0] * 100, (1 - Fb[:, 1]) * 100, s=26, c="tab:blue", label="44 wells chosen by the optimizer")
    nd = nondominated(Fb)
    ax.scatter(Fb[nd, 0] * 100, (1 - Fb[nd, 1]) * 100, s=90, facecolor="none", edgecolor="k", label="non-dominated among optimizer wells")
    ax.axhspan(RULE_LDH_MAX, ldh_g.max() + 5, color="red", alpha=0.08); ax.axvline(RULE_TEER_MIN, color="gray", ls=":", lw=1)
    ax.set_xlabel("TEER recovery (%)"); ax.set_ylabel("LDH release (%)")
    ax.set_title("B. What each strategy actually tested (one campaign)"); ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    xs = np.array(checks)
    for arr, lab, col in ((rn_hv, "Random wells", "tab:gray"), (bo_hv, "Multi-objective optimizer", "tab:blue")):
        ax.plot(xs, np.median(arr, axis=0), color=col, label=lab, marker="o")
        ax.fill_between(xs, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0), color=col, alpha=0.2)
    ax.set_xlabel("wells run"); ax.set_ylabel("share of best possible trade-off coverage")
    ax.set_title("C. How much of the trade-off menu was found (30 repeats)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("fig4_pareto.png", dpi=130)

    # ---------- the menu from one real campaign (rules applied)
    dose_uM, timing = to_units(bo_X)
    T = Fb[:, 0] * 100; L = (1 - Fb[:, 1]) * 100
    ok = nd & (L <= RULE_LDH_MAX) & (T >= RULE_TEER_MIN)
    order = np.argsort(-T[ok])
    print("\nMENU (optimizer campaign, seed 1): non-dominated AND pass rules (true values)")
    print("dose uM | timing h | TEER % | LDH %")
    for i in np.where(ok)[0][order]:
        print(f"{dose_uM[i]:7.2f} | {timing[i]:8.1f} | {T[i]:6.1f} | {L[i]:5.1f}")
    rej = nd & ~ok
    print(f"\nNon-dominated but rejected by rules: {rej.sum()}")
    for i in np.where(rej)[0]:
        print(f"{dose_uM[i]:7.2f} | {timing[i]:8.1f} | {T[i]:6.1f} | {L[i]:5.1f}")
