"""
Session 1: Bayesian optimization of a SIMULATED epithelial barrier-protection screen.

Biological setup (simulated, no real data):
  Injury lowers epithelial barrier function (TEER). A protective compound is added at some
  dose and at some time relative to injury. Readout = TEER recovery at 72 h, as % of uninjured baseline.
  Two experimental variables: log10 dose (uM) and timing relative to injury (h, negative = pre-treatment).

What this script shows:
  1. A hidden, noisy dose x timing landscape (with a decoy local optimum and a toxicity cliff).
  2. A Gaussian process surrogate (prediction + uncertainty), written in plain numpy.
  3. Expected improvement (EI) choosing each new plate of experiments.
  4. Comparison against random selection with the same experiment budget.

Design note: the problem framing, variables, and biological shape of the landscape are my design.
The code was written with Claude's help.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "."  # figures saved next to the script
DOSE_RANGE = (-2.0, 2.0)     # log10 uM  (0.01 to 100 uM)
TIME_RANGE = (-24.0, 24.0)   # hours relative to injury
NOISE_SD = 3.0               # assay noise, TEER % points

# ---------------------------------------------------------------- hidden "biology"
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def true_teer(u, v):
    """Noise-free TEER recovery (%) at normalized dose u and timing v, both in [0, 1]."""
    x = DOSE_RANGE[0] + (DOSE_RANGE[1] - DOSE_RANGE[0]) * np.asarray(u)
    t = TIME_RANGE[0] + (TIME_RANGE[1] - TIME_RANGE[0]) * np.asarray(v)
    baseline = 20.0                                                     # spontaneous recovery
    main = 70.0 * np.exp(-((x - 0.5) ** 2 / (2 * 0.45 ** 2) + (t + 6) ** 2 / (2 * 8 ** 2)))
    decoy = 55.0 * np.exp(-((x - 1.5) ** 2 / (2 * 0.30 ** 2) + (t - 14) ** 2 / (2 * 5 ** 2)))
    toxicity = 25.0 * _sigmoid((x - 1.4) / 0.2)                        # high doses hurt the cells
    return np.clip(baseline + main + decoy - toxicity, 0, 100)

def run_experiment(X, rng):
    """'Run' wells at X (n x 2, normalized). Returns noisy TEER recovery."""
    return true_teer(X[:, 0], X[:, 1]) + rng.normal(0, NOISE_SD, len(X))

# ---------------------------------------------------------------- Gaussian process (numpy only)
def rbf(A, B, ls):
    d = (A[:, None, :] - B[None, :, :]) / np.asarray(ls)
    return np.exp(-0.5 * np.sum(d ** 2, axis=2))

def fit_hyperparams(X, y):
    """Pick length scales and noise by maximizing log marginal likelihood on a small grid."""
    best, best_lml = None, -np.inf
    for l1 in (0.10, 0.15, 0.22, 0.32, 0.45):
        for l2 in (0.10, 0.15, 0.22, 0.32, 0.45):
            for nz in (0.005, 0.02, 0.06, 0.15):
                K = rbf(X, X, (l1, l2)) + nz * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, y))
                lml = -0.5 * y @ a - np.sum(np.log(np.diag(L))) - 0.5 * len(X) * math.log(2 * math.pi)
                if lml > best_lml:
                    best, best_lml = (l1, l2, nz), lml
    return best

def gp_posterior(X, y, Xs, hp):
    l1, l2, nz = hp
    K = rbf(X, X, (l1, l2)) + nz * np.eye(len(X))
    Ks = rbf(X, Xs, (l1, l2))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ a
    v = np.linalg.solve(L, Ks)
    var = np.clip(1.0 - np.sum(v ** 2, axis=0), 1e-12, None)
    return mu, np.sqrt(var)

# ---------------------------------------------------------------- acquisition
def _phi(z):  return np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
def _Phi(z):  return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))

def expected_improvement(mu, sd, best, xi=0.01):
    z = (mu - best - xi) / sd
    return (mu - best - xi) * _Phi(z) + sd * _phi(z)

GRID_N = 40
g = np.linspace(0, 1, GRID_N)
GRID = np.array([[a, b] for b in g for a in g])   # candidate experiments

def lhs(n, rng):
    """Latin hypercube: spread first-plate wells evenly across both variables."""
    cols = []
    for _ in range(2):
        perm = rng.permutation(n)
        cols.append((perm + rng.uniform(0, 1, n)) / n)
    return np.column_stack(cols)

def propose_batch(X, y_raw, batch, rng_unused=None):
    """Pick `batch` new experiments by greedy EI with fantasized outcomes (kriging believer)."""
    mu0, sd0 = y_raw.mean(), y_raw.std() + 1e-9
    y = (y_raw - mu0) / sd0
    hp = fit_hyperparams(X, y)
    Xf, yf = X.copy(), y.copy()
    picks = []
    for _ in range(batch):
        mu, sd = gp_posterior(Xf, yf, GRID, hp)
        mu_obs, _ = gp_posterior(Xf, yf, Xf, hp)
        ei = expected_improvement(mu, sd, mu_obs.max())
        ei[[i for i, p in enumerate(GRID) if any(np.allclose(p, q) for q in picks)]] = -1
        k = int(np.argmax(ei))
        picks.append(GRID[k])
        Xf = np.vstack([Xf, GRID[k]])
        yf = np.append(yf, mu[k])          # fantasize: assume the model's own prediction
    return np.array(picks), hp, (mu0, sd0)

# ---------------------------------------------------------------- campaigns
def true_at_best_measured(X, y):
    """The compound/condition you'd actually advance = best MEASURED well. How good is it truly?"""
    return float(true_teer(X[np.argmax(y)][0], X[np.argmax(y)][1]))

def campaign_bo(seed, first=8, rounds=6, batch=6, keep_state=False):
    rng = np.random.default_rng(seed)
    X = lhs(first, rng)
    y = run_experiment(X, rng)
    trace = [true_at_best_measured(X[:i], y[:i]) for i in range(1, first + 1)]
    states = [(X.copy(), y.copy())]
    for _ in range(rounds):
        newX, hp, _ = propose_batch(X, y, batch)
        newy = run_experiment(newX, rng)
        for i in range(batch):
            trace.append(true_at_best_measured(np.vstack([X, newX[: i + 1]]), np.append(y, newy[: i + 1])))
        X, y = np.vstack([X, newX]), np.append(y, newy)
        states.append((X.copy(), y.copy()))
    return (trace, states) if keep_state else trace

def campaign_random(seed, n_total):
    rng = np.random.default_rng(10_000 + seed)
    X = rng.uniform(0, 1, (n_total, 2))
    y = run_experiment(X, rng)
    return [true_at_best_measured(X[:i], y[:i]) for i in range(1, n_total + 1)]

# ---------------------------------------------------------------- plotting helpers
def _axes(ax):
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels([f"{DOSE_RANGE[0] + (DOSE_RANGE[1]-DOSE_RANGE[0])*q:.0f}" for q in np.linspace(0, 1, 5)])
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f"{TIME_RANGE[0] + (TIME_RANGE[1]-TIME_RANGE[0])*q:.0f}" for q in np.linspace(0, 1, 5)])
    ax.set_xlabel("log10 dose (uM)")
    ax.set_ylabel("timing vs injury (h)")

def plot_truth():
    Z = true_teer(GRID[:, 0], GRID[:, 1]).reshape(GRID_N, GRID_N)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    _axes(ax)
    ax.set_title("Hidden truth (the experimenter never sees this)")
    ax.annotate("true optimum", xy=(0.6, 0.375), xytext=(0.64, 0.13), color="white",
                arrowprops=dict(arrowstyle="->", color="white"))
    ax.annotate("decoy optimum", xy=(0.875, 0.79), xytext=(0.55, 0.9), color="white",
                arrowprops=dict(arrowstyle="->", color="white"))
    ax.text(0.98, 0.03, "toxicity at high dose", color="white", ha="right", fontsize=8)
    fig.colorbar(im, label="TEER recovery at 72 h (% of baseline)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_hidden_truth.png", dpi=150); plt.close(fig)

def plot_rounds(states, seed):
    show = [0, 2, 6]
    titles = ["After plate 1 (8 wells, spread evenly)", "After round 2 (20 wells)", "After round 6 (44 wells)"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for col, (r, ttl) in enumerate(zip(show, titles)):
        X, y_raw = states[r]
        mu0, sd0 = y_raw.mean(), y_raw.std() + 1e-9
        y = (y_raw - mu0) / sd0
        hp = fit_hyperparams(X, y)
        mu, sd = gp_posterior(X, y, GRID, hp)
        mu_obs, _ = gp_posterior(X, y, X, hp)
        ei = expected_improvement(mu, sd, mu_obs.max())
        panels = [(mu * sd0 + mu0, "Model prediction (TEER %)", "viridis", (0, 100)),
                  (sd * sd0, "Model uncertainty (SD, TEER %)", "magma", None),
                  (ei, "Expected improvement (where to test next)", "cividis", None)]
        for row, (Z, name, cmap, lim) in enumerate(panels):
            ax = axes[row, col]
            kw = dict(vmin=lim[0], vmax=lim[1]) if lim else {}
            im = ax.imshow(Z.reshape(GRID_N, GRID_N), origin="lower", extent=(0, 1, 0, 1),
                           aspect="auto", cmap=cmap, **kw)
            ax.scatter(X[:, 0], X[:, 1], c="white", edgecolor="black", s=22)
            _axes(ax)
            ax.set_title((ttl + "\n" if row == 0 else "") + name, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_model_over_rounds.png", dpi=130); plt.close(fig)

def plot_convergence(bo, rnd):
    bo, rnd = np.array(bo), np.array(rnd)
    n = bo.shape[1]
    xs = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arr, lab, col in ((rnd, "Random wells", "tab:gray"), (bo, "Bayesian optimization (EI)", "tab:blue")):
        med = np.median(arr, axis=0)
        ax.plot(xs, med, label=lab, color=col)
        ax.fill_between(xs, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0), color=col, alpha=0.2)
    ax.axhline(true_teer(0.625, 0.375), ls="--", color="k", lw=0.8)
    ax.text(n, true_teer(0.625, 0.375) + 0.5, "true optimum", ha="right", fontsize=8)
    ax.set_xlabel("wells run"); ax.set_ylabel("true TEER recovery of the condition you would advance (%)")
    ax.set_title("Reaching a good condition with fewer wells (median and IQR, 30 repeats)")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_bo_vs_random.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    plot_truth()
    trace, states = campaign_bo(seed=1, keep_state=True)
    plot_rounds(states, seed=1)
    n_total = 8 + 6 * 6
    bo = [campaign_bo(s) for s in range(30)]
    rnd = [campaign_random(s, n_total) for s in range(30)]
    plot_convergence(bo, rnd)
    bo, rnd = np.array(bo), np.array(rnd)
    for k in (8, 20, 32, 44):
        print(f"{k:>3} wells | BO median true TEER {np.median(bo[:, k-1]):5.1f} | random median {np.median(rnd[:, k-1]):5.1f}")
    opt = float(true_teer(0.625, 0.375))
    print(f"true optimum approx {opt:.1f}")
