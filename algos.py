"""
Offline dataset collection, the four learned policies and the evaluator.

All policies are tabular, so the only sources of randomness are the dataset
seed and the evaluation seeds, both of which are logged.
"""

import numpy as np
import env as E

GAMMA = 0.97
GAMMA_C = 0.95
VI_ITERS = 400


# ------------------------------------------------------------- dataset
def collect_dataset(seed, n_episodes, eps=0.25):
    """Roll out the behaviour policy and log (s, a, r, c, s', done)."""
    e = E.SafeLane(seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    S, A, R, C, S2, D = [], [], [], [], [], []
    for _ in range(n_episodes):
        s = e.reset()
        done = False
        while not done:
            a = E.behaviour_action(e, rng, eps=eps)
            s2, r, c, done, _ = e.step(a)
            S.append(s); A.append(a); R.append(r); C.append(c)
            S2.append(s2); D.append(float(done))
            s = s2
    return {"s": np.asarray(S), "a": np.asarray(A), "r": np.asarray(R),
            "c": np.asarray(C), "s2": np.asarray(S2), "d": np.asarray(D)}


def filter_dataset(data):
    """
    Cleaning step declared in the Methods: drop transitions whose logged
    reward or cost is non-finite, and drop the final truncated transition of
    each episode (done by horizon, not by collision), which carries no valid
    bootstrap target. Returns (clean_data, n_removed, reasons).
    """
    r, c, d = data["r"], data["c"], data["d"]
    finite = np.isfinite(r) & np.isfinite(c)
    # truncation: done==1 but no collision (collision gives r <= -7)
    truncated = (d == 1.0) & (r > -7.0)
    keep = finite & ~truncated
    reasons = {
        "non-finite reward or cost": int((~finite).sum()),
        "horizon-truncated last step": int((truncated & finite).sum()),
    }
    clean = {k: v[keep] for k, v in data.items() if isinstance(v, np.ndarray)}
    return clean, int((~keep).sum()), reasons


def empirical_model(data):
    """Counts, mean reward, mean cost and transition probabilities."""
    nS, nA = E.N_STATES, E.N_ACTIONS
    n = np.zeros((nS, nA))
    rsum = np.zeros((nS, nA))
    csum = np.zeros((nS, nA))
    P = np.zeros((nS, nA, nS))
    s, a, r, c, s2, d = (data["s"], data["a"], data["r"],
                         data["c"], data["s2"], data["d"])
    np.add.at(n, (s, a), 1.0)
    np.add.at(rsum, (s, a), r)
    np.add.at(csum, (s, a), c)
    np.add.at(P, (s, a, s2), 1.0 - d)          # terminal -> absorbing
    seen = n > 0
    rhat = np.where(seen, rsum / np.maximum(n, 1), 0.0)
    chat = np.where(seen, csum / np.maximum(n, 1), 0.0)
    tot = P.sum(axis=2, keepdims=True)
    P = np.divide(P, tot, out=np.zeros_like(P), where=tot > 0)
    return n, rhat, chat, P, seen


# ------------------------------------------------------------- policies
def fit_bc(data):
    """Behavioural cloning: per-state modal action, marginal-mode fallback."""
    cnt = np.zeros((E.N_STATES, E.N_ACTIONS))
    np.add.at(cnt, (data["s"], data["a"]), 1.0)
    fallback = int(np.argmax(cnt.sum(axis=0)))
    pi = np.full(E.N_STATES, fallback, dtype=int)
    seen = cnt.sum(axis=1) > 0
    pi[seen] = np.argmax(cnt[seen], axis=1)
    return pi


def fit_value_iteration(data, alpha):
    """
    Model-based offline value iteration with count-based pessimism.

    alpha = 0 reproduces the naive certainty-equivalent baseline: unvisited
    state-action pairs inherit the dataset-mean reward and a uniform model,
    so their values are extrapolated. alpha > 0 penalises low-count pairs by
    alpha / sqrt(n) and floors unvisited pairs, which is the pessimistic
    (conservative) offline RL family.
    """
    n, rhat, chat, P, seen = empirical_model(data)
    nS, nA = E.N_STATES, E.N_ACTIONS

    r_global = float(data["r"].mean())
    visited_states = np.flatnonzero(n.sum(axis=1) > 0)
    uni = np.zeros(nS)
    if visited_states.size:
        uni[visited_states] = 1.0 / visited_states.size

    R = np.where(seen, rhat, r_global)
    Pw = P.copy()
    unseen_idx = np.flatnonzero(~seen.ravel())
    if unseen_idx.size:
        flatP = Pw.reshape(nS * nA, nS)
        flatP[unseen_idx] = uni                     # naive uniform model
        Pw = flatP.reshape(nS, nA, nS)

    pen = np.where(seen, alpha / np.sqrt(np.maximum(n, 1.0)), 0.0)
    v_floor = (r_global - 8.0) / (1.0 - GAMMA)      # pessimistic floor

    Q = np.zeros((nS, nA))
    for _ in range(VI_ITERS):
        V = Q.max(axis=1)
        Q = R - pen + GAMMA * (Pw @ V)
        if alpha > 0:
            Q = np.where(seen, Q, v_floor)
    return Q.argmax(axis=1), Q, (n, chat, P, seen)


def fit_cost_critic(n, chat, P, seen, beta):
    """
    Pessimistic (upper-bound) cost-to-go used by the shield:
    Qc(s,a) = chat + beta/sqrt(n) + gamma_c * E[min_a' Qc(s',a')].
    Unvisited pairs are assigned the maximum cost-to-go, so the shield
    refuses actions the dataset cannot vouch for.
    """
    nS, nA = E.N_STATES, E.N_ACTIONS
    c_max = 1.0 / (1.0 - GAMMA_C)
    Cp = np.where(seen, chat + beta / np.sqrt(np.maximum(n, 1.0)), c_max)
    Qc = np.zeros((nS, nA))
    for _ in range(VI_ITERS):
        Vc = Qc.min(axis=1)
        Qc = Cp + GAMMA_C * (P @ Vc)
        Qc = np.where(seen, Qc, c_max)
    return np.clip(Qc, 0.0, c_max)


def shielded_policy(Qr, Qc, tau):
    """argmax reward-Q among actions whose cost-to-go is within tau."""
    allowed = Qc <= tau
    masked = np.where(allowed, Qr, -np.inf)
    pi = masked.argmax(axis=1)
    none_allowed = ~allowed.any(axis=1)
    pi[none_allowed] = Qc[none_allowed].argmin(axis=1)
    return pi


# ------------------------------------------------------------- evaluation
EVENTS = ["collision", "near_miss", "unsafe_cut_in", "invalid_lane"]


def evaluate(policy, n_episodes, seed, obs_noise=0.0, behaviour=False, eps=0.25):
    """
    Online evaluation in the environment. `policy` is an array of actions
    indexed by observed state, or None when behaviour=True.
    Returns per-episode arrays plus aggregated failure-mode counts.
    """
    e = E.SafeLane(seed=seed, obs_noise=obs_noise)
    rng = np.random.default_rng(seed + 777)
    rets, costs, crashes, lens = [], [], [], []
    ev = {k: 0 for k in EVENTS}
    gap_bucket_unsafe = np.zeros(E.N_GAP)
    gap_bucket_steps = np.zeros(E.N_GAP)

    for _ in range(n_episodes):
        s = e.reset()
        done, R, Csum, T = False, 0.0, 0.0, 0
        while not done:
            gb = (s // (E.N_SIDE * E.N_SIDE)) % E.N_GAP
            a = E.behaviour_action(e, rng, eps=eps) if behaviour else int(policy[s])
            s, r, c, done, info = e.step(a)
            R += r; Csum += c; T += 1
            gap_bucket_steps[gb] += 1
            if c > 0:
                gap_bucket_unsafe[gb] += 1
            if info["event"] in ev:
                ev[info["event"]] += 1
        rets.append(R); costs.append(Csum); lens.append(T)
        crashes.append(1.0 if info["collision"] else 0.0)

    return {
        "return": np.asarray(rets),
        "cost": np.asarray(costs),
        "crash": np.asarray(crashes),
        "len": np.asarray(lens),
        "events": ev,
        "gap_unsafe_rate": np.divide(gap_bucket_unsafe, np.maximum(gap_bucket_steps, 1)),
        "gap_occupancy": gap_bucket_steps / max(gap_bucket_steps.sum(), 1.0),
        "gap_steps": gap_bucket_steps,
    }
