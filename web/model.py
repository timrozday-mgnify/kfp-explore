"""The analytic anchor model, batched with NumPy (docs/plan.md, The model)."""

import json

import numpy as np

# Conservation fitted to kfp's 90%-AAI table on m 8: conserved share, rate ratio, block length.
CONSERVATION = (0.9, 32.0, 40.0)
INDEPENDENT = (0.5, 1.0, 10.0)
CEILING = 0.96  # kfp's recall against representatives of chained groups, whatever the k-mers
S2 = 0.0597  # chance that two residues agree, amino-acid background
# Chance over composition alone, measured: kfp's table at k 6-9, the mini-slice pilot at k 5.
HOT = {5: 1.2, 6: 1.3, 7: 1.5, 8: 1.4, 9: 3.6}
MEAN_LENGTH = 225.0  # with a representative share of 0.155, kfp's ~2e11 residues at release
BYTES_PER_POSTING = 6.0  # anchor design: group id, offset, Elias-Fano keys
FRAMES = 3  # ponytail: flat, as kfp's table; replace with measured frames per read


def classes(p, phi, rho):
    """Identity of the conserved and variable classes, exp(-r t), with mean identity p."""
    p = np.asarray(p, float)
    lo, hi = np.zeros_like(p), np.full_like(p, 200.0)
    for _ in range(60):
        t = (lo + hi) / 2
        above = phi * np.exp(-t) + (1 - phi) * np.exp(-rho * t) > p
        lo, hi = np.where(above, t, lo), np.where(above, hi, t)
    return np.exp(-lo), np.exp(-rho * lo)


def miss(length, k, m, p, conservation=CONSERVATION):
    """P(no sampled k-mer match in ``length`` residues) for arrays m and p of one shape.

    The chain's state is (site class, match run capped at k); a matching k-mer is kept at 1/m."""
    phi, rho, ell = conservation
    m, p = np.broadcast_arrays(np.asarray(m, float), np.asarray(p, float))
    id0, id1 = classes(p.ravel(), phi, rho)
    keep = 1 - 1 / m.ravel()
    to_v = 1 / ell
    to_c = to_v * phi / (1 - phi)
    s0, s1 = np.zeros((id0.size, k + 1)), np.zeros((id0.size, k + 1))
    s0[:, 0], s1[:, 0] = phi, 1 - phi
    for _ in range(length):
        q0, q1 = (1 - to_v) * s0 + to_c * s1, to_v * s0 + (1 - to_c) * s1
        s0, s1 = step(q0, id0, keep), step(q1, id1, keep)
    return (s0.sum(1) + s1.sum(1)).reshape(p.shape)


def step(q, identity, keep):
    """One residue's emission for one site class."""
    n = np.empty_like(q)
    n[:, 0] = q.sum(1) * (1 - identity)
    n[:, 1:-1] = q[:, :-2] * identity[:, None]
    n[:, -1] = (q[:, -2] + q[:, -1]) * identity * keep
    return n


def recall(length, k, m, p, paired=True, conservation=CONSERVATION, ceiling=CEILING):
    """P(a read, or a pair of mates of ``length`` residues, has an anchor on its true group)."""
    q = miss(length, k, m, p, conservation)
    return ceiling * (1 - (q**2 if paired else q))


def postings(proteins, rep_share, m):
    """Index postings: representative residues, 1 in m."""
    return proteins * rep_share * MEAN_LENGTH / m


def chance_anchors(length, k, n_postings, paired=True):
    """Chance anchors per read from composition, and with the measured correction (or None)."""
    base = (2 if paired else 1) * max(length - k + 1, 0) * FRAMES * n_postings * S2**k
    return base, base * HOT[k] if k in HOT else None


def compute(params_json):
    """Everything the page draws, as JSON."""
    a = json.loads(params_json)
    k, m, length, identity, paired = a["k"], a["m"], a["length"], a["identity"], a["paired"]
    ps = np.round(np.arange(0.5, 1.001, 0.01), 2)
    ms = 2.0 ** np.arange(0, 9)
    n_postings = postings(a["proteins"], a["rep_share"], m)
    base, corrected = chance_anchors(length, k, n_postings, paired)
    return json.dumps({
        "identity": ps.tolist(),
        "by_identity": recall(length, k, m, ps, paired).tolist(),
        "by_identity_independent": recall(length, k, m, ps, paired, INDEPENDENT, 1.0).tolist(),
        "ms": ms.tolist(),
        "by_m": {
            str(j): recall(length, j, ms, identity, paired).tolist() for j in range(5, 12)
        } | {str(k): recall(length, k, ms, identity, paired).tolist()},
        "recall": float(recall(length, k, m, identity, paired)),
        "postings": n_postings,
        "index_gb": n_postings * BYTES_PER_POSTING / 1e9,
        "chance": base,
        "chance_corrected": corrected,
    })
