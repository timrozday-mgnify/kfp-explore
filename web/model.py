"""The analytic anchor model, batched with NumPy (docs/plan.md, The model)."""

import json
from functools import lru_cache

import numpy as np

# Conservation fitted to kfp's 90%-AAI table on m 8: conserved share, rate ratio, block length.
CONSERVATION = (0.9, 32.0, 40.0)
INDEPENDENT = (0.5, 1.0, 10.0)
READ_RATE = 4.0  # shape of the rate each read shares across everything it hits (measure/fit_rate.py)
RATE_NODES = 16
CEILING = 0.96  # kfp's recall against representatives of chained groups, whatever the k-mers
S2 = 0.0597  # chance that two residues agree, amino-acid background
AMINO = "ACDEFGHIKLMNPQRSTVWY"
BACKGROUND = np.array([0.0825, 0.0137, 0.0545, 0.0675, 0.0386, 0.0707, 0.0227, 0.0596, 0.0584, 0.0966,
                       0.0242, 0.0406, 0.0470, 0.0393, 0.0553, 0.0656, 0.0534, 0.0687, 0.0108, 0.0292])
# BLOSUM62 half-bit scores in AMINO's order.
BLOSUM62 = np.array([
    [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2],
    [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3],
    [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2],
    [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3],
    [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3],
    [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2],
    [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1],
    [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2],
    [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
    [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1],
    [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2],
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3],
    [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1],
    [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2],
    [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2],
    [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2],
    [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1],
    [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
    [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7],
])
# kfp's classes (crates/core/src/alphabet.rs); "protein" and "dna" are handled by name.
REDUCED = {
    "murphy15": "LVIM C A G S T P FY W E Q D N KR H",
    "seb14": "A C D EQ FY G H IV KR LM N P ST W",
    "opt14": "FWY AC EQ HN LM D G I K P R S T V",
    "murphy12": "LVIM C A G ST P FY W EQ DN KR H",
    "opt12": "FWY AC EQ HN IV KR LM D G P S T",
    "diamond11": "KREDQN C G H ILV M F Y W P STA",
    "murphy10": "LVIM C A G ST P FYW EDNQ KR H",
    "seb10": "AST C DN EQ FY G HW ILMV KR P",
    "sev10": "AST C DEN FY G H ILMV KQR P W",
    "lia10": "AC DE FWY G HN IV KQR LM P ST",
    "lib10": "AST C DEQ FWY G HN IV KR LM P",
    "opt10": "FHWY AC DN EQ IV KR LM PT G S",
    "murphy8": "LVIMC AG ST P FYW EDNQ KR H",
    "seb8": "AST C DHN EKQR FWY G ILMV P",
    "opt8": "FHWY DEQ KNR AC GS IV LM PT",
    "dayhoff": "AGPST C DENQ HKR ILMV FWY",
    "seb6": "AST CP DEHKNQR FWY G ILMV",
    "opt6": "FHWY ILMV ACT DEQ GNS KPR",
    "opt4": "CFILMVWY DEHKNQR PST AG",
}
WITHIN_SCALE = 1.6  # real substitutions stay within classes more than BLOSUM62 says (checks/test_grid.py)
MAX_K = {"dna": 51}  # 31 otherwise
DNA_S2 = 0.25  # ponytail: uniform bases; GC-rich sets are higher
LENGTH_FACTOR = {"dna": 3}  # letters per protein residue
# Chance over composition alone, measured: kfp's table at k 6-9, the mini-slice pilot at k 5.
HOT = {5: 1.2, 6: 1.3, 7: 1.5, 8: 1.4, 9: 3.6}
MEAN_LENGTH = 225.0  # with a representative share of 0.155, kfp's ~2e11 residues at release
BYTES_PER_POSTING = 6.0  # anchor design: group id, offset, Elias-Fano keys
FRAMES = 3  # ponytail: flat, as kfp's table; replace with measured frames per read


def alphabet_stats(name):
    """(letters, chance two letters agree, share of substitutions staying in a class).

    Substitutions are weighted by BLOSUM62's target frequencies, f_i f_j 2^(S_ij / 2), then scaled
    as fitted to simulated reads."""
    if name == "dna":
        return 4, DNA_S2, 0.0
    groups = REDUCED[name].split() if name != "protein" else list(AMINO)
    index = {a: g for g, members in enumerate(groups) for a in members}
    group = np.array([index[a] for a in AMINO])
    freq = np.bincount(group, BACKGROUND, len(groups))
    same = group[:, None] == group[None, :]
    off = ~np.eye(20, dtype=bool)
    q = BACKGROUND[:, None] * BACKGROUND[None, :] * 2.0 ** (BLOSUM62 / 2)
    within = float(q[same & off].sum() / q[off].sum())
    return len(groups), float((freq**2).sum() / BACKGROUND.sum() ** 2), min(within * WITHIN_SCALE, 1.0)


def classes(p, phi, rho):
    """Identity of the conserved and variable classes, exp(-r t), with mean identity p."""
    p = np.asarray(p, float)
    lo, hi = np.zeros_like(p), np.full_like(p, 200.0)
    for _ in range(60):
        t = (lo + hi) / 2
        above = phi * np.exp(-t) + (1 - phi) * np.exp(-rho * t) > p
        lo, hi = np.where(above, t, lo), np.where(above, hi, t)
    return np.exp(-lo), np.exp(-rho * lo)


def miss(length, k, m, p, conservation=CONSERVATION, within=0.0):
    """P(no sampled k-mer match in ``length`` letters) for arrays m and p of one shape.

    The chain's state is (site class, match run capped at k); a matching k-mer is kept at 1/m.
    ``within``: the share of substitutions an alphabet does not see."""
    phi, rho, ell = conservation
    m, p = np.broadcast_arrays(np.asarray(m, float), np.asarray(p, float))
    id0, id1 = classes(p.ravel(), phi, rho)
    id0, id1 = id0 + (1 - id0) * within, id1 + (1 - id1) * within
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


@lru_cache
def rates(shape, nodes=RATE_NODES):
    """Equal-weight nodes of a read's rate, Gamma(shape, 1 / shape): identity becomes identity^rate.

    A read in a conserved region is close to everything it can hit, so hits are independent only
    given the rate. ponytail: quantiles from a fixed sample, as Pyodide's NumPy has no gamma ppf."""
    draws = np.sort(np.random.default_rng(1).gamma(shape, 1 / shape, 100_000))
    return draws[((np.arange(nodes) + 0.5) / nodes * draws.size).astype(int)]


def recall(
    length, k, m, p, paired=True, conservation=CONSERVATION, ceiling=CEILING, shape=READ_RATE, within=0.0
):
    """P(a read, or a pair of mates of ``length`` letters, has an anchor on its true group)."""
    m, p = np.broadcast_arrays(np.asarray(m, float), np.asarray(p, float))
    if shape is None:
        q = miss(length, k, m, p, conservation, within)
        return ceiling * (1 - (q**2 if paired else q))
    lam = rates(shape)
    local = np.clip(p[..., None] ** lam, 1e-4, 1 - 1e-9)
    q = miss(length, k, m[..., None], local, conservation, within)
    return ceiling * (1 - (q**2 if paired else q)).mean(-1)


def hit_both(length, k, m, p, x, share, conservation=CONSERVATION, shape=READ_RATE):
    """(P(true group missed), P(true group missed and a relative hit)) for a read pair, at identity
    p to the true group and x to the relative, arrays of one shape.

    A k-mer both groups hold is kept or dropped for both, so a relative is hit after a miss only by
    k-mers the true group lacks. ``share`` (one value, or an array of p's shape) is the part of the relative's matches that are also the
    true group's: 1 nests them, 0 makes them independent. The chain's state adds the relative's run."""
    phi, rho, ell = conservation
    p, x = np.broadcast_arrays(np.asarray(p, float), np.asarray(x, float))
    lam = rates(shape)
    a = np.clip(p[..., None] ** lam, 1e-4, 1 - 1e-9).ravel()
    b = np.clip(x[..., None] ** lam, 1e-4, 1 - 1e-9).ravel()
    keep = np.broadcast_to(1 - 1 / np.asarray(m, float)[..., None], (*p.shape, lam.size)).ravel()
    share = np.broadcast_to(np.asarray(share, float)[..., None], (*p.shape, lam.size)).ravel()
    to_v = 1 / ell
    to_c = to_v * phi / (1 - phi)
    ids = [classes(a, phi, rho), classes(b, phi, rho)]
    s = [np.zeros((a.size, k + 1, k + 1)), np.zeros((a.size, k + 1, k + 1))]
    s[0][:, 0, 0], s[1][:, 0, 0] = phi, 1 - phi
    alone = [np.full(a.size, phi), np.full(a.size, 1 - phi)]
    for _ in range(length):
        q = [(1 - to_v) * s[0] + to_c * s[1], to_v * s[0] + (1 - to_c) * s[1]]
        for c in (0, 1):
            pa, pb = ids[0][c], ids[1][c]
            both = share * np.minimum(pa, pb) + (1 - share) * pa * pb
            q[c] = pair_step(q[c], both, pa - both, pb - both, keep)
        s = q
    neither = (s[0].sum((1, 2)) + s[1].sum((1, 2))).reshape(*p.shape, -1)
    missed = miss(length, k, m, np.clip(p[..., None] ** lam, 1e-4, 1 - 1e-9), conservation)
    return (missed**2).mean(-1), (missed**2 - neither**2).mean(-1)


def pair_step(q, both, only_a, only_b, keep):
    """One residue for two match runs at once. Both runs at k are one k-mer, so one decision."""
    k = q.shape[1] - 1
    by_a, by_b = q.sum(2), q.sum(1)
    n = np.zeros_like(q)
    n[:, 0, 0] = q.sum((1, 2)) * (1 - both - only_a - only_b)
    n[:, 1:, 0] = by_a[:, :-1] * only_a[:, None]
    n[:, k, 0] += by_a[:, k] * only_a
    n[:, 0, 1:] = by_b[:, :-1] * only_b[:, None]
    n[:, 0, k] += by_b[:, k] * only_b
    extended = np.zeros_like(q)
    extended[:, 1:, 1:] = q[:, :-1, :-1]
    extended[:, k, 1:] += q[:, k, :-1]
    extended[:, 1:, k] += q[:, :-1, k]
    extended[:, k, k] += q[:, k, k]
    n += extended * both[:, None, None]
    n[:, k, :] *= keep[:, None]
    n[:, :k, k] *= keep[:, None]
    return n


def postings(proteins, rep_share, m, alphabet="protein"):
    """Index postings: representative letters, 1 in m."""
    return proteins * rep_share * MEAN_LENGTH * LENGTH_FACTOR.get(alphabet, 1) / m


def chance_anchors(length, k, n_postings, paired=True, alphabet="protein"):
    """Chance anchors per read from composition, and with the measured correction (or None)."""
    _, s2, _ = alphabet_stats(alphabet)
    frames = 2 if alphabet == "dna" else FRAMES  # DNA: both strands
    base = (2 if paired else 1) * max(length - k + 1, 0) * frames * n_postings * s2**k
    return base, base * HOT[k] if alphabet == "protein" and k in HOT else None


# Precision from shared k-mer tables (measure/precision_tables.py).
TABLES = {}
NATURAL_MIX = [0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.8]  # kfp's nearest relatives, full mini-slice
CHANCE_GROWTH = 0.6  # distinct chance groups grow as reads^this (kfp: 0.5-0.7)
CHANCE_FROM_READS = 261_591  # the read pairs that exponent starts from in kfp's test
WRONG_FRAMES = 2  # protein reads are in frame; real reads also meet chance in 2 more (FRAMES in all)


def load_table(name, text):
    """Arrays for one index. Cubes are by level, floor, identity bin and band; band 0 is the true
    group, then relative identity bands, then unknown identity. Best-group cubes add a margin first."""
    t = json.loads(text)
    bins, bands = len(t["pairs_by_bin"]), t["n_bands"]
    for entry in t["k"].values():
        entry["cells"] = np.array(entry["cells"], float).reshape(-1, 4)
        entry["decoy"] = np.array(entry["decoy"], float).reshape(-1, 2)
        cube = np.array(entry["floor_cube"], float) / t["salts"]
        entry["floor_cube"] = cube.reshape(*cube.shape[:2], bins, bands)
        best = np.array(entry["best_cube"], float) / t["salts"]
        entry["best_cube"] = best.reshape(*best.shape[:3], bins, bands)
        entry["floor_query_pass"] = np.array(entry["floor_query_pass"], float) / t["salts"]
        entry["floor_decoys"] = np.array(entry["floor_decoys"], float) / t["salts"] / t["pairs"]
        entry["floor_precision_sd"] = np.array(entry["floor_precision_sd"])
        entry["floor_rows"] = np.array(entry["floor_rows"], float).reshape(bins, bands)
    TABLES[name] = t


def correct_bands(t, truth):
    """Which bands count as correct other than the true group: relatives at identity >= truth."""
    edges = np.array(t["x_edges"][:-1])
    return np.r_[False, edges >= truth, False]


def floor_rates(entry, t, level, truth, weights, rule=None):
    """Per read pair, per floor: (true hits, other correct hits, wrong hits). ``rule``: a margin's
    index for the best-group rule, or None for every group over the floor."""
    cube = entry["floor_cube"][level] if rule is None else entry["best_cube"][rule, level]
    pairs = np.maximum(np.array(t["pairs_by_bin"], float), 1)
    per_band = (cube * (np.asarray(weights) / pairs)[None, :, None]).sum(1)
    ok = correct_bands(t, truth)
    return per_band[:, 0], per_band[:, ok].sum(1), per_band[:, 1:][:, ~ok[1:]].sum(1)


def target_rates(entry, t, level, f, truth, weights, depth):
    """Per target at ``depth`` read pairs each, at one floor: (targets called, other correct groups,
    wrong groups). A group is called by any read pair. The true group's rate is each query's own, as
    a target's reads share its conservation; a relative's read pairs are taken as independent."""
    weight = np.asarray(weights, float)
    counts = entry["floor_query_pass"][level, f]  # queries by bin and read pairs passing
    n_pairs = counts.shape[1] - 1
    rate = np.arange(n_pairs + 1) / n_pairs
    by_bin = (counts * (1 - (1 - rate) ** depth)).sum(1) / np.maximum(counts.sum(1), 1e-12)
    called = float((weight * by_bin).sum())
    rows = np.maximum(entry["floor_rows"], 1)  # (read pair, group) rows sharing any k-mer
    hit = entry["floor_cube"][level, f] / rows
    pairs = np.maximum(np.array(t["pairs_by_bin"], float), 1)
    groups = entry["floor_rows"] * (weight / pairs)[:, None]  # such groups per read pair
    called_band = (groups * (1 - (1 - np.minimum(hit, 1)) ** depth)).sum(0)
    ok = correct_bands(t, truth)
    return called, float(called_band[ok].sum()), float(called_band[1:][~ok[1:]].sum())


def compute_precision(a):
    """The precision panel, from one index's tables: every floor, at the selected rule."""
    t = TABLES[a["index"]]
    k, m = str(a["k"]), a["m"]
    if k not in t["k"]:
        return {"available": False, "reason": "k", "ks": sorted(map(int, t["k"]))}
    level = int(round(np.log2(m)))
    entry = t["k"][k]
    floors = t["floors"]
    f = int(a["floor"])
    n_bins = len(t["pairs_by_bin"])
    mix = np.array(NATURAL_MIX if a["mix"] == "natural" else [1.0] * (n_bins - 1), float)
    weights = np.r_[mix / mix.sum() * (1 - a["no_relative"]), a["no_relative"]]
    truth = a["truth"]
    margin = a.get("margin", -1)
    rule = t["margins"].index(margin) if margin in t["margins"] else None

    def evaluate(e):
        true_hits, other_ok, wrong = floor_rates(e, t, level, truth, weights, rule)
        wrong = wrong + e["floor_decoys"][level] * WRONG_FRAMES
        correct = true_hits + other_ok
        precision = correct / np.maximum(correct + wrong, 1e-12)
        recall = true_hits / max(1 - a["no_relative"], 1e-12)
        return recall, precision, wrong + other_ok

    recall, precision, others = evaluate(entry)
    # AUPRC by floors, from recall 0 up: each step's precision over its gain in recall.
    r, q = recall[::-1], precision[::-1]
    auprc = float((np.diff(np.r_[0, np.maximum.accumulate(r)]) * q).sum() / max(r.max(), 1e-12))
    by_k = {}
    for other_k, e in t["k"].items():
        ri, pi, _ = evaluate(e)
        by_k[other_k] = {"recall": ri.tolist(), "precision": pi.tolist()}
    n_reads, targets = a["reads"], a["targets"]
    depth = n_reads / max(targets, 1)
    called, called_ok, called_wrong = target_rates(entry, t, level, f, truth, weights, depth)
    per_read_chance = float(entry["floor_decoys"][level, f]) * FRAMES
    from_reads = min(n_reads, CHANCE_FROM_READS)
    chance_groups = per_read_chance * from_reads * (n_reads / from_reads) ** CHANCE_GROWTH / max(targets, 1)
    called_correct = called + called_ok
    sd = float(entry["floor_precision_sd"][level, f]) if rule is None and truth >= 1 else None
    return {
        "available": True,
        "floors": floors,
        "floor": floors[f],
        "recall": float(recall[f]),
        "precision": float(precision[f]),
        "others": float(others[f]),
        "chance": per_read_chance,
        "auprc": auprc,
        "hash_sd": sd,
        "key_bits": entry["key_bits"],
        "bits_per_hit": entry["bits_per_true_hit"][level],
        "charge": entry["charge"],
        "curve_recall": recall.tolist(),
        "curve_precision": precision.tolist(),
        "by_k": by_k,
        "depth": depth,
        "target_recall": called / max(1 - a["no_relative"], 1e-12),
        "target_precision": called_correct / max(called_correct + called_wrong + chance_groups, 1e-12),
    }


def compute(params_json):
    """Everything the page draws, as JSON."""
    a = json.loads(params_json)
    k, m, length, identity, paired = a["k"], a["m"], a["length"], a["identity"], a["paired"]
    ps = np.round(np.arange(0.5, 1.001, 0.01), 2)
    ms = 2.0 ** np.arange(0, 14)
    alphabet = a["alphabet"]
    letters, s2, within = alphabet_stats(alphabet)
    n_postings = postings(a["proteins"], a["rep_share"], m, alphabet)
    base, corrected = chance_anchors(length, k, n_postings, paired, alphabet)
    nearby = sorted({j for j in range(k - 4, k + 5, 2) if 1 <= j <= MAX_K.get(alphabet, 31)})
    model = {"paired": paired, "within": within}
    return json.dumps({
        "precision": (
            compute_precision(a) if alphabet == "protein" and a.get("index") in TABLES else {"available": False}
        ),
        "identity": ps.tolist(),
        "by_identity": recall(length, k, m, ps, **model).tolist(),
        "by_identity_independent": recall(length, k, m, ps, paired, INDEPENDENT, 1.0, None, within).tolist(),
        "ms": ms.tolist(),
        "by_m": {str(j): recall(length, j, ms, identity, **model).tolist() for j in nearby},
        "recall": float(recall(length, k, m, identity, **model)),
        "letters": letters,
        "bits_per_letter": -float(np.log2(s2)),
        "bits_per_kmer": -float(k * np.log2(s2)),
        "within": within,
        "postings": n_postings,
        "index_gb": n_postings * BYTES_PER_POSTING / 1e9,
        "chance": base,
        "chance_corrected": corrected,
    })
