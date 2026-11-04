"""Analysis of a small protein set with NumPy, run by the page's worker (docs/plan.md)."""

import json
import zlib

import numpy as np

AMINO = b"ACDEFGHIKLMNPQRSTVWY"
BACKGROUND = [0.0825, 0.0137, 0.0545, 0.0675, 0.0386, 0.0707, 0.0227, 0.0596, 0.0584, 0.0966,
              0.0242, 0.0406, 0.0470, 0.0393, 0.0553, 0.0656, 0.0534, 0.0687, 0.0108, 0.0292]
OTHER, SEPARATOR = 20, 21
ENCODE = np.full(256, OTHER, np.uint8)  # wasm32 indexes with int32, so not int64
ENCODE[np.frombuffer(AMINO, np.uint8)] = np.arange(20)
ENCODE[ord("*")] = SEPARATOR
MAX_BYTES = 400_000_000  # uncompressed input
MAX_RESIDUES = 20_000_000  # above this, wasm's 4 GB limit gets close (docs/plan.md)
MAX_K = 13  # 21**14 still fits an int64
FRACTIONS = (1 / 64, 1 / 16, 1 / 4, 1 / 2, 1)

state = {}


def gunzip(raw):
    if raw[:2] != b"\x1f\x8b":
        return raw
    d = zlib.decompressobj(wbits=31)
    out = d.decompress(raw, MAX_BYTES + 1)
    if len(out) > MAX_BYTES or d.unconsumed_tail:
        raise ValueError(f"Over {MAX_BYTES / 1e6:.0f} MB uncompressed")
    return out


def parse_fasta(raw):
    """Sequences of a FASTA file, plain or gzipped, in upper case."""
    text = gunzip(raw)
    if len(text) > MAX_BYTES:
        raise ValueError(f"Over {MAX_BYTES / 1e6:.0f} MB uncompressed")
    if text[:1] != b">":
        raise ValueError("Not FASTA: the file must start with '>'")
    seqs = []
    for record in text[1:].split(b"\n>"):
        _, _, body = record.partition(b"\n")
        seq = body.translate(None, b"\r\n \t*").upper()
        if seq:
            seqs.append(seq)
    if not seqs:
        raise ValueError("No sequences found")
    return seqs


def load(raw):
    """Parse and encode a set; returns its summary. Sets over the limit are sampled at random."""
    seqs = parse_fasta(bytes(raw))
    rng = np.random.default_rng(1)
    total, sampled_from = sum(map(len, seqs)), None
    if total > MAX_RESIDUES:
        sampled_from = len(seqs)
        order = rng.permutation(len(seqs))
        keep = order[np.cumsum(np.array([len(seqs[i]) for i in order])) <= MAX_RESIDUES]
        seqs = [seqs[i] for i in np.sort(keep)]
    if not seqs:
        raise ValueError(f"Over {MAX_RESIDUES:,} residues in one sequence")
    a = ENCODE[np.frombuffer(b"*".join(seqs), np.uint8)]
    lengths = np.array([len(s) for s in seqs], float)
    counts = np.bincount(a, minlength=22)
    f = counts[:20] / max(counts[:20].sum(), 1)
    if f[[0, 2 + 3, 1, 16]].sum() > 0.9:  # A, G, C, T
        raise ValueError("This looks like DNA; the analysis expects protein sequences")
    if counts[:20].sum() == 0:
        raise ValueError("No amino acids found")
    s2 = float((f**2).sum())
    per_protein = []
    for s in seqs[:: max(len(seqs) // 20000, 1)]:
        c = np.bincount(ENCODE[np.frombuffer(s, np.uint8)], minlength=22)[:20]
        per_protein.append(((c / max(c.sum(), 1)) ** 2).sum())
    state.clear()
    state.update(a=a, seq=np.cumsum(a == SEPARATOR), n=len(seqs), f=f, s2=s2)
    edges = np.geomspace(max(lengths.min(), 1), lengths.max() + 1, 40)
    hist, _ = np.histogram(lengths, edges)
    return json.dumps({
        "sequences": len(seqs),
        "sampled_from": sampled_from,
        "residues": int(counts[:20].sum()),
        "other_residues": int(counts[OTHER]),
        "length_quantiles": np.quantile(lengths, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
        "length_dispersion": float((lengths**2).mean() / lengths.mean() ** 2),
        "length_x": np.sqrt(edges[:-1] * edges[1:]).tolist(),
        "length_y": hist.tolist(),
        "amino": AMINO.decode(),
        "composition": f.tolist(),
        "background": BACKGROUND,
        "s2": s2,
        "protein_s2": np.quantile(per_protein, [0.05, 0.5, 0.95]).tolist(),
    })


def kmers(a, k):
    """(k-mers, the position of each) over runs of the 20 amino acids."""
    n = len(a) - k + 1
    if n <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    bad = np.convolve(a >= OTHER, np.ones(k, int), "valid") > 0
    x = np.zeros(n, np.int64)
    for i in range(k):
        x = x * 21 + a[i : n + i]
    return x[~bad], np.flatnonzero(~bad)


def collisions(x, y):
    """Chance a window of x equals a window of y, and the share from the top 1% of k-mers."""
    if not len(x) or not len(y):
        return 0.0, 0.0
    ux, cx = np.unique(x, return_counts=True)
    uy, cy = np.unique(y, return_counts=True)
    _, ix, iy = np.intersect1d(ux, uy, assume_unique=True, return_indices=True)
    h = np.sort(cx[ix].astype(float) * cy[iy])
    total = h.sum()
    if not total:
        return 0.0, 0.0
    return float(total / (len(x) * len(y))), float(h[-max(len(h) // 100, 1) :].sum() / total)


def independent(f, k, n, draws=200_000):
    """Expected (distinct k-mers, share of random windows held) for ``n`` windows of independent
    residues, from a sample of k-mers: E[(1 - exp(-n q)) / q] and E[1 - exp(-n q)], x ~ q."""
    if n <= 0:
        return 0.0, 0.0
    idx = np.random.default_rng(k).choice(20, (draws, k), p=f)
    q = np.prod(f[idx], axis=1)
    held = -np.expm1(-n * q)
    return float((held / q).mean()), float(held.mean())


def reversed_kmers(x, k):
    """The same windows read backwards."""
    r = np.zeros_like(x)
    for _ in range(k):
        r = r * 21 + x % 21
        x = x // 21
    return r


def spectrum(k):
    """One k's spectrum, decoys and collisions between random halves, with the model's values."""
    a, seq, s2, f = state["a"], state["seq"], state["s2"], state["f"]
    x, at = kmers(a, k)
    _, counts = np.unique(x, return_counts=True)
    counts = np.sort(counts)
    half = np.random.default_rng(k).random(state["n"]) < 0.5
    side = half[seq[at]]
    # Decoys: one half reversed against the other, as a whole set would match its own palindromes.
    decoy, decoy_top = collisions(x[side], reversed_kmers(x[~side], k))
    between, between_top = collisions(x[side], x[~side])
    return json.dumps({
        "k": k,
        "windows": len(x),
        "distinct": len(counts),
        "distinct_share": len(counts) / max(len(x), 1),
        "distinct_independent": independent(f, k, len(x))[0] / max(len(x), 1),
        "top1_postings": float(counts[-max(len(counts) // 100, 1) :].sum() / max(counts.sum(), 1)),
        "decoy_ratio": decoy / s2**k,
        "decoy_top1": decoy_top,
        "between_ratio": between / s2**k,
        "between_top1": between_top,
    })


def rarefy(k):
    """Distinct k-mers as sequences are subsampled (nested samples), and hits per window of one
    random half's reversed sequences on the other half's subsample."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be 1 to {MAX_K}")
    a, seq, n, f = state["a"], state["seq"], state["n"], state["f"]
    x, at = kmers(a, k)
    rank = np.random.default_rng(k).permutation(n)[seq[at]]
    side = np.random.default_rng(k + 1).random(n)[seq[at]] < 0.5
    decoy, counts = np.unique(reversed_kmers(x[~side], k), return_counts=True)
    rows = []
    for fraction in FRACTIONS:
        size = max(int(n * fraction), 1)
        sub = x[rank < size]
        distinct = np.unique(sub)
        target = x[(rank < size) & side]
        hits = counts[np.isin(decoy, np.unique(target), assume_unique=True)].sum()
        model_distinct = independent(f, k, len(sub))[0]
        model_hit = independent(f, k, len(target))[1]
        rows.append({
            "sequences": size,
            "windows": len(sub),
            "decoy_windows": len(target),
            "distinct": len(distinct),
            "distinct_independent": model_distinct,
            "decoy_hit": float(hits / max(counts.sum(), 1)),
            "decoy_hit_independent": model_hit,
        })
    return json.dumps({"k": k, "rows": rows})
