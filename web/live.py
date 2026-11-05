"""Precision on demand: what measure/precision_tables.py does for one k, m and read length, from a
small exported file. One sampling level at a time, so nothing is computed that is not asked for.
Results have the tables' shape, one level only, so model.compute_precision reads them unchanged."""

import io
import json

import numpy as np

import analyse
import model

BINS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0001]
X_EDGES = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
N_BANDS = len(X_EDGES) + 1  # the true group, relative identity bands, unknown identity
FLOORS = (0, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200)
MARGINS = (0, 3, 10)
SALTS = 8
PAIRS_PER_QUERY = 3
MAX_GAP = 40
TRANSLATED_FRAMES = 6
LEVELS = 14
HIT_LIMIT = 15_000_000  # (read pair, group) k-mer hits: short k-mers are held by most groups, and wasm's memory is limited
COVERAGES = (1, 2, 5, 10, 20, 50, 100)  # read coverage of a target
TILE_GAP = 20  # between a tiled pair's mates; a random read's is 0 to MAX_GAP

DATA = {}  # index name: arrays
ENTRIES = {}  # (index, alphabet, length, paired, level): {k: entry}, and "meta"
_index_memo, _rows_memo = {}, {}
_loaded = None


def load(name, raw):
    with np.load(io.BytesIO(bytes(raw))) as z:
        DATA[name] = {key: z[key] for key in z.files}
    d = DATA[name]
    scale = int(d["groups"]) + 2
    key = d["aligned_q"].astype(np.int64) * scale + d["aligned_g"].astype(np.int64)
    o = np.argsort(key)
    d["aligned_key"], d["aligned_x"] = key[o], d["aligned_x"][o].astype(float) / 250
    d["query_list"] = bytes(d["queries"]).split(b"*")


def encoding(alphabet):
    table = analyse.ENCODE.copy()
    if alphabet != "protein":
        for g, members in enumerate(model.REDUCED[alphabet].split()):
            for aa in members:
                table[ord(aa)] = g
    return table


def build_index(name, alphabet, k):
    """Sorted distinct (k-mer, group), each k-mer's first row and holders, and its key bits."""
    memo = (name, alphabet, k)
    if memo not in _index_memo:
        a = encoding(alphabet)[DATA[name]["index"]]
        # Above k 14 the base-21 code wraps around 64 bits, which is then a hash: level() mixes it,
        # and among ~10^7 k-mers a collision is very unlikely.
        x, at = analyse.kmers(a, k)
        group = np.cumsum(a == analyse.SEPARATOR)[at]
        letters = np.bincount(a[a < 21], minlength=21)[:20].astype(float)
        log_f = np.zeros(256)
        log_f[:20] = -np.log2(np.maximum(letters / letters.sum(), 1e-12))
        running = np.r_[0.0, np.cumsum(log_f[a])]
        q0 = running[at + k] - running[at]  # each k-mer's chance from composition, in bits
        o = np.argsort(x, kind="stable")  # groups rise along the sequence, so (k-mer, group) order
        x, group = x[o], group[o]
        keep = np.r_[True, (x[1:] != x[:-1]) | (group[1:] != group[:-1])]
        x, group = x[keep], group[keep].astype(np.int32)
        first = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
        keys, counts = x[first], np.diff(np.r_[first, len(x)])
        bits = np.maximum(q0[o][keep][first] - np.log2(counts), 0)
        _index_memo.clear()  # one index at a time: wasm's memory is limited
        _index_memo[memo] = (keys, first, counts, group, bits)
    return _index_memo[memo]


def read_set(name, length, paired, max_pairs=None):
    """Each kept query's reads, and their reversed decoys. The same reads for every k and m, unless
    ``max_pairs`` caps them: then evenly across identity bins, and one pair per query if very few."""
    d = DATA[name]
    rng = np.random.default_rng(1)
    pair_q, reads, decoys = [], [], []
    need = 2 * length if paired else length
    kept = [q for q, s in enumerate(d["query_list"]) if len(s) >= need]
    each = PAIRS_PER_QUERY
    if max_pairs is not None and max_pairs < len(kept) * PAIRS_PER_QUERY:
        each = PAIRS_PER_QUERY if max_pairs >= 3 * PAIRS_PER_QUERY * len(BINS) else 1
        per_bin = max(int(np.ceil(max_pairs / (each * len(BINS)))), 1)
        order = rng.permutation(kept)
        kept = sorted(q for b in range(len(BINS)) for q in order[d["bin"][order] == b][:per_bin].tolist())
    for q in kept:
        s = d["query_list"][q]
        for _ in range(each):
            pair_q.append(q)
            if paired:
                gap = int(rng.integers(0, min(MAX_GAP, len(s) - need) + 1))
                start = int(rng.integers(0, len(s) - need - gap + 1))
                got = [s[start : start + length], s[start + length + gap : start + need + gap]]
            else:
                start = int(rng.integers(0, len(s) - need + 1))
                got = [s[start : start + length]]
            reads += got
            decoys += [m[::-1] for m in got]
    return np.array(pair_q, np.int32), reads, decoys, each


def hits_per_pair(reads, per_owner, table, idx, k):
    """Mean k-mer hits per read pair, from a sample of them."""
    keys, _, counts, _, _ = idx
    sample = reads[:: max(len(reads) // (60 * per_owner), 1) * per_owner][:60]
    encoded = table[np.frombuffer(b"*".join(sample), np.uint8)]
    x, _ = analyse.kmers(encoded, k)
    pos = np.searchsorted(keys, x)
    pos[pos == len(keys)] = 0
    return max(float(counts[pos][keys[pos] == x].sum(dtype=np.float64)) * per_owner / max(len(sample), 1), 1.0)


def expand(seqs, per_owner, table, idx, scale):
    """Rows: distinct (read pair, group) sharing a k-mer; hits: each shared k-mer's row and key."""
    keys, first, counts, group, _ = idx
    k = _rows_k[0]
    encoded = table[np.frombuffer(b"*".join(seqs), np.uint8)]
    x, at = analyse.kmers(encoded, k)
    owner = (np.cumsum(encoded == analyse.SEPARATOR)[at] // per_owner).astype(np.int64)
    o = np.lexsort((x, owner))
    x, owner = x[o], owner[o]
    d = np.r_[True, (x[1:] != x[:-1]) | (owner[1:] != owner[:-1])]
    x, owner = x[d], owner[d]
    pos = np.searchsorted(keys, x)
    pos[pos == len(keys)] = 0
    f = keys[pos] == x
    pos, owner = pos[f], owner[f]
    n = counts[pos]
    rows = np.repeat(first[pos], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    uniq, inverse = np.unique(np.repeat(owner, n) * scale + group[rows], return_inverse=True)
    return uniq // scale, uniq % scale, inverse.astype(np.int32), np.repeat(pos, n)


_rows_k = [0]


def rows_for(name, alphabet, k, length, paired):
    memo = (name, alphabet, k, length, paired)
    if memo not in _rows_memo:
        d = DATA[name]
        idx = build_index(name, alphabet, k)
        table = encoding(alphabet)
        per_owner = 2 if paired else 1
        pair_q, reads, decoys, each = read_set(name, length, paired)
        cap = int(HIT_LIMIT / hits_per_pair(reads, per_owner, table, idx, k))
        if cap < len(pair_q):
            pair_q, reads, decoys, each = read_set(name, length, paired, cap)
        scale = int(d["groups"]) + 2
        _rows_k[0] = k
        pair, group, inverse, key_of = expand(reads, per_owner, table, idx, scale)
        d_pair, _, d_inverse, d_key = expand(decoys, per_owner, table, idx, scale)
        q_of = pair_q[pair].astype(np.int64)
        true = group == d["best"][q_of]
        wanted = q_of * scale + group
        at = np.searchsorted(d["aligned_key"], wanted)
        at[at == len(d["aligned_key"])] = 0
        known = d["aligned_key"][at] == wanted
        band = np.where(known, np.digitize(d["aligned_x"][at], X_EDGES) - 1, len(X_EDGES) - 1)
        band_at = np.where(true, 0, band + 1)
        pair_bin = d["bin"][pair_q].astype(np.int32)
        cell = (pair_bin[pair] * N_BANDS + band_at).astype(np.int32)
        _rows_memo.clear()
        _rows_memo[memo] = {
            "pair": pair.astype(np.int32), "inverse": inverse, "key_of": key_of, "true": true, "cell": cell,
            "d_pair": d_pair, "d_inverse": d_inverse, "d_key": d_key, "pair_q": pair_q, "pair_bin": pair_bin,
            "keys": idx[0], "bits": idx[4], "each": each,
        }  # fmt: skip
    return _rows_memo[memo]


def tile_reads(name, length, paired):
    """Reads across each query with a relative, every half read length and to its end, so a target
    with no passing tile has no passing read. (read's query, reads)."""
    d = DATA[name]
    tile_q, reads = [], []
    gap = TILE_GAP if paired else 0
    span = 2 * length + gap if paired else length
    for q, s in enumerate(d["query_list"]):
        if d["best"][q] < 0:
            continue
        last = max(len(s) - span, 0)
        for start in sorted({*range(0, last + 1, max(length // 2, 1)), last}):
            tile_q.append(q)
            reads.append(s[start : start + length])
            if paired:
                reads.append(s[start + length + gap : start + span])
    return np.array(tile_q, np.int32), reads


def true_group_hits(name, alphabet, k, reads, read_q, paired):
    """(read, key) for each distinct k-mer a read pair shares with its query's true group."""
    d = DATA[name]
    keys, _, counts, group, _ = build_index(name, alphabet, k)
    scale = int(d["groups"]) + 2
    rows = np.repeat(np.arange(len(keys), dtype=np.int64), counts) * scale + group
    encoded = encoding(alphabet)[np.frombuffer(b"*".join(reads), np.uint8)]
    x, at = analyse.kmers(encoded, k)
    owner = (np.cumsum(encoded == analyse.SEPARATOR)[at] // (2 if paired else 1)).astype(np.int64)
    pos = np.searchsorted(keys, x)
    pos[pos == len(keys)] = 0
    f = keys[pos] == x
    owner, pos = owner[f], pos[f]
    want = pos.astype(np.int64) * scale + d["best"][read_q[owner]]
    j = np.searchsorted(rows, want)
    j[j == len(rows)] = 0
    ok = rows[j] == want
    pair = np.unique(owner[ok] * len(keys) + pos[ok])  # distinct (read, key)
    return (pair // len(keys)).astype(np.int32), (pair % len(keys)).astype(np.int32)


def target_called(name, alphabet, k, length, paired, at):
    """Targets called at each coverage, by floor and identity bin, summed over queries: for a target
    of length L, a Poisson number of reads with mean n = coverage * L / (mates * read length), each
    passing at the share r of its tiled reads, within one hash salt, so reads of a target share its
    conservation and hash."""
    d = DATA[name]
    tile_q, reads = tile_reads(name, length, paired)
    read, key = true_group_hits(name, alphabet, k, reads, tile_q, paired)
    keys, bits = _index_memo[(name, alphabet, k)][0], _index_memo[(name, alphabet, k)][4]
    n_q = len(d["query_list"])
    tiles = np.maximum(np.bincount(tile_q, minlength=n_q), 1)
    lengths = np.array([len(s) for s in d["query_list"]], float)
    mates = 2 if paired else 1
    bins = d["bin"].astype(np.int32)
    charge = float(np.log2(mates * TRANSLATED_FRAMES * max(length - k + 1, 1)))
    called = np.zeros((len(FLOORS), len(BINS), len(COVERAGES)))
    for salt in range(1, SALTS + 1):
        lv = level(keys ^ (salt * 0x51ED27))
        score, _, _ = scored(read, key, bits, lv, at, len(tile_q), charge)
        for f, floor in enumerate(FLOORS):
            on = score >= floor if floor else score > -np.inf
            r = np.bincount(tile_q[on], minlength=n_q) / tiles
            for c, coverage in enumerate(COVERAGES):
                n = coverage * lengths / (mates * length)
                called[f, :, c] += np.bincount(bins, weights=1 - np.exp(-n * r), minlength=len(BINS))
    return (called / SALTS).round(3).tolist()


def level(x):
    """The tables' sampling level: kept at 1 in 2^j when level >= j (splitmix64 of the k-mer)."""
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.minimum(np.log2((z & (~z + np.uint64(1))).astype(float) + (z == 0)).astype(np.int8), LEVELS - 1)


def scored(inverse, key_of, bits, lv, at, n, charge):
    keep = lv[key_of] >= at
    hits = np.bincount(inverse[keep], minlength=n)
    summed = np.bincount(inverse[keep], weights=bits[key_of[keep]], minlength=n)
    return np.where(hits > 0, summed - np.maximum(hits - 1, 0) * charge, -np.inf), hits, summed


def entry_for(name, alphabet, k, length, paired, at):
    """One k's table entry at one sampling level, over the hash salts."""
    d = DATA[name]
    r = rows_for(name, alphabet, k, length, paired)
    n_bins, n_floors = len(BINS), len(FLOORS)
    n_cells = n_bins * N_BANDS
    pair, inverse, key_of, true, cell = r["pair"], r["inverse"], r["key_of"], r["true"], r["cell"]
    n, d_n = len(pair), int(r["d_inverse"].max(initial=-1)) + 1
    charge = float(np.log2((2 if paired else 1) * TRANSLATED_FRAMES * (length - k + 1)))
    cube = np.zeros((n_floors, n_cells), np.int64)
    best = np.zeros((len(MARGINS), n_floors, n_cells), np.int64)
    query_pass = np.zeros((n_floors, n_bins, r["each"] + 1), np.int64)
    decoys = np.zeros(n_floors, np.int64)
    precision = np.zeros((SALTS, n_floors))
    hit_bits = np.zeros(2)
    queries = np.unique(r["pair_q"])
    query_of_pair = np.searchsorted(queries, r["pair_q"]).astype(np.int32)
    bin_of_query = d["bin"][queries].astype(np.int32)
    starts = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]]) if n else np.zeros(0, np.int32)
    for salt in range(1, SALTS + 1):
        lv = level(r["keys"] ^ (salt * 0x51ED27))
        score, hits, summed = scored(inverse, key_of, r["bits"], lv, at, n, charge)
        d_score, _, _ = scored(r["d_inverse"], r["d_key"], r["bits"], lv, at, d_n, charge)
        hit_bits += [summed[true].sum(), hits[true].sum()]
        top = np.repeat(np.maximum.reduceat(score, starts), np.diff(np.r_[starts, n])) if n else score
        for f, floor in enumerate(FLOORS):
            on = score >= floor if floor else score > -np.inf  # no floor keeps every hit
            cube[f] += np.bincount(cell[on], minlength=n_cells)
            for g, margin in enumerate(MARGINS):
                best[g, f] += np.bincount(cell[on & (score >= top - margin)], minlength=n_cells)
            precision[salt - 1, f] = (on & true).sum() / max(on.sum(), 1)
            true_pairs = np.bincount(query_of_pair[pair[on & true]], minlength=len(queries))
            np.add.at(query_pass[f], (bin_of_query, true_pairs), 1)
            decoys[f] += (d_score >= floor if floor else d_score > -np.inf).sum()
    bits = r["bits"]
    return {
        "postings": int(_index_memo[(name, alphabet, k)][3].size),
        "cells": [], "decoy": [],
        "key_bits": [float(bits.mean()), *map(float, np.quantile(bits, [0.05, 0.95]))],
        "charge": charge,
        "floor_cube": cube[None].tolist(),
        "best_cube": best[:, None].tolist(),
        "bits_per_true_hit": [round(float(hit_bits[0] / max(hit_bits[1], 1)), 2)],
        "floor_query_pass": query_pass[None].tolist(),
        "floor_decoys": decoys[None].tolist(),
        "floor_precision_sd": (precision.std(0) / np.maximum(precision.mean(0), 1e-9)).round(3)[None].tolist(),
        "floor_rows": np.bincount(cell, minlength=n_cells).tolist(),
        "target_called": target_called(name, alphabet, k, length, paired, at),
    }  # fmt: skip


def _group(a):
    return (a["index"], a["alphabet"], a["length"], a["paired"], int(round(np.log2(a["m"]))))


def have(request):
    a = json.loads(request)
    return str(a["k"]) in ENTRIES.get(_group(a), {}).get("k", {})


def add(request, text):
    """A stored result, back into memory."""
    a = json.loads(request)
    saved = json.loads(text)
    group = ENTRIES.setdefault(_group(a), {"k": {}})
    group["meta"] = saved["meta"]
    group["k"][str(a["k"])] = saved["entry"]
    group.pop("loaded", None)


def compute(request):
    """Compute one k's entry; returns the text to store."""
    a = json.loads(request)
    name, alphabet, k, length, paired, at = a["index"], a["alphabet"], a["k"], a["length"], a["paired"], _group(a)[4]
    entry = entry_for(name, alphabet, k, length, paired, at)
    r = _rows_memo[(name, alphabet, k, length, paired)]
    meta = {
        "pairs": len(r["pair_q"]), "pairs_per_query": r["each"], "mate": length, "groups": int(DATA[name]["groups"]),
        "identity_bins": BINS, "x_edges": X_EDGES.tolist(), "floors": list(FLOORS), "salts": SALTS, "n_bands": N_BANDS,
        "margins": list(MARGINS), "pairs_by_bin": np.bincount(r["pair_bin"], minlength=len(BINS)).tolist(),
        "queries_by_bin": (np.bincount(r["pair_bin"], minlength=len(BINS)) // r["each"]).tolist(),
        "coverages": list(COVERAGES), "mates": 2 if paired else 1,
        "mean_length": float(np.mean([len(s) for s in DATA[name]["query_list"]])),
        "targets_by_bin": np.bincount(DATA[name]["bin"], minlength=len(BINS)).tolist(),
    }  # fmt: skip
    text = json.dumps({"meta": meta, "entry": entry}, separators=(",", ":"))
    add(request, text)
    return text


def panel(request):
    """The precision panel from what is held, at this request's other controls."""
    global _loaded
    a = json.loads(request)
    key = _group(a)
    group = ENTRIES[key]
    if not group.get("loaded") or _loaded != key:
        model.load_table("live", json.dumps({**group["meta"], "k": group["k"]}))
        group["loaded"], _loaded = True, key
    return json.dumps(model.compute_precision({**a, "index": "live", "m": 1}))


# Combination k-mers on demand: recall on the true group, chance from a sample of the index's features.
COMBO_READ_WORK = 6e7  # feature sets x letters, reads and true groups together
COMBO_INDEX_WORK = 5e7  # feature sets x index letters
COMBO_INDEX_KEEP = 3e6  # index features held, by sampling on the hash
MIX = np.uint64(0x9E3779B97F4A7C15)
FEATURE = np.uint64(0x100000001B3)


def _at_level(h, at):
    """level(h) >= at, without the rest of level's work: the mixed hash's low bits are zero."""
    z = h.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(2**at - 1)) == 0


def _codes(seqs, table, k):
    """Each position's k-mer code (-1 if none) and sequence, over sequences joined by separators."""
    a = table[np.frombuffer(b"*".join(seqs), np.uint8)]
    x, at = analyse.kmers(a, k)
    code = np.full(len(a), -1, np.int64)
    code[at] = x
    return code, np.cumsum(a == analyse.SEPARATOR).astype(np.int32)


def _features(code, owner, offsets):
    """(owner, hash) of one set of offsets, gaps not hashed. A later k-mer must share the sequence."""
    last = offsets[-1] if offsets else 0
    i = np.arange(max(len(code) - last, 0))
    ok = code[i] >= 0
    h = code[i].astype(np.uint64)
    for o in offsets:
        ok &= (code[i + o] >= 0) & (owner[i + o] == owner[i])
        h = h * FEATURE + code[i + o].astype(np.uint64)
    return owner[i[ok]], h[ok]


def _distinct(owner, h):
    """Distinct (owner, hash), as (owner, hash) arrays."""
    key = h + owner.astype(np.uint64) * MIX
    _, first = np.unique(key, return_index=True)
    return owner[first], h[first]


def combo_simulate(request):
    """Recall per read pair by identity bin at each sampling level, and decoy matches, for one window.
    Very wide windows use fewer read pairs, and a part of the index for chance, which is scaled up."""
    a = json.loads(request)
    name, alphabet, k, n, lo, hi, length, paired = (
        a["index"], a["alphabet"], a["k"], a["combo_n"], a["combo_lo"], a["combo_hi"], a["length"], a["paired"],
    )  # fmt: skip
    d = DATA[name]
    if "group_list" not in d:
        d["group_list"] = bytes(d["index"]).split(b"*")
    sets = model.combo_sets(n, lo, hi)
    table = encoding(alphabet)
    mates = 2 if paired else 1
    need = mates * length * len(d["query_list"]) * PAIRS_PER_QUERY + sum(map(len, d["query_list"]))
    cap = int(len(d["query_list"]) * PAIRS_PER_QUERY * min(COMBO_READ_WORK / (len(sets) * need), 1))
    pair_q, reads, decoys, _ = read_set(name, length, paired, cap)
    pair_bin = d["bin"][pair_q].astype(np.int32)
    best = d["best"][pair_q]
    wanted = np.unique(best[best >= 0])
    group_code, group_owner = _codes([d["group_list"][g] for g in wanted], table, k)
    true_group = np.searchsorted(wanted, best).astype(np.int64)
    true_group[best < 0] = -1

    group_keys = []
    for offsets in sets:
        o, h = _features(group_code, group_owner, offsets)
        group_keys.append(h + o.astype(np.uint64) * MIX)
    group_keys = np.unique(np.concatenate(group_keys))
    read_code, read_owner = _codes(reads, table, k)
    hit_owner, hit_h = [], []
    for offsets in sets:
        o, h = _features(read_code, read_owner, offsets)
        pair = o // mates
        keep = true_group[pair] >= 0
        o, h, pair = o[keep], h[keep], pair[keep]
        key = h + true_group[pair].astype(np.uint64) * MIX
        at = np.searchsorted(group_keys, key)
        at[at == len(group_keys)] = 0
        on = group_keys[at] == key
        hit_owner.append(o[on])
        hit_h.append(h[on])
    owner, h = _distinct(np.concatenate(hit_owner), np.concatenate(hit_h))
    lv = level(h)
    pair = owner // mates
    counts = np.maximum(np.bincount(pair_bin, minlength=len(BINS)), 1)
    recall = []
    for j in range(LEVELS):
        passed = np.bincount(pair[lv >= j], minlength=len(pair_q)) > 0
        recall.append((np.bincount(pair_bin, weights=passed, minlength=len(BINS)) / counts)[:-1].round(4).tolist())

    # Chance: a share of the index's groups, and of their features by hash level.
    residues = int(d["index"].size)
    stride = int(np.ceil(len(sets) * residues / COMBO_INDEX_WORK))
    part = d["group_list"][:: max(stride, 1)]
    part_letters = sum(map(len, part))
    keep_level = max(int(np.ceil(np.log2(max(len(sets) * part_letters / COMBO_INDEX_KEEP, 1)))), 0)
    index_code, index_owner = _codes(part, table, k)
    held = []
    for offsets in sets:
        o, h = _features(index_code, index_owner, offsets)
        kept = _at_level(h, keep_level)
        held.append(_distinct(o[kept], h[kept])[1])
    held = np.sort(np.concatenate(held))
    decoy_code, decoy_owner = _codes(decoys, table, k)
    matches = 0
    for offsets in sets:
        o, h = _features(decoy_code, decoy_owner, offsets)
        kept = _at_level(h, keep_level)
        o, h = _distinct(o[kept] // mates, h[kept])
        matches += int((np.searchsorted(held, h, side="right") - np.searchsorted(held, h, side="left")).sum())
    scale = 2.0**keep_level * residues / max(part_letters, 1)
    return json.dumps({
        "recall": recall, "bins": BINS, "pairs": len(pair_q), "sets": len(sets), "matches": matches,
        "index_share": part_letters / residues, "keep_level": keep_level, "chance_m1": matches * scale / len(pair_q),
        "composition_m1": model.combo_chance(length, k, n, lo, hi, paired, 1, residues, model.alphabet_stats(alphabet)[1]),
    })  # fmt: skip


# Precision for combination k-mers: hits on every group, for a range of sampling levels from one pass.
PRECISION_LEVELS = 6  # levels computed upward from the one asked for
PRECISION_KEEP = 8e6  # index features held, over the salts
PRECISION_WORK = 5e7  # feature sets x index letters, per salt
PRECISION_HITS = 1.5e7  # (read pair, group) hits
SALT = 0x51ED27


def _features_q(code, owner, running, k, offsets):
    """As _features, with each feature's composition bits over the letters it fixes."""
    last = offsets[-1] if offsets else 0
    i = np.arange(max(len(code) - last - k + 1, 0))
    ok = code[i] >= 0
    h = code[i].astype(np.uint64)
    q = running[i + k] - running[i] if len(i) else np.zeros(0)
    prev = 0
    for o in offsets:
        ok &= (code[i + o] >= 0) & (owner[i + o] == owner[i])
        h = h * FEATURE + code[i + o].astype(np.uint64)
        q = q + running[i + o + k] - running[i + max(prev + k, o)]
        prev = o
    return owner[i[ok]], h[ok], q[ok]


def _coded(seqs, table, k):
    a = table[np.frombuffer(b"*".join(seqs), np.uint8)]
    x, at = analyse.kmers(a, k)
    code = np.full(len(a), -1, np.int64)
    code[at] = x
    return a, code, np.cumsum(a == analyse.SEPARATOR).astype(np.int32)


def combo_precision(request):
    """Precision, recall and chance per read pair, by floor, for levels from the one asked for. Features
    are scored as kfp's keys are: composition less log2 of the groups holding them, with the charge per
    further hit counting every feature a read tries. Where the index is too large, a share of its groups
    is used, true groups always among them, and hits on the others are scaled up."""
    a = json.loads(request)
    name, alphabet, k, n, lo, hi, length, paired, at = (
        a["index"], a["alphabet"], a["k"], a["combo_n"], a["combo_lo"], a["combo_hi"], a["length"], a["paired"],
        int(a["level"]),
    )  # fmt: skip
    d = DATA[name]
    if "group_list" not in d:
        d["group_list"] = bytes(d["index"]).split(b"*")
    sets = model.combo_sets(n, lo, hi)
    g_sets = len(sets)
    salts = [SALT * s for s in range(1, 5 if g_sets <= 20 else 3)]
    table = encoding(alphabet)
    mates = 2 if paired else 1
    pair_q, reads, decoys, each = read_set(name, length, paired)
    residues = int(d["index"].size)
    share = min(1.0, PRECISION_WORK / (g_sets * residues), PRECISION_KEEP * 2.0**at / (len(salts) * g_sets * residues))
    stride = max(int(np.ceil(1 / share)), 1)
    wanted = np.unique(d["best"][d["best"] >= 0])
    ids = np.union1d(np.arange(0, len(d["group_list"]), stride), wanted)
    share = float(len(np.arange(0, len(d["group_list"]), stride))) / len(d["group_list"])
    a_index, code, owner = _coded([d["group_list"][i] for i in ids], table, k)
    letters = np.bincount(a_index[a_index < 21], minlength=21)[:20].astype(float)
    log_f = np.zeros(256)
    log_f[:20] = -np.log2(np.maximum(letters / letters.sum(), 1e-12))
    held_h, held_g = [], []
    for offsets in sets:
        o, h = _features(code, owner, offsets)
        kept = np.zeros(len(h), bool)
        for salt in salts:
            kept |= _at_level(h ^ np.uint64(salt), at)
        o, h = _distinct(o[kept], h[kept])
        held_h.append(h)
        held_g.append(o)
    held_h, held_g = np.concatenate(held_h), np.concatenate(held_g)
    held_g, held_h = _distinct(held_g, held_h)
    o = np.argsort(held_h, kind="stable")
    held_h, held_g = held_h[o], held_g[o]
    first = np.flatnonzero(np.r_[True, held_h[1:] != held_h[:-1]]) if len(held_h) else np.zeros(0, int)
    keys = held_h[first]
    holders = np.diff(np.r_[first, len(held_h)]).astype(np.intp)  # wasm32 repeats by int32
    q_of_key = np.zeros(len(keys))

    def lookup(seqs):
        """Distinct (read pair, key position) a read's features share with the held index."""
        a_reads, c, w = _coded(seqs, table, k)
        running = np.r_[0.0, np.cumsum(log_f[a_reads])]
        pairs, positions = [], []
        for offsets in sets:
            mate, h, q = _features_q(c, w, running, k, offsets)
            p = np.searchsorted(keys, h)
            p[p == len(keys)] = 0
            found = keys[p] == h if len(keys) else np.zeros(len(h), bool)
            q_of_key[p[found]] = q[found]
            pairs.append(mate[found] // mates)
            positions.append(p[found])
        pairs, positions = np.concatenate(pairs), np.concatenate(positions)
        unique_rows = np.unique(pairs.astype(np.int64) * (len(keys) + 1) + positions)
        return (unique_rows // (len(keys) + 1)).astype(np.int32), unique_rows % (len(keys) + 1)

    r_pair, r_key = lookup(reads)
    total_hits = float(holders[r_key].sum()) if len(r_key) else 0.0
    n_pairs = len(pair_q)
    kept_pairs = n_pairs
    if total_hits > PRECISION_HITS:  # every few read pairs, across all queries
        step = int(np.ceil(total_hits / PRECISION_HITS))
        keep = r_pair % step == 0
        r_pair, r_key = r_pair[keep], r_key[keep]
        kept_pairs = len(np.arange(0, n_pairs, step))
    d_pair, d_key = lookup(decoys)
    d_keep = d_pair % max(n_pairs // max(kept_pairs, 1), 1) == 0
    d_pair, d_key = d_pair[d_keep], d_key[d_keep]

    def expand_rows(pairs, positions):
        positions = positions.astype(np.intp)
        counts = holders[positions]
        total = int(counts.sum(dtype=np.int64))
        rows = np.repeat(first[positions], counts) + (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
        groups = ids[held_g[rows]].astype(np.int64)
        scale = int(d["groups"]) + 2
        uniq, inverse = np.unique(np.repeat(pairs, counts).astype(np.int64) * scale + groups, return_inverse=True)
        return uniq // scale, uniq % scale, inverse.astype(np.int32), np.repeat(positions, counts)

    pair, group, inverse, key_of = expand_rows(r_pair, r_key)
    _, _, d_inverse, d_key_of = expand_rows(d_pair, d_key)
    q_of = pair_q[pair].astype(np.int64)
    true = group == d["best"][q_of]
    scale = int(d["groups"]) + 2
    wanted_key = q_of * scale + group
    at_key = np.searchsorted(d["aligned_key"], wanted_key)
    at_key[at_key == len(d["aligned_key"])] = 0
    known = d["aligned_key"][at_key] == wanted_key
    band = np.where(known, np.digitize(d["aligned_x"][at_key], X_EDGES) - 1, len(X_EDGES) - 1)
    pair_bin = d["bin"][pair_q].astype(np.int32)
    cell = (pair_bin[pair] * N_BANDS + np.where(true, 0, band + 1)).astype(np.int32)
    in_group_all = ~true
    bits = np.maximum(q_of_key - np.log2(np.maximum(holders / share, 1)), 0)
    charge = float(np.log2(mates * TRANSLATED_FRAMES * max(length - k + 1, 1) * g_sets))
    n_cells = len(BINS) * N_BANDS
    row_n, d_n = len(pair), int(d_inverse.max(initial=-1)) + 1
    starts = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]]) if row_n else np.zeros(0, np.int32)
    weight = np.where(in_group_all, 1 / share, 1.0)  # hits on the other groups stand for the rest
    out = {}
    for level_at in range(at, min(at + PRECISION_LEVELS, LEVELS)):
        cube = np.zeros((len(FLOORS), n_cells))
        best = np.zeros((len(MARGINS), len(FLOORS), n_cells))
        decoy_count = np.zeros(len(FLOORS))
        for salt in salts:
            lv = level(keys ^ np.uint64(salt))
            score, _, _ = scored(inverse, key_of, bits, lv, level_at, row_n, charge)
            d_score, _, _ = scored(d_inverse, d_key_of, bits, lv, level_at, d_n, charge)
            top = np.repeat(np.maximum.reduceat(score, starts), np.diff(np.r_[starts, row_n])) if row_n else score
            for f, floor in enumerate(FLOORS):
                on = score >= floor if floor else score > -np.inf
                cube[f] += np.bincount(cell[on], weights=weight[on], minlength=n_cells)
                for g, margin in enumerate(MARGINS):
                    kept = on & (score >= top - margin)
                    best[g, f] += np.bincount(cell[kept], weights=weight[kept], minlength=n_cells)
                decoy_count[f] += (d_score >= floor if floor else d_score > -np.inf).sum() / share
        out[level_at] = {"cube": (cube / len(salts)).round(2).tolist(), "best": (best / len(salts)).round(2).tolist(),
                         "decoys": (decoy_count / len(salts)).round(2).tolist()}  # fmt: skip
    return json.dumps({
        "levels": out, "from_level": at, "pairs": kept_pairs,
        "pairs_by_bin": (np.bincount(pair_bin, minlength=len(BINS)) * kept_pairs // max(n_pairs, 1)).tolist(),
        "floors": list(FLOORS), "margins": list(MARGINS), "x_edges": X_EDGES.tolist(), "n_bands": N_BANDS,
        "index_share": share, "sets": g_sets, "salts": len(salts), "each": each,
    })  # fmt: skip


def combo_view(text, level_at, a_json):
    """By floor, at one level: recall, precision and others per read pair, at the precision panel's
    rule, truth and identity mix."""
    t = json.loads(text)
    a = json.loads(a_json)
    entry = t["levels"][str(level_at)]
    bins, bands = len(t["pairs_by_bin"]), t["n_bands"]
    rule = t["margins"].index(a["margin"]) if a.get("margin") in t["margins"] else None
    cube = np.array(entry["cube"] if rule is None else entry["best"][rule]).reshape(len(t["floors"]), bins, bands)
    mix = np.array(model.NATURAL_MIX if a["mix"] == "natural" else [1.0] * (bins - 1), float)
    weights = np.r_[mix / mix.sum() * (1 - a["no_relative"]), a["no_relative"]]
    per_pair = (cube * (weights / np.maximum(np.array(t["pairs_by_bin"], float), 1))[None, :, None]).sum(1)
    ok = np.r_[False, X_EDGES[:-1] >= a["truth"], False]
    true_hits = per_pair[:, 0]
    good = per_pair[:, ok].sum(1)
    wrong = per_pair[:, 1:][:, ~ok[1:]].sum(1) + np.array(entry["decoys"]) / max(t["pairs"], 1) * model.WRONG_FRAMES
    correct = true_hits + good
    return json.dumps({
        "floors": t["floors"], "recall": (true_hits / max(1 - a["no_relative"], 1e-12)).tolist(),
        "precision": (correct / np.maximum(correct + wrong, 1e-12)).tolist(), "others": (good + wrong).tolist(),
        "pairs": t["pairs"], "index_share": t["index_share"], "salts": t["salts"], "from_level": t["from_level"],
    })  # fmt: skip
