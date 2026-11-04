"""The browser analysis against the native pilot (docs/plan.md, Pilot results) and random sequences.

Run: uv run --no-project --with numpy python checks/test_analyse.py"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "web"))
import analyse  # noqa: E402

PILOT = {  # k: distinct / total, decoy / composition, between units / composition
    5: (0.168, 1.23, 3.23),
    6: (0.53, 1.41, 24.92),
    7: (0.68, 1.93, 306.27),
    8: (0.727, 4.71, 4194.89),
    9: (0.753, 33.56, 60105.74),
}


def rejects(raw, text):
    try:
        analyse.load(raw)
    except ValueError as e:
        assert text in str(e), e
    else:
        raise AssertionError(f"accepted: {text}")


def main():
    rejects(b"", "Not FASTA")
    rejects(b"@read\nACGT\n+\nIIII\n", "Not FASTA")
    rejects(b">empty\n>also\n", "No sequences")
    rejects(b">dna\nACGTACGTTTGACCAGT\n", "DNA")
    rejects(gzip.compress(b"\0" * 10), "Not FASTA")

    plain = b">a desc\nmkv\nLLA\r\n>b\nWWX*\n"
    assert analyse.parse_fasta(plain) == [b"MKVLLA", b"WWX"]
    assert analyse.parse_fasta(gzip.compress(plain)) == [b"MKVLLA", b"WWX"]
    s = json.loads(analyse.load(plain))
    assert (s["sequences"], s["residues"], s["other_residues"]) == (2, 8, 1)
    x, _ = analyse.kmers(analyse.state["a"], 2)
    assert len(x) == 5 + 1  # k-mers neither cross sequences nor include X
    assert (analyse.reversed_kmers(x, 2) == analyse.kmers(analyse.state["a"][::-1], 2)[0][::-1]).all()

    assert analyse.BACKGROUND[analyse.AMINO.index(b"L")] == 0.0966  # the commonest residue
    assert analyse.BACKGROUND[analyse.AMINO.index(b"W")] == 0.0108  # the rarest

    # Independent residues: the model's values must hold.
    rng = np.random.default_rng(3)
    f = np.array(analyse.BACKGROUND) / sum(analyse.BACKGROUND)
    amino = np.frombuffer(analyse.AMINO, np.uint8)
    seqs = [rng.choice(amino, 200, p=f).tobytes() for _ in range(5000)]
    analyse.load(b"".join(b">s\n" + q + b"\n" for q in seqs))
    for k in (3, 5, 7):
        r = json.loads(analyse.spectrum(k))
        print(f"random k{k}: distinct {r['distinct_share']:.4f}, independent "
              f"{r['distinct_independent']:.4f}, decoy ratio {r['decoy_ratio']:.3f}")
        assert abs(r["distinct_share"] - r["distinct_independent"]) < 0.01
        assert abs(r["decoy_ratio"] - 1) < 0.15 and abs(r["between_ratio"] - 1) < 0.15
    for row in json.loads(analyse.rarefy(5))["rows"]:
        print(f"random {row['sequences']}: hit {row['decoy_hit']:.4f}, "
              f"independent {row['decoy_hit_independent']:.4f}")
        assert abs(row["decoy_hit"] - row["decoy_hit_independent"]) < 0.01

    summary = json.loads(analyse.load((ROOT / "web/data/mini-slice-44k.fa.gz").read_bytes()))
    assert summary["sequences"] == 44270 and abs(summary["s2"] - 0.0600) < 5e-5
    print("quantiles", summary["length_quantiles"], "dispersion", round(summary["length_dispersion"], 2))
    assert summary["length_quantiles"][0] == 65 and summary["length_quantiles"][2] == 193
    assert abs(summary["length_dispersion"] - 2.62) < 0.01
    for k, (distinct, decoy, between) in PILOT.items():
        r = json.loads(analyse.spectrum(k))
        print(f"k{k}: distinct {r['distinct_share']:.3f} (pilot {distinct}), decoy {r['decoy_ratio']:.2f} "
              f"({decoy}), between {r['between_ratio']:.4g} ({between})")
        assert abs(r["distinct_share"] - distinct) < 0.005
        assert 0.5 < r["between_ratio"] / between < 2  # the split is random
    rows = json.loads(analyse.rarefy(7))["rows"]
    assert all(a["distinct"] < b["distinct"] for a, b in zip(rows, rows[1:]))
    print("ok")


if __name__ == "__main__":
    main()
