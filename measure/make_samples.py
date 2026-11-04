"""The bundled samples: mini-slice representatives as gzipped FASTA, for the page's analysis.

Run: uv run --no-project --with polars python measure/make_samples.py <kfp>/data/mini-slice/members.parquet"""

import gzip
import sys
from pathlib import Path

import polars as pl

OUT = Path(__file__).parent.parent / "web" / "data"


def write(reps, path):
    text = "".join(f">{i}\n{s}\n" for i, s in zip(reps["protein_id"], reps["sequence"], strict=True))
    # mtime 0: the same input gives the same bytes
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(text.encode())
    print(path.name, reps.height, "sequences,", path.stat().st_size / 1e6, "MB")


def main():
    members = pl.read_parquet(sys.argv[1], columns=["protein_id", "cluster_rep", "sequence"])
    reps = members.filter(pl.col("protein_id") == pl.col("cluster_rep")).sort("protein_id")
    write(reps, OUT / "mini-slice-44k.fa.gz")
    write(reps.sample(10_000, seed=1), OUT / "mini-slice-10k.fa.gz")


if __name__ == "__main__":
    main()
