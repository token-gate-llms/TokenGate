"""Merge fan-out runs of the accuracy harness into one benchmark tree.

    python -m kernels.bench.bench_merge --root <iso_bench_root> --from <root_tagA> <root_tagB> ...

Every lane CSV under <root_tag>/repobench/<lane>/grouped_eval.csv is appended
to <root>/repobench/<lane>/grouped_eval.csv, keyed by (baseline, K,
sample_idx, task): a row already present wins, so a lane re-scored by a later
fan-out job does not overwrite the run of record.  The sample indices are the
draw's own (bench_grouped_eval --sample-slice keeps them), so slices and K
subsets of one run line up.
"""

import argparse
import csv
import glob
import os


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    a = p.parse_args()
    for src in a.sources:
        for path in sorted(glob.glob(os.path.join(src, "repobench", "*", "grouped_eval.csv"))):
            lane = os.path.basename(os.path.dirname(path))
            dst = os.path.join(a.root, "repobench", lane, "grouped_eval.csv")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            have, rows, header = set(), [], None
            if os.path.exists(dst):
                with open(dst, newline="") as f:
                    r = csv.DictReader(f)
                    header = r.fieldnames
                    for row in r:
                        have.add((row["baseline"], row["K"], row["sample_idx"], row["task"]))
                        rows.append(row)
            added = 0
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                if header is None:
                    header = r.fieldnames
                elif r.fieldnames != header:
                    raise ValueError(f"{path}: columns {r.fieldnames} differ from {dst}'s {header}")
                for row in r:
                    key = (row["baseline"], row["K"], row["sample_idx"], row["task"])
                    if key in have:
                        continue
                    have.add(key)
                    rows.append(row)
                    added += 1
            with open(dst, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(rows)
            print(f"[merge] {lane}: +{added} rows from {path} -> {len(rows)} in {dst}", flush=True)


if __name__ == "__main__":
    main()
