"""Convert generated NIAH eval rows to the frozen 150-row benchmark source."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("split_json", type=Path)
    parser.add_argument("out_jsonl", type=Path)
    args = parser.parse_args()
    split = json.loads(args.split_json.read_text())["niah_kamradt"]
    ids = set(split["test"])
    assert len(split["val"]) == 50 and len(ids) == 150
    rows = []
    for line in args.raw_jsonl.read_text().splitlines():
        item = json.loads(line)
        if item["id"] in ids:
            assert item["split"] == "test"
            rows.append({"id": item["id"],
                         "article": item["body"] + "\n\nQuestion: " + item["question"],
                         "target": item["answer"]})
    assert {row["id"] for row in rows} == ids and len(rows) == 150
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} frozen held-out rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
