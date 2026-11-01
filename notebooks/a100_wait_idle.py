"""Wait for genuinely idle A100s before a shared-cluster validation run."""

import argparse
import subprocess
import sys
import time


def output(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True).strip()


def available(min_free_mib: int, max_util: int) -> list[int]:
    busy = set(output("--query-compute-apps=gpu_uuid", "--format=csv,noheader").splitlines())
    rows = output(
        "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    free = []
    for line in rows.splitlines():
        index, uuid, name, memory, utilization = (x.strip() for x in line.split(","))
        if ("A100" in name and uuid not in busy and int(memory) >= min_free_mib
                and int(utilization) <= max_util):
            free.append(int(index))
    return free


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--min-free-mib", type=int, default=78000)
    parser.add_argument("--max-util", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_seconds
    previous = None
    while time.monotonic() < deadline:
        candidates = available(args.min_free_mib, args.max_util)
        selected = tuple(candidates[: args.count])
        if len(selected) == args.count and selected == previous:
            print(",".join(map(str, selected)))
            return
        previous = selected if len(selected) == args.count else None
        print(f"waiting for {args.count} idle A100s; candidates={candidates}",
              file=sys.stderr, flush=True)
        time.sleep(args.poll_seconds)
    raise SystemExit("timed out waiting for idle A100s")


if __name__ == "__main__":
    main()
