"""All first_fix runs in one pool; each result goes to the folder compare_center_open --sweep would use, so that the
script's own --sweep call then finds them done and only writes the table."""
import json
from multiprocessing import Pool
from pathlib import Path
from rl_search.scripts.compare_center_open import run_job
SETTINGS = (("no_overlap", True, True), ("sweep", False, False))
VARIANTS = (("intercept", True, False), ("adjacent", False, True), ("intercept_adjacent", True, True))
def folder(j):
    tag = "_".join(t for t, on in (("intercept", j[7]), ("adjacent", j[8])) if on)
    return Path(f"rl_search/data/center_open/first_fix/{tag}/{'no_overlap' if j[6] else 'sep' if j[5] else 'sweep'}")
if __name__ == "__main__":
    jobs = [(s, "score", 5, 70.0, 1.0, sep, no, ic, adj) for _, sep, no in SETTINGS for _, ic, adj in VARIANTS for s in range(500, 510)]
    done = set()
    for j in jobs:
        p = folder(j)/"results.jsonl"
        if p.exists():
            done |= {(r["seed"], folder(j)) for r in map(json.loads, p.read_text().splitlines())}
    jobs = [j for j in jobs if (j[0], folder(j)) not in done]
    print(len(jobs), "jobs", flush=True)
    with Pool(9) as pool:
        for j, r in zip(jobs, pool.imap(run_job, jobs, chunksize=1)):
            r.pop("actions", None)
            f = folder(j); f.mkdir(parents=True, exist_ok=True)
            with open(f/"results.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
    print("done", flush=True)
