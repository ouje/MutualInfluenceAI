# run_grid_parallel.py
# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2025 Vaclav Oujezsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import csv
import time
import random
import asyncio
import itertools
from typing import Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

# one_pass must return the dict you've been using (short metrics + raw JSON if present)
from run_grid import one_pass as one_pass_existing  # fallback if you kept one_pass here

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

# ===========================
# CONFIG
# ===========================
# Narrower defaults (fast) - this can be adjusted per needs:
GRID = {
    "beta":  [0.2, 0.4, 0.6, 0.8],
    "k":     [3.0, 6.0],
    "tau":   [0.3, 0.5, 0.7],
    "alpha": [0.4, 0.8, 1.2],
}
SEEDS = [1, 2, 3, 4, 5]
ADVERSARIAL = [False, True]

# Concurrency + time budget (seconds). Set TIME_BUDGET_S=None to run all.
MAX_CONCURRENCY = 4
TIME_BUDGET_S = 20 * 60  # e.g., 20 minutes. Set to None for full sweep.

# Output files
OUT_CSV = "results.csv"            # short metrics only
RESUME_ENABLED = True              # skip rows already in OUT_CSV
SHUFFLE_TASKS = True               # randomize order so early stop covers whole space
LOG_EVERY = 1                      # print every N completed tasks


# ===========================
# FIELDS (short CSV schema)
# ===========================
FIELDNAMES = [
    "beta", "k", "tau", "alpha", "seed", "adversarial",
    "mu_planner", "mu_researcher", "mu_critic",
    "RoundsToApproval_baseline", "RoundsToApproval_influence",
    "AgreementRate_baseline", "AgreementRate_influence",
    "RevisionDepth_between_rounds",
    "PlannerResearcher_Canonical_baseline", "PlannerResearcher_Canonical_influence",
    "Planner_SelfAgreement", "Researcher_SelfAgreement"
]


# ===========================
# TASK BUILDING / RESUME
# ===========================
def build_tasks() -> List[Tuple[float, float, float, float, int, bool]]:
    grid_prod = list(itertools.product(
        GRID["beta"], GRID["k"], GRID["tau"], GRID["alpha"], SEEDS, ADVERSARIAL
    ))
    tasks = [(b, k, t, a, s, adv) for (b, k, t, a, s, adv) in grid_prod]
    if SHUFFLE_TASKS:
        random.shuffle(tasks)
    return tasks


def read_done_keys(path: str) -> set:
    """
    Return a set of 6-tuples (beta, k, tau, alpha, seed, adversarial) already in CSV.
    """
    done = set()
    if not RESUME_ENABLED:
        return done
    if not os.path.exists(path):
        return done
    try:
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    key = (
                        float(row["beta"]),
                        float(row["k"]),
                        float(row["tau"]),
                        float(row["alpha"]),
                        int(row["seed"]),
                        bool(int(row["adversarial"])),
                    )
                    done.add(key)
                except Exception:
                    continue
    except Exception:
        pass
    return done


def ensure_header(path: str, fieldnames: List[str]) -> None:
    """
    Create file with header if it doesn't exist. If exists and empty, write header.
    """
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


# ===========================
# ASYNC RUNNER
# ===========================
async def run_one(
    sem: asyncio.Semaphore,
    writer_lock: asyncio.Lock,
    csv_path: str,
    params: Tuple[float, float, float, float, int, bool],
    idx: int,
    total: int,
) -> None:
    b, k, t, a, s, adv = params
    key = (b, k, t, a, s, adv)

    async with sem:
        try:
            # Call existing one_pass
            out: Dict = await one_pass_existing(beta=b, k=k, tau=t, alpha=a, seed=s, adversarial=adv)

            # Filter to short schema
            short = {fn: out.get(fn, None) for fn in FIELDNAMES}

            # Append to CSV
            async with writer_lock:
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    w.writerow(short)

            if (idx + 1) % LOG_EVERY == 0:
                print(f"[{idx+1}/{total}] saved {key} -> "
                      f"agr_inf={short['AgreementRate_influence']} "
                      f"r2={short['RoundsToApproval_influence']}")
        except Exception as e:
            print(f"[{idx+1}/{total}] ERROR for {key}: {e}")


async def main():
    random.seed(1234)

    # Build all tasks and optionally resume
    tasks = build_tasks()
    done = read_done_keys(OUT_CSV)

    # Filter out already-done keys
    todo = [p for p in tasks if (p[0], p[1], p[2], p[3], p[4], p[5]) not in done]
    total = len(todo)
    print(f"Planned total combos: {len(tasks)} | already in CSV: {len(done)} | to run now: {total}")

    ensure_header(OUT_CSV, FIELDNAMES)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    writer_lock = asyncio.Lock()

    start = time.time()
    completed = 0

    for idx, params in enumerate(todo):
        # Time budget check
        if TIME_BUDGET_S is not None and (time.time() - start) > TIME_BUDGET_S:
            print(f" Stopping due to TIME_BUDGET_S={TIME_BUDGET_S}s. Completed {completed} tasks.")
            break

        await run_one(sem, writer_lock, OUT_CSV, params, idx, total)
        completed += 1

    print(f"\nDone. Wrote/updated: {OUT_CSV}. Completed {completed} tasks in "
          f"{int(time.time()-start)}s.")


if __name__ == "__main__":
    asyncio.run(main())
