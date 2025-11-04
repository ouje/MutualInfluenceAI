# run_grid.py
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
import asyncio
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

from mutual_influence_agents import MutualInfluenceAssistant
from metrics import (
    extract_features_from_json,
    critic_decision_from_json,
    jaccard,
    canonical_overlap,
    safe_json_loads
)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

DEBUG_LOG = True  # False for production

def _short(s: str, n: int = 800) -> str:
    if not isinstance(s, str):
        return str(s)
    return s if len(s) <= n else s[:n] + " ...[truncated]"

def debug_dump(label: str, text: str):
    if not DEBUG_LOG:
        return
    print(f"\n--- RAW {label} ---")
    print(_short(text))
    d = safe_json_loads(text)
    if d is None:
        print(f"[parse] ❌ not valid JSON")
    else:
        try:
            print(f"[parse] valid JSON with keys: {list(d.keys())[:8]}")
        except Exception:
            print(f"[parse] valid JSON (non-dict)")


# ---- Prompts ---------------------------------------------------------------

_ALLOWED = '["flow_bytes","packets","rate","iat","src_ip","dst_ip","src_port","dst_port","protocol","entropy","payload_len"]'

def prompts(seed: int):
    base_hint = (
      "Return exactly ONE JSON object. No prose, no explanations, no markdown, no code fences.\n"
      "If you cannot comply, output {}.\n"
    )
    return {
      "planner_baseline": f"""{base_hint}
Return:
{{
  "features": ["<name1>", "<name2>", "<name3>"],
  "steps": ["<step1>", "<step2>", "<step3>"]
}}
Task: [seed={seed}] Propose exactly 3 streaming features for malware triage and a 3-step plan that uses exactly those features.
Choose features ONLY from this allowed set (use exact tokens): {_ALLOWED}.
""",

      "researcher_baseline": f"""{base_hint}
Return:
{{ "features": ["<name1>", "<name2>", "<name3>"] }}
Task: [seed={seed}] List exactly 3 streaming features (names only) computable in real time for malware triage.
Choose ONLY from this allowed set (use exact tokens): {_ALLOWED}.
""",
      
      "critic_baseline": f"""{base_hint}
Return exactly: {{ "decision": "APPROVE" or "REVISE" }}

You will receive two JSON objects below as PLANNER and RESEARCHER.

APPROVAL RUBRIC (deterministic):
- APPROVE only if ALL are true:
  1) Both JSONs are valid, PLANNER has "features"(3) and "steps"(≥2), RESEARCHER has "features"(3).
  2) All features in both JSONs belong to {_ALLOWED}.
  3) Planner and Researcher share at least 2 out of 3 features (Jaccard ≥ 0.66).
  4) At least 2 of Planner's steps explicitly mention (by exact token) features used by Planner.
- Otherwise REVISE.
""",

      "planner_influenced": f"""{base_hint}
Return:
{{
  "features": ["<name1>", "<name2>", "<name3>"],
  "steps": ["<step1>", "<step2>"]
}}
Refine the plan to reduce false positives and keep exactly 3 features consistent with the plan.
Each step MUST explicitly mention by name one or more of the chosen features.
Choose features ONLY from this allowed set (use exact tokens): {_ALLOWED}.
""",

      "researcher_influenced": f"""{base_hint}
Return:
{{ "features": ["<name1>", "<name2>", "<name3>"] }}
List exactly 3 minimal streaming features we can compute now.
Choose ONLY from this allowed set (use exact tokens): {_ALLOWED}.
""",

      "critic_influenced": f"""{base_hint}
Return exactly: {{ "decision": "APPROVE" or "REVISE" }}

You will receive two JSON objects below as PLANNER and RESEARCHER.

APPROVAL RUBRIC (deterministic):
- APPROVE only if ALL are true:
  1) Both JSONs are valid, PLANNER has "features"(3) and "steps"(≥2), RESEARCHER has "features"(3).
  2) All features in both JSONs belong to {_ALLOWED}.
  3) Planner and Researcher share at least 2 out of 3 features (Jaccard ≥ 0.66).
  4) At least 2 of Planner's steps explicitly mention (by exact token) features used by Planner.
- Otherwise REVISE.
"""
    }


# ---- One pass --------------------------------------------------------------

async def one_pass(beta=0.6, k=6.0, tau=0.5, alpha=0.8, seed=1, adversarial=False) -> Dict:
    planner = MutualInfluenceAssistant("planner", "Role: Planner. Short prioritized plan.",
                                       API_KEY, k, tau, alpha)
    researcher = MutualInfluenceAssistant("researcher", "Role: Researcher. List concise streaming features and brief reasons.",
                                          API_KEY, k, tau, alpha)
    critic = MutualInfluenceAssistant("critic", "Role: Critic. Point gaps/risks. Reply 'APPROVE' if sufficient.",
                                      API_KEY, k, tau, alpha)

    P = prompts(seed)

    # --- Round 1 (baseline) ---
    p1 = await planner.call(P["planner_baseline"], influenced=False)
    r1 = await researcher.call(P["researcher_baseline"], influenced=False)

    # We call the critique with embedded JSONs (so that it can make a deterministic decision).
    c1_prompt = P["critic_baseline"] + f"\nPLANNER:\n{p1}\n\nRESEARCHER:\n{r1}\n"
    c1 = await critic.call(c1_prompt, influenced=False, base_temp=0.2)

    # DEBUG dump
    debug_dump("planner_baseline", p1)
    debug_dump("researcher_baseline", r1)
    debug_dump("critic_baseline", c1)

    # --- Simulated peer feedback (EMA) ---
    if adversarial:
        planner.receive_feedback("critic", 0.1, beta)
        researcher.receive_feedback("critic", 0.1, beta)
    else:
        planner.receive_feedback("critic", 0.9, beta)
        researcher.receive_feedback("critic", 0.7, beta)

    planner.receive_feedback("researcher", 0.8, beta)
    researcher.receive_feedback("planner", 0.85, beta)
    critic.receive_feedback("planner", 0.8 if not adversarial else 0.4, beta)
    critic.receive_feedback("researcher", 0.75 if not adversarial else 0.4, beta)

    # --- Round 2 (influenced) ---
    # We take the trio of RESEARCHERS from the baseline and give it to the planner as a constraint.
    RF1 = extract_features_from_json(r1)
    researcher_feats_str = ", ".join(sorted(RF1)) if RF1 else ""
    planner_infl_prompt = P["planner_influenced"]
    if researcher_feats_str:
        planner_infl_prompt += (
            f"\n\nConstraint: Use EXACTLY these three features from Researcher baseline "
            f"(use exact tokens, same order not required): [{researcher_feats_str}]. Do not rename them."
        )

    p2 = await planner.call(planner_infl_prompt, influenced=True)
    r2 = await researcher.call(P["researcher_influenced"], influenced=True)

    # Criticism in influenced again with embedded JSONs
    c2_prompt = P["critic_influenced"] + f"\nPLANNER:\n{p2}\n\nRESEARCHER:\n{r2}\n"
    c2 = await critic.call(c2_prompt, influenced=True)

    # DEBUG dump
    debug_dump("planner_influence", p2)
    debug_dump("researcher_influence", r2)
    debug_dump("critic_influence", c2)

    # --- Metrics (JSON-based) ---
    PF1 = extract_features_from_json(p1)
    RF1 = extract_features_from_json(r1)
    PF2 = extract_features_from_json(p2)
    RF2 = extract_features_from_json(r2)

    agr1 = jaccard(PF1, RF1)
    agr2 = jaccard(PF2, RF2)

    planner_self_agree = jaccard(PF1, PF2)
    researcher_self_agree = jaccard(RF1, RF2)

    dec1 = critic_decision_from_json(c1)  # "APPROVE"|"REVISE"|None
    dec2 = critic_decision_from_json(c2)

    rA1 = 1 if dec1 == "APPROVE" else None
    rA2 = 1 if dec2 == "APPROVE" else None

    # Depth of revision: how many new features were added (across both roles)
    revd = len((PF2 | RF2) - (PF1 | RF1))

    planner_researcher_can_baseline = canonical_overlap(p1, r1)
    planner_researcher_can_infl     = canonical_overlap(p2, r2)

    return {
        "beta": beta, "k": k, "tau": tau, "alpha": alpha, "seed": seed, "adversarial": int(adversarial),
        "mu_planner": round(planner.mu, 4),
        "mu_researcher": round(researcher.mu, 4),
        "mu_critic": round(critic.mu, 4),

        "RoundsToApproval_baseline": rA1,
        "RoundsToApproval_influence": rA2,

        "AgreementRate_baseline": None if agr1 is None else round(agr1, 4),
        "AgreementRate_influence": None if agr2 is None else round(agr2, 4),

        "RevisionDepth_between_rounds": int(revd),

        "PlannerResearcher_Canonical_baseline": None if planner_researcher_can_baseline is None else round(planner_researcher_can_baseline, 3),
        "PlannerResearcher_Canonical_influence": None if planner_researcher_can_infl is None else round(planner_researcher_can_infl, 3),

        "Planner_SelfAgreement": None if planner_self_agree is None else round(planner_self_agree, 3),
        "Researcher_SelfAgreement": None if researcher_self_agree is None else round(researcher_self_agree, 3),

        # Auditing info
        "planner_baseline": p1,
        "researcher_baseline": r1,
        "critic_baseline": c1,
        "planner_influence": p2,
        "researcher_influence": r2,
        "critic_influence": c2,
    }


# ---- Sweep runner ----------------------------------------------------------

GRID = {
    "beta":  [0.3],
    "k":     [3.0],
    "tau":   [0.4, 0.5],
    "alpha": [0.4, 0.8, 1.2],
}
SEEDS = [1]  # SEEDS = [1, 2] 

async def main():
    rows: List[Dict] = []
    for adv in (False, True):   
        for beta in GRID["beta"]:
            for k in GRID["k"]:
                for tau in GRID["tau"]:
                    for alpha in GRID["alpha"]:
                        for seed in SEEDS:
                            out = await one_pass(beta, k, tau, alpha, seed, adversarial=adv)
                            log = {kk: out[kk] for kk in ("beta","k","tau","alpha","seed","adversarial",
                                                          "mu_planner","mu_researcher","mu_critic",
                                                          "RoundsToApproval_baseline","RoundsToApproval_influence",
                                                          "AgreementRate_baseline","AgreementRate_influence",
                                                          "RevisionDepth_between_rounds")}
                            print("OK", log)
                            rows.append(out)

    # --- Save CSV (short metrics only) ---
    out_path = "results.csv"
    fieldnames = [
        "beta", "k", "tau", "alpha", "seed", "adversarial",
        "mu_planner", "mu_researcher", "mu_critic",
        "RoundsToApproval_baseline", "RoundsToApproval_influence",
        "AgreementRate_baseline", "AgreementRate_influence",
        "RevisionDepth_between_rounds",
        "PlannerResearcher_Canonical_baseline", "PlannerResearcher_Canonical_influence",
        "Planner_SelfAgreement", "Researcher_SelfAgreement"
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in fieldnames})

    print(f"\nVýsledky uloženy do: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
