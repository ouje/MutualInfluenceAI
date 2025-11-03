# Mutual Influence AI – Demo & Evaluation Suite

This repository contains both a **demo** of a simple multi‑agent system with **Mutual Influence** (peer feedback mechanism) built on top of [autogen-agentchat](https://pypi.org/project/autogen-agentchat/) **and** a lightweight **evaluation harness** for batch experiments and analysis.

Three agents (`planner`, `researcher`, `critic`) collaborate in a round‑robin conversation. They adjust their reasoning based on peer feedback using the mutual influence parameter **μ**.

> All code files are MIT‑licensed and include SPDX headers. Author: **Vaclav Oujezsky** (2025).

---

## Features

- **Multi‑agent collaboration:** Planner, Researcher, Critic
- **Mutual Influence mechanism:**
  - Temperature modulation based on μ
  - λ‑mixing between self vs. peer alignment
  - Exponential‑moving‑average peer scoring (`receive_feedback`)
- **JSON‑only protocol** for model outputs with one‑shot self‑repair
- **Round‑robin group chat** with termination condition (demo)
- **Batch evaluation:**
  - Single‑process or parallel grid sweep
  - Time budget & resume support
  - Short metrics CSV output
- **Streaming console output** for demonstration
- **Analysis notebook** (publication version) for tables & plots

---

## Requirements

- Python **3.10+** (tested on 3.12)
- Virtual environment recommended (`venv`)
- Core dependencies: `autogen-agentchat`, `openai`, `python-dotenv`, `pandas`, `numpy`, `matplotlib`

Install (example):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt  # or install packages listed above
```

Environment variables:

```bash
# via .env or shell; required
export OPENAI_API_KEY="sk-..."
# optional (defaults to gpt-4o)
export OPENAI_MODEL_NAME="gpt-4o"
```

---

## Repository Structure

- `metrics.py` – JSON helpers and agreement/overlap metrics (Jaccard, canonical tag overlap), plus legacy text utilities.
- `mutual_influence_agents.py` – Agent class with μ‑driven temperature and λ gate; OpenAI client with JSON‑only responses.
- `run_grid.py` – Small synchronous sweep; writes **short metrics** to `results.csv`.
- `run_grid_parallel.py` – Parallel sweep with concurrency, resume, and time budget; appends to `results.csv`.
- `mutual_influence_analysis_published.ipynb` – Publication‑oriented analysis notebook (documentation‑first; code unchanged).

---

## Quick Start

### Demo (round‑robin group chat)
Integrate the three agents and run your small task to observe how μ changes behavior. (See `mutual_influence_agents.py` for usage.)

### Evaluation (batch runs)
Run a small grid:

```bash
python run_grid.py
```

Run a parallel grid with resume/time budget:

```bash
python run_grid_parallel.py
```

Both commands write **short metrics** to `results.csv`.

---

## Configuration

- Allowed feature whitelist (used in prompts) lives in the runners:
  ```
  ["flow_bytes","packets","rate","iat","src_ip","dst_ip","src_port","dst_port","protocol","entropy","payload_len"]
  ```
- Model responses are enforced to be **JSON objects** via `response_format={"type":"json_object"}` with a one‑shot self‑repair if keys are missing.
- Mutual‑influence scheduling (see `mutual_influence_agents.py`):
  - `temperature_from_mu(μ, T0=0.7, alpha=0.8)` (clamped to [0.1, 1.5])
  - `lambda_from_mu(μ, k=6.0, tau=0.5)`
  - `receive_feedback(peer, score, beta)` – EMA update

---

## Output (`results.csv`) – Short Metrics Schema

Columns written by the runners:

- `beta`, `k`, `tau`, `alpha`, `seed`, `adversarial`
- `mu_planner`, `mu_researcher`, `mu_critic`
- `RoundsToApproval_baseline`, `RoundsToApproval_influence`
- `AgreementRate_baseline`, `AgreementRate_influence` (Jaccard on feature sets)
- `RevisionDepth_between_rounds`
- `PlannerResearcher_Canonical_baseline`, `PlannerResearcher_Canonical_influence`
- `Planner_SelfAgreement`, `Researcher_SelfAgreement`

> Raw JSON strings are available during a run (see `one_pass`) and can be added to the CSV if needed for auditing.

---

## Analysis Notebook

Use `mutual_influence_analysis_published.ipynb` to generate **publication‑ready** plots/tables:
- Agreement (baseline → influence)
- Approval@Round1
- Sensitivity heatmap (α × β)
- Revision depth
- Adversarial robustness

The notebook includes **Abstract**, **How to Run**, **Data Schema**, and a **Reproducibility checklist**.

---

## Troubleshooting

- **`OPENAI_API_KEY is not set.`** – Provide the key via `.env` or environment.
- **Non‑JSON model output** – Ensure `response_format={"type":"json_object"}` and the prompt text “Return exactly ONE JSON object.”
- **Empty plots / NaNs** – Verify your `results.csv` columns match the schema above.

---

## License

SPDX: `MIT`  
Copyright (c) 2025
**Vaclav Oujezsky**

This project is released under the MIT License. See file headers or the full text below.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Citation

If you use this code or derived figures in a paper or report, please cite:

> Vaclav Oujezsky (2025). **Mutual Influence AI**.  
> *Technical Report / Preprint*. (Will be updated with venue or arXiv ID when available.)