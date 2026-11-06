# Midterm project plan — Rossmann daily sales forecasting

**Status:** working scaffold, 7 November 2026
**Dataset:** Rossmann Store Sales (Kaggle; HF mirror)
**Model:** tuned XGBoost, temporal split
**Test result:** RMSE **936.59**, MAE **677.60**, R² **0.898** (July 2015, never seen)

---

## 0. Snapshot

| Item | Value |
|---|---|
| Task | Daily sales regression, 1,115 stores |
| Rows | 844,338 open store-days (from 1,017,209; 172,871 closed/zero rows removed) |
| Split | train < 2015-06-01; validation June 2015; test July 2015 |
| Baseline | store median: test RMSE 1,776.75 |
| Linear | Ridge (alpha=0.1): test RMSE 1,341.70 |
| Trees | Random forest: test RMSE 1,835.96; XGBoost: **936.59** |
| Artifact | `models/model.joblib` + metadata (22 MB) |
| Service | FastAPI `/health`, `/predict`; verified in Docker |

The model cuts test RMSE by **47% vs the store-median baseline** and by **30% vs Ridge**.

---

## 1. What is built

| File | Purpose | Status |
|---|---|---|
| `README.md` | Problem, data, EDA highlights, results, API, setup, rubric checklist | ✅ |
| `notebook.ipynb` | Executed EDA + model comparison + tuning table + importance + error slices | ✅ |
| `prepare_data.py` | Download, SHA-256 verify, clean, feature engineer, temporal split | ✅ |
| `train.py` | Baseline, Ridge, RF, tuned XGBoost, refit, metrics, artifact | ✅ |
| `predict.py` | FastAPI service with validation and model hash | ✅ |
| `model.py` | Shared feature engineering for training and serving | ✅ |
| `tests/test_api.py` | 7 tests: health, prediction, determinism, validation | ✅ |
| `Dockerfile` | `python:3.12-slim` + `uv sync --locked --no-dev` | ✅ built and run |
| `pyproject.toml` / `uv.lock` | Pinned environment, Python 3.12 | ✅ |
| `data/processed/rossmann.parquet` | 4.3 MB committed processed table | ✅ |
| `models/*.json` | Metrics, tuning, feature categories, store metadata, hashes | ✅ |
| `docs/DATASETS.md` | Shortlist that led to Rossmann | ✅ |
| Cloud deploy | Render/Fly URL + screenshot | ⬜ optional, 2 bonus points |

### Verified end-to-end

```text
uv run pytest -q                  -> 7 passed
uv run jupyter nbconvert --execute -> no errors, 4 figures
docker build -t rossmann-sales .  -> success
container /predict store 1, 2015-08-15, promo 1 -> 4829.68
host     /predict same payload                 -> 4829.68
```

The container and host agree exactly, and `/health` returns the model SHA-256.

---

## 2. Rubric status

| Criterion | Max | Evidence | Status |
|---|---:|---|---|
| Problem description | 2 | README “The problem”: who uses it, decision, metric | ✅ |
| EDA | 2 | README highlights + executed notebook (distributions, promo, weekday, store type, seasonality, missingness) | ✅ can deepen |
| Model training | 3 | Ridge + RF + XGBoost; Ridge alpha and 6-point XGBoost grid; validation selection | ✅ |
| Notebook → script | 1 | `train.py` is the canonical run | ✅ |
| Reproducibility | 1 | committed parquet + raw SHA-256 + `uv.lock` + split metadata | ✅ |
| Deployment | 1 | FastAPI, tested host and container | ✅ |
| Dependency management | 2 | `uv` + `uv.lock` + README setup | ✅ |
| Containerization | 2 | `Dockerfile` + build/run instructions | ✅ |
| Cloud | 2 | not included | ⬜ optional |

Base: **14/16**. Cloud is the only missing rubric item.

---

## 3. What remains before submission

### Must do

1. **Expand the notebook EDA slightly.** It already has the core views; add:
   - a boxplot of sales by store type or day-of-week;
   - a residual histogram;
   - one paragraph on the `Open=0` filtering decision and its effect.
2. **Add a screenshot or short video** of the API response (host and container).
3. **Fresh-clone test:** clone the repo into `/tmp`, `uv sync`, `make test`, `make serve`. Reviewer confidence depends on this.
4. **Record the exact commit SHA** in the course submission form.

### Nice to do (timeboxed)

5. **Cloud deploy (2 points):** Render Docker web service from GitHub. Add `render.yaml`, the public URL, a screenshot, and teardown notes. Timebox: 2 hours. If it fights you, ship 14/16.
6. **CI:** GitHub Actions at the repo root, running `uv sync --locked` and `pytest`. Cheap job signal; not required by the rubric.
7. **README polish:** add the model’s limitations and a “what I would do with more data” paragraph.

### Do not do

- Do not switch datasets again. The deadline is close; the project is coherent.
- Do not tune on the test month. It has been touched once.
- Do not commit `data/raw/` (42 MB) — it is gitignored; the processed parquet and download script are enough.
- Do not replace FastAPI + Docker with Streamlit/Gradio; the rubric expects a service.

---

## 4. Deadline and schedule

The course repo table says midterm submission **9 November 2026**; the live platform (build 6 November) says **16 November 2026, 23:00 UTC = 17 November, 07:00 Manila**, and its homework dates are one week later. **Verify while logged in.** Plan for 16 November.

| Day | Date | Task |
|---|---|---|
| 1 | Fri 7 Nov | Scaffold done (this session); review files; fresh-clone test |
| 2 | Sat 8 Nov | Expand notebook EDA; add residual and boxplot views; re-execute |
| 3 | Sun 9 Nov | Add API screenshots; README polish; commit checkpoint |
| 4 | Mon 10 Nov | Cloud deploy attempt (timebox 2 h) |
| 5 | Tue 11 Nov | Fresh-clone verification; fix any path/lock issues |
| 6 | Wed 12 Nov | Peer-review preparation: read the rubric; plan the 3 reviews |
| 7 | Thu 13 Nov | Buffer; record 60-second demo |
| 8 | Fri 14 Nov | Final read-through; check rubric table |
| 9 | Sat 15 Nov | Submit repo URL + exact commit SHA |
| 10 | Sun 16 Nov | Buffer before 23:00 UTC; start peer reviews |

Peer reviews are mandatory: **3 reviews or the project fails**, due the week after submission.

If the deadline is actually 9 November, skip cloud and CI, finish the notebook, and submit.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| **Temporal leakage** — future data in training | Split by date; test touched once; `Customers` dropped |
| **Closed-day shortcut** — model learns `Open` | Filter `Open=1` and `Sales>0`; document it |
| **Random forest looks bad** | Explain: ordinal store ID + additive structure; XGBoost native categorical wins. Tune forest in notebook if time |
| **Model artifact is 22 MB** | Acceptable under GitHub’s 100 MB limit; do not commit multiple versions |
| **Docker build cache/permissions** | `DOCKER_CONFIG`/`BUILDX_CONFIG` workaround used here; normal Docker Desktop works without it |
| **Cloud eats a day** | Timebox 2 h; 14/16 is already strong |
| **Peer review skipped** | Calendar block; 3 reviews required |
| **Wrong commit submitted** | Put the SHA in the README and the submission form |
| **Notebook outputs stale after edits** | Re-run `jupyter nbconvert --execute --inplace` before committing |

---

## 6. Job-market positioning

From the 7 readable MLE postings sampled on 7 November 2026:

- **Covered by this project:** Python, pandas, scikit-learn, XGBoost, feature engineering, regression metrics, temporal validation, FastAPI, Docker, Git, model metadata.
- **Partially covered:** SQL (not used), MLOps (no registry/monitoring), cloud (optional), CI (optional).
- **Not covered:** LLMs/RAG/agents, vector DBs, PyTorch, Kubernetes, Spark.

**How to pitch it:** “I built an end-to-end sales forecaster with a temporal validation design, tuned XGBoost, a FastAPI service, and a Docker image that reproduces host predictions exactly. The model reduces RMSE by 47% over a store-median baseline; I can explain every feature and every split.”

**Next project (capstone):** an LLM/RAG service with an evaluation harness, because that is now a required skill in PH mid-level postings. The midterm proves the classical ML + serving spine; the capstone should prove the LLM/MLOps layer.

---

## 7. Commands

```bash
cd rossmann-sales-forecasting
uv sync
make data       # optional: rebuild processed parquet from raw
make train      # ~4 min: baseline, Ridge, RF, 6 XGBoost candidates, refit
make serve      # http://localhost:9696
make test       # 7 tests
docker build -t rossmann-sales .
docker run --rm -p 9696:9696 rossmann-sales
```

If `uv` cannot write its cache in a restricted environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

---

## 8. Decisions already made

- **Dataset:** Rossmann (sales), not DataCo (logistics).
- **Target:** `Sales` on open store-days, not `Customers` or a binary “high sales” label.
- **Split:** temporal, not random.
- **Final model:** XGBoost; Ridge and RF kept as comparison.
- **Deployment:** FastAPI + Docker; cloud optional.
- **Scope:** no LLM/RAG in the midterm; save that for the capstone.
