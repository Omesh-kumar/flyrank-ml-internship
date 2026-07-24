# Capstone Report — AI-Powered Content Refresh Prioritization

- **Author:** Omesh Kumar
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Omesh-kumar/flyrank-ml-internship
- **Date:** July 2026

---

## Abstract

Content teams cannot manually review every page on a site to decide which ones need a refresh, so this project asks: which content pages should be prioritized for review based on search performance signals? Using a 30,000-row anonymized sample of FlyRank's content performance data (drawn from a 79-million-row production warehouse spanning March 2026), we built a rule-based baseline score and compared it against a Random Forest Regressor trained on engagement and traffic features. Under an honest, client-grouped validation split, the model reached an R² of 0.698 (MAE 8.66) versus an inflated R² of 0.882 under a naive random split — confirming that grouped validation is necessary to avoid overstating performance. The model's most influential signals were impression frequency, historical clicks, and content length, and it produced a ranked, reason-coded action queue intended to support — not replace — human editorial decisions.

---

## 1. Problem Framing

**Unit of analysis:** one content page (`content_id`), belonging to one client (`client_id`), scored once per analysis run.

**Decision supported:** which content pages should a client's content/SEO team prioritize for review and refresh first, instead of manually reviewing every page.

**Output:** a ranked priority score, a reason code (e.g. `STALE_LOW_CTR`, `LOW_CTR_HIGH_IMPRESSIONS`), and a recommended action (e.g. Refresh Immediately, Monitor).

**Action a human takes:** a content editor or SEO specialist reviews the top of the queue and decides whether to refresh, expand, or leave a page alone.

**Cost of a wrong call:** if the score is wrong, the team may spend time refreshing pages that didn't need it, while pages that are genuinely losing visibility keep declining unnoticed. This is a resource-allocation cost, not a safety-critical one.

**Why ML helps:** a single rule can only combine a couple of signals at a time. Content performance is driven by an interaction of many features (impressions, clicks, position, freshness, content length, engagement), and a model can learn how these interact instead of relying on one fixed formula.

---

## 2. Data Safety

**Source:** FlyRank ML Internship dataset, drawn from the production search-performance warehouse (`fact_content_daily_performance`).

- Full warehouse scale (verified via query): **78,835,655 rows**.
- March 2026 partition alone: **9,841,378 rows**, of which **3,611,061 rows** had GSC (Google Search Console) data available, across all 31 days of the month (2026-03-01 to 2026-03-31).
- The modeling work in this report uses the provided **anonymized sample export**, `content_refresh_anonymized.csv` — 30,000 rows × 44 columns, one row per anonymized `content_id`/`client_id` pair.

**Columns deliberately excluded from modeling:**
- `trend_direction`, `trend_pct` — label-derived fields computed from the same performance window as the target; including them would leak future/derived information into the prediction.
- `client_id`, `content_id` — used only for grouping and identification, never as model features, to avoid the model memorizing per-client identity instead of learning generalizable signal.
- Any manually engineered "future" columns — excluded on principle to keep every feature available *before* the decision point.

**Leakage risk considered:** during feature development (see Section 5 / leakage hunt), a label-derived feature was deliberately added as a test. It pushed R² to a perfect **1.0**, which is a textbook leakage signature — this confirmed the need to strictly separate label-derived fields from the feature set, and that feature was permanently removed.

**Privacy confirmation:** all client and content identifiers in this dataset are pseudonymous hashes (e.g. `client_73cda7b4e4f265ea`, `content_b7e512995f79d5a6`). No real client names, URLs, or raw search queries appear anywhere in this report or in `work/`.

---

## 3. Baseline

**The rule (plain words):** score a page higher if it has gone a long time without an update, has a low click-through rate, and still receives meaningful impressions — because that combination signals wasted visibility.
baseline_score = (days_since_last_update / 30)
+ (100 - ctr * 100)
+ (impressions_90d / 1000)
Every input is available before the decision is made — no future data is used.

**Fairness as a comparison point:** the baseline is fully transparent (a human can compute it by hand) and uses only features that are also given to the model, so any performance gap reflects genuine modeling value, not an unfair advantage.

**Baseline behavior on the data:** across 30,000 pages, the score ranged from roughly -9,900 to 607 (mean ≈ 55.7), with the top of the queue dominated by pages with high impressions, low CTR, and long update gaps (e.g. the top-ranked page had 104 days since last update, 0.14 CTR, and 517,715 impressions in 90 days).

Two supporting signal checks were run before finalizing the rule:
- **CTR vs. position (confirmed):** average CTR dropped steadily from 1.57 (positions 1–5) to 0.15 (positions 51+), matching the expected relationship.
- **Impressions vs. search volume (mixed):** higher impression buckets generally aligned with higher search volume, but the top impression bucket did not have the highest search volume — indicating other factors (ranking, CTR, content quality) also drive traffic.

---

## 4. Model / Analysis

**Method:** Random Forest Regressor (`n_estimators=200`, `random_state=42`), chosen because it captures non-linear interactions between features that a single linear rule cannot, while remaining relatively interpretable via feature importances.

**Final feature list** (after removing leakage columns `ctr`, `impressions_90d`, `days_since_last_update`, and the target itself):
all remaining numeric columns from the anonymized dataset — including `clicks_90d`, `word_count`, `char_count`, `avg_position`, `engagement_rate`, `scroll_rate`, `content_age_days`, `days_with_impressions`, `impressions_last_30d`, `impressions_prev_30d`, and similar historical/structural fields.

**Deliberately left out:** `client_id`, `content_id` (identifiers only), `trend_direction`, `trend_pct` (label-derived).

**Target / proxy definition (one sentence):** the model predicts the Week-4 rule-based `baseline_score` — a composite proxy for "how much this page needs review" built from update recency, CTR, and impression volume — rather than a directly observed ground-truth outcome.

---

## 5. Evaluation

**Split design:** two splits were compared deliberately —
1. **Random 80/20 split** (`random_state=42`) — the naive approach.
2. **Grouped split by `client_id`** (`GroupShuffleSplit`, 80/20, `random_state=42`) — ensures no client appears in both train and test, simulating how the model would perform on a client it has never seen before, which is the realistic deployment scenario.

**Model vs. baseline, same split, same metric:**

| Validation | MAE | RMSE | R² |
|---|---|---|---|
| Week-4 Baseline (rule) | N/A (deterministic rule, not fit) | N/A | N/A |
| Random Forest — Random split | 12.22 | — | 0.882 |
| Random Forest — **Grouped split (honest)** | **8.66** | — | **0.698** |

The grouped split is the number that should be trusted for deployment decisions. The drop from 0.882 to 0.698 is itself an important finding: random splitting overstated performance because it let the model see other pages from the *same* client during training.

**Error analysis:** the model performs strongly overall but is most reliable for clients with longer historical data (more `days_with_impressions`). For clients with sparse or very new history, predictions are noisier — this is expected, since the model has fewer signals to generalize from for those clients.

---

## 6. Interpretation

**Top feature importances (Random Forest):**

| Feature | Importance |
|---|---|
| `days_with_impressions` | 0.570 |
| `clicks_90d` | 0.204 |
| `word_count` | 0.091 |
| `char_count` | 0.015 |
| `avg_position` | 0.014 |

**In plain words:** how consistently a page has been receiving impressions over time, and its historical click volume, dominate the prediction — far more than content length or exact ranking position. This suggests that *sustained visibility history* is a stronger review-priority signal than any single snapshot metric.

**Surprise / negative result:** search volume alone was not a clean predictor of impressions (Signal 2 was "mixed," not confirmed) — the highest search-volume bucket did not correspond to the highest impression bucket. This is a valid negative result: search volume by itself is not a reliable stand-in for actual visibility.

---

## 7. Ranked Recommendations

An editor using this playbook tomorrow would work top-down through this action queue:

| Priority | Action | Reason Code | When it applies |
|---|---|---|---|
| High | Refresh Immediately | `LOW_CTR_HIGH_IMPRESSIONS` | Page gets strong visibility but poor CTR |
| High | Update Content | `STALE_CONTENT` | Content is old and hasn't been touched recently |
| Medium | Improve SEO | `LOW_TRAFFIC` | Metadata/headings/keyword targeting likely weak |
| Medium | Expand Content | `THIN_CONTENT` | Page is too short to compete |
| Low | Monitor | `STABLE_PERFORMANCE` | Performing acceptably, no action needed yet |

**Confidence:** high for clients with rich history; lower for new/sparse clients (see Section 5). **This queue is decision-support only** — every recommendation must pass human review before action (see Section 8/limits below): check current analytics, confirm no conflicting work is already planned, and confirm the recommendation isn't driven by a seasonal or temporary spike.

**Never automate:** publishing/deleting content, legal or compliance-sensitive edits, medical/financial/policy content, or brand-voice decisions — these always require a human.

---

## 8. Limitations & Honest Framing

- This work shows **observed correlations**, not causation — it does not prove that refreshing a page *causes* better performance.
- The target is a **proxy score** (the Week-4 rule), not a directly measured business outcome like revenue or conversions.
- Client histories are unbalanced — some clients have only GSC or only GA4 data, and some started later than others, so cross-client comparisons carry uncertainty.
- The analysis window is limited to **March 2026** and a 30,000-row anonymized sample; it does not capture seasonality across a full year.
- The honest R² (0.698, grouped split) should be trusted over the optimistic random-split number (0.882) — this gap is the report's clearest limitation and its most important lesson.
- Nothing here predicts search engine algorithm behavior; it only reflects patterns in FlyRank's own historical performance data.

---

## 9. Reproducibility

**Environment:** Google Colab, Python 3.12, key packages: `pandas`, `scikit-learn`, `duckdb`.

**To re-run from a fresh clone:**
```bash
git clone https://github.com/Omesh-kumar/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
# Open and run top-to-bottom in order:
# work/notebooks/w04_baseline_score.ipynb
# work/notebooks/w05_model.ipynb
# work/notebooks/w06_validation_audit.ipynb
# work/notebooks/w07_action_playbook.ipynb
# work/notebooks/capstone.ipynb
```

**Random seed:** `random_state=42` used consistently across all train/test splits (random and grouped) for reproducibility.

**Notebooks (source of truth for every number in this report):**
- [`w04_baseline_score.ipynb`](work/notebooks/w04_baseline_score.ipynb) — baseline rule + queue
- [`w05_model.ipynb`](work/notebooks/w05_model.ipynb) — Random Forest model + feature importances
- [`w06_validation_audit.ipynb`](work/notebooks/w06_validation_audit.ipynb) — random vs. grouped split comparison
- [`w07_action_playbook.ipynb`](work/notebooks/w07_action_playbook.ipynb) — ranked action queue + exports
- [`capstone.ipynb`](work/notebooks/capstone.ipynb) — end-to-end mirror of this paper

---

## Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset**, provided by [FlyRank](https://flyrank.ai) as part of its ML Internship program. This work would not have been possible without access to their production-scale search performance data.
