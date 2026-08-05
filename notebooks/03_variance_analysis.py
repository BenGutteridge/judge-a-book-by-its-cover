# %% [markdown]
# # Variance analysis for the COLM 2026 rebuttal
#
# Re-runs of OCR+PAGE1 / OCR+PAGEN (Gemini-2.5-Pro, temperature=0) across 3 API seeds, on IAM
# and Malvern-Hills. We report, per method:
#   - per-seed mean CER, and mean +/- std + 95% CI across the 3 seeds
#   - the original paper number (reference) for the match-check
#   - % of (doc, page) transcriptions byte-identical across all 3 seeds (determinism stat)
#   - how many docs changed across seeds, and what fraction of those had >=1 JSON retry
#     (evidence that seed-variation perturbs exactly the retry-affected docs the reviewer flagged)
#
# CER is computed identically to notebooks/02_experiments_eval.py: per-doc CER (over the doc's
# pages), then mean over docs.

# %%
import math
from pathlib import Path
import pandas as pd
from evaluate import load
from judge_htr import results

cer = load("cer")

MODEL = "gemini-2.5-pro"
SEEDS = [1, 2, 3]
T_975_DF2 = 4.302653  # Student-t 97.5th pct, df=2 (n=3)
T_975_DF3 = 3.182446  # Student-t 97.5th pct, df=3 (n=4, including paper as seed 0)

DATASETS = {
    "IAM": {"task": "iam_multipage_minpages=02", "split": "0.20"},
    "Malvern-Hills": {"task": "malvern_hills_multipage", "split": "1.00"},
}

# methods to report (baseline OCR is a zero-variance anchor)
def modes_for(task: str) -> list[tuple[str, str]]:
    """Returns list of (mode_column_name, run_label) tuples."""
    # IAM pkls have no run_label (created before the labelling system was added)
    pagex_label = "pagex" if task == "malvern_hills_multipage" else ""
    modes = [
        ("azure_ocr+page1->" + MODEL, pagex_label),
        ("azure_ocr+pageN->" + MODEL, pagex_label),
    ]
    if task == "malvern_hills_multipage":
        modes += [
            ("vision-pbp->" + MODEL, "images"),
            ("azure_ocr+all-pages-pbp->" + MODEL, "images"),
        ]
    return modes


DISPLAY_NAMES = {
    "azure_ocr+page1->" + MODEL: "OCR+PAGE1",
    "azure_ocr+pageN->" + MODEL: "OCR+PAGEN",
    "vision-pbp->" + MODEL:      "IMAGES",
    "azure_ocr+all-pages-pbp->" + MODEL: "OCR+IMAGES",
}


def pkl_path(task: str, split: str, api_seed, run_label: str = "") -> Path:
    base = f"{task}_{MODEL}_split={split}_seed=00"
    if api_seed is None:
        return results / f"{base}.pkl"
    suffix = f"_apiseed={api_seed:02d}"
    if run_label:
        suffix += f"_{run_label}"
    return results / f"{base}{suffix}.pkl"


def per_doc_cers(df: pd.DataFrame, mode: str) -> dict:
    """{doc_index: CER} for one method, matching 02_experiments_eval.py."""
    out = {}
    for idx, row in df.iterrows():
        page_ids = row["page_id"]
        pred = row.get(mode) if mode in df.columns else None
        if not pred:
            preds = [""] * len(page_ids)
        else:
            pred = {int(k): str(v) for k, v in pred.items()}
            preds = [pred.get(p, "") for p in page_ids]
        gts = [row["indexed_gt"][p] for p in page_ids]
        out[idx] = cer.compute(predictions=preds, references=gts)
    return out


def outputs_map(df: pd.DataFrame, mode: str) -> dict:
    """{(doc_index, page_id): text} for cross-seed identical comparison."""
    out = {}
    for idx, row in df.iterrows():
        pred = (row.get(mode) if mode in df.columns else None) or {}
        pred = {int(k): str(v) for k, v in pred.items()}
        for p in row["page_id"]:
            out[(idx, p)] = pred.get(p, "")
    return out


def retry_counts(df: pd.DataFrame, mode: str) -> dict:
    """{doc_index: number of failed/retried calls} for one method."""
    col = f"{mode}_failed_calls"
    if col not in df.columns:
        return {idx: 0 for idx in df.index}
    return {idx: len(v) if isinstance(v, (list, tuple)) else 0 for idx, v in df[col].items()}


# %%
rows = []
for ds_name, cfg in DATASETS.items():
    task, split = cfg["task"], cfg["split"]
    # load reference (paper) — seed 0, no label
    paper_df = pd.read_pickle(pkl_path(task, split, None))

    for mode, run_label in modes_for(task):
        # load seeds 1-3; paper pkl is seed 0
        seed_dfs = {0: paper_df}
        for s in SEEDS:
            p = pkl_path(task, split, s, run_label)
            if p.exists():
                seed_dfs[s] = pd.read_pickle(p)
            else:
                print(f"WARNING: missing {p.name}")
        present = sorted(seed_dfs)
        if not present:
            continue
        # per-seed mean CER (paper = seed 0)
        seed_cer = {s: per_doc_cers(seed_dfs[s], mode) for s in present}
        seed_means = {s: (sum(v.values()) / len(v)) for s, v in seed_cer.items()}

        means = [seed_means[s] for s in present]
        n = len(means)
        mean = sum(means) / n if n else float("nan")
        std = (sum((m - mean) ** 2 for m in means) / (n - 1)) ** 0.5 if n > 1 else 0.0
        # use t df=3 (n=4) when all seeds present, df=2 (n=3) as fallback
        t_crit = T_975_DF3 if n == 4 else T_975_DF2
        ci = t_crit * std / math.sqrt(n) if n > 1 else 0.0

        # cross-seed determinism: identical (doc,page) outputs across all present seeds
        omaps = {s: outputs_map(seed_dfs[s], mode) for s in present}
        keys = set().union(*[set(m) for m in omaps.values()]) if omaps else set()
        identical = sum(1 for k in keys if len({omaps[s].get(k) for s in present}) == 1)
        ident_pct = 100.0 * identical / len(keys) if keys else float("nan")

        # docs whose output changed across seeds, vs docs with >=1 retry
        doc_ids = set().union(*[set(seed_dfs[s].index) for s in present]) if present else set()
        rmaps = {s: retry_counts(seed_dfs[s], mode) for s in present}
        n_changed = n_retry = n_changed_and_retry = 0
        for d in doc_ids:
            pages = [k for k in keys if k[0] == d]
            changed = any(len({omaps[s].get(k) for s in present}) > 1 for k in pages)
            had_retry = any(rmaps[s].get(d, 0) > 0 for s in present)
            n_changed += changed
            n_retry += had_retry
            n_changed_and_retry += changed and had_retry

        rows.append(
            {
                "dataset": ds_name,
                "method": DISPLAY_NAMES.get(mode, mode),
                **{f"seed{s}_CER%": round(seed_means[s] * 100, 3) for s in present},
                "mean_CER%": round(mean * 100, 3),
                "std%": round(std * 100, 4),
                "ci95_halfwidth%": round(ci * 100, 4),
                "identical_output_%": round(ident_pct, 2),
                "n_docs_changed": n_changed,
                "n_docs_with_retry": n_retry,
                "changed_docs_with_retry": n_changed_and_retry,
                "n_docs": len(doc_ids),
            }
        )

summary = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(summary.to_string(index=False))

out_csv = results / "variance_analysis_summary.csv"
summary.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

# %% markdown table for the rebuttal
print("\n\n### Markdown (per method)\n")
cols = ["dataset", "method"] + [f"seed{s}_CER%" for s in [0] + list(SEEDS)] + [
    "mean_CER%", "std%", "ci95_halfwidth%", "identical_output_%",
    "n_docs_changed", "n_docs_with_retry", "changed_docs_with_retry",
]
cols = [c for c in cols if c in summary.columns]
print("| " + " | ".join(cols) + " |")
print("|" + "|".join(["---"] * len(cols)) + "|")
for _, r in summary.iterrows():
    print("| " + " | ".join(str(r[c]) for c in cols) + " |")
