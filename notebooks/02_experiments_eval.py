# %% [markdown]
# Evaluate results from experiments run in 01_experiments.py
# and save summary tables (see Tables 1 and 2 in the paper)

import pandas as pd
from judge_htr import results
from loguru import logger
import os
from tqdm import tqdm
from dotenv import load_dotenv
from judge_htr.transcription_errors import get_transcription_errors
from judge_htr import data
from judge_htr.mllm.caller import MLLM, api_call
from evaluate import load
import ast
from collections import Counter

try:
    logger.level("COST", no=35, color="<yellow>")
except:
    logger.info("Logger already set up.")

load_dotenv()
tqdm.pandas()

# %% [markdown]
# ### Load & pre-process dataframes

run_filepaths_all = {
    "mhills": [
        results / "malvern_hills_multipage_gemini-2.5-pro_split=1.00_seed=00.pkl",
        results / "malvern_hills_multipage_gpt-4o_split=1.00_seed=00.pkl",
        results / "malvern_hills_multipage_gemma-3-27b-it_split=1.00_seed=00.pkl",
    ],
    "iam": [
        results / "iam_multipage_minpages=02_gemini-2.5-pro_split=0.20_seed=00.pkl",
        results / "iam_multipage_minpages=02_gpt-4o_split=0.20_seed=00.pkl",
        results / "iam_multipage_minpages=02_gemma-3-27b-it_split=0.20_seed=00.pkl",
    ],
    "bentham": [
        results / "bentham_multipage_consecutive_gemini-2.5-pro_split=1.00_seed=00.pkl",
        results / "bentham_multipage_consecutive_gpt-4o_split=1.00_seed=00.pkl",
        results / "bentham_multipage_consecutive_gemma-3-27b-it_split=1.00_seed=00.pkl",
    ],
}

dfs_all = {}
for dataset, run_filepaths in run_filepaths_all.items():

    dfs = [pd.read_pickle(run_filepath) for run_filepath in run_filepaths]

    # Concatenate along columns. For columns that are the same, keep leftmost
    df = pd.concat(dfs, axis=1)

    # Rename all vision-pbp to just the base model name
    df = df.rename(columns={col: col.replace("vision-pbp->", "") for col in df.columns})

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    dfs_all[dataset] = df

# %% [markdown]
# #### Eval metrics
cer = load("cer")
wer = load("wer")

score_functions = {
    "CER": lambda gt, pred: cer.compute(
        predictions=pred,
        references=gt,
    ),
    "WER": lambda gt, pred: wer.compute(
        predictions=pred,
        references=gt,
    ),
}

# %% [markdown]
### Get all scores

results_all = {}
for dataset, df in dfs_all.items():
    modes = [col.replace("_cost", "") for col in df.columns if col.endswith("_cost")]

    baseline_mode, baseline_scores = "azure_ocr", {}
    modes = [baseline_mode] + [m for m in modes if m != baseline_mode]

    cers, results = {}, {}
    for mode in modes:
        logger.info(f"Logging runs for mode {mode}...")
        agg_scores = []
        for id, row in tqdm(df.iterrows(), total=len(df)):
            page_ids = row["page_id"]
            pred = row[mode]
            if pred is None:
                preds = [""] * len(page_ids)
                logger.warning(f"Missing predictions for id {id} mode {mode}.")
            else:
                pred = {int(k): str(v) for k, v in pred.items()}
                preds = [pred[page_id] for page_id in page_ids]
            gts = [row["indexed_gt"][page_id] for page_id in page_ids]
            scores = {}
            for name, fn in score_functions.items():
                scores[name] = fn(gts, preds)
                # Pages after the first only
                scores[f"{name}-2+"] = fn(gts[1:], preds[1:])

            scores["cost"] = row.get(f"{mode}_cost", 0.0)
            scores["cost/kdoc"] = scores["cost"] * 1000

            agg_scores.append(scores)

        cers[mode] = [c["CER"] for c in agg_scores]
        results[mode] = pd.DataFrame(agg_scores).mean().to_dict()

    #
    for mode, cer_list in cers.items():
        df[f"{mode}_CER"] = cer_list

    dfs_all[dataset] = df
    results_all[dataset] = pd.DataFrame(results).T


# %% Reformat mode names

for dataset, results in results_all.items():

    names = results.index.tolist()
    names = [
        name.split("->")[1] + "+non-pbp" if name.startswith("vision->") else name
        for name in names
    ]

    res = {"ocr": [], "extra": [], "postproc": []}
    for name in names:
        tmp = name.split("->")
        if len(tmp) == 1 and "+" not in tmp[0]:
            ocr, extra, postproc = tmp[0], "", ""
        elif len(tmp) == 1 and "+" in tmp[0]:
            rem = tmp[0].split("+")
            ocr = rem[0]
            extra = rem[1]
            postproc = ""
        else:
            rem, postproc = tmp
            rem = rem.split("+")
            ocr = rem[0]
            extra = rem[1] if len(rem) > 1 else ""
        res["ocr"].append(ocr)
        res["extra"].append(extra)
        res["postproc"].append(postproc)

    # Add new columns
    results["ocr"] = pd.Series(res["ocr"], index=results.index)
    results["extra"] = pd.Series(res["extra"], index=results.index)
    results["postproc"] = pd.Series(res["postproc"], index=results.index)
    results[["ocr", "extra", "postproc", "CER", "cost/kdoc"]]

    results_all[dataset] = results

# %% [markdown]
# ### Save results

from judge_htr import results as tables_dir

# %% [markdown]
# # Malvern-Hills results
results = results_all["mhills"]

results = results.sort_values(by="CER")[
    ["ocr", "extra", "postproc", "CER", "cost/kdoc"]
]

results.to_pickle(tables_dir / "mhills_results.pkl")

dfs_all["mhills"].to_pickle(tables_dir / "all_results_mhills.pkl")

logger.info("\n=== Malvern Hills results ===\n")
print(results.reset_index(drop=True))

# %% [markdown]
# # IAM results
results = results_all["iam"]

results = results.sort_values(by="CER")[
    ["ocr", "extra", "postproc", "CER", "cost/kdoc"]
]
results.to_pickle(tables_dir / "iam_results.pkl")

dfs_all["iam"].to_pickle(tables_dir / "all_results_iam.pkl")

logger.info("\n=== IAM results ===\n")
print(results.reset_index(drop=True))

# %% [markdown]
# # Bentham results
results = results_all["bentham"]

results = results.sort_values(by="CER")[
    ["ocr", "extra", "postproc", "CER", "cost/kdoc"]
]
results.to_pickle(tables_dir / "bentham_results.pkl")

dfs_all["bentham"].to_pickle(tables_dir / "all_results_bentham.pkl")

logger.info("\n=== Bentham results ===\n")
print(results.reset_index(drop=True))


# %%
# # LLM-assessed error types
# See the Minor and Major Error columns in Table 1 for IAM, and full results in Table 6


# %% [markdown]
# ### Get error types for each mode for IAM dataset

dataset = "iam"

cache_paths = {"iam": data / "gpt_cache_iam.db"}

mllm = MLLM(
    model="gemini-2.5-flash",
    api_call=api_call,
    verbose=False,
    update_cache_every_n_calls=5,
    cache_path=cache_paths[dataset],
)

modes = [
    col.replace("_cost", "")
    for col in dfs_all[dataset].columns
    if col.endswith("_cost")
]

errors = {}
for mode in tqdm(modes):
    print(f"Running for mode {mode}...")
    errors[mode] = {}
    for id, row in tqdm(dfs_all[dataset].iterrows(), total=len(dfs_all[dataset])):
        page_ids = row["page_id"]
        preds = {int(k): str(v) for k, v in row[mode].items()}
        res = get_transcription_errors(
            gts=row["indexed_gt"],
            preds=preds,
            mllm=mllm,
        )
        errors[mode][id] = res
    mllm.update_cache()


# %% Get errors into the df
errors_df = pd.DataFrame(errors)

# Count errors per mode (each column)
totals = {}
for col in errors_df.columns:
    c = Counter()
    for cell in errors_df[col].dropna():
        # parse cell -> dict of {section: {'errors': [...], 'diff': ...}, ...}
        d = (
            ast.literal_eval(cell)
            if isinstance(cell, str)
            else cell if isinstance(cell, dict) else None
        )
        if not isinstance(d, dict):
            continue
        for section in d.values():
            for e in section.get("errors", []):
                et = e.get("error_type")
                if et:
                    c[et] += 1
    totals[col] = c

# Make the results table: rows = modes, cols = error types
results = pd.DataFrame.from_dict(totals, orient="index").fillna(0).astype(int)
results.index.name = "mode"
results["total"] = results.sum(axis=1)

# "formatting" and "semantic" are "minor" errors, the rest are "major".
# Add "major_total" and "minor_total" columns
minor_errors = ["formatting", "semantic"]
results["minor"] = results[minor_errors].sum(axis=1)
results["major"] = results["total"] - results["minor"]

# Add percentage columns for each error type and subtype, in the col to the right
col_arrangement = []
for col in results.columns:
    if col == "total":
        continue
    results[f"{col}_%"] = (results[col] / results["total"] * 100).round(1)
    col_arrangement.append(col)
    col_arrangement.append(f"{col}_%")
results = results[col_arrangement + ["total"]]

results["minor_errors_per_doc"] = results["minor"] / len(errors_df)
results["major_errors_per_doc"] = results["major"] / len(errors_df)
results["errors_per_doc"] = results["total"] / len(errors_df)


error_types_results = results.sort_values(by="major")

# %% [markdown]
# # Error types results for IAM
logger.info("\n=== IAM error types results ===\n")
print(error_types_results)
