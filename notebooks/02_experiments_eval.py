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

results_dir = results

load_dotenv()
tqdm.pandas()

# %% [markdown]
# ### Load & pre-process dataframes

run_filepaths_all = {
    # Original paper
    "mhills": [
        results / "malvern_hills_multipage_gemini-2.5-pro_split=1.00_seed=00.pkl",
        results / "malvern_hills_multipage_gpt-4o_split=1.00_seed=00.pkl",
        results / "malvern_hills_multipage_gemma-3-27b-it_split=1.00_seed=00.pkl",
    ],
    "iam": [
        results / "iam_multipage_minpages=02_gemini-2.5-pro_split=0.20_seed=00.pkl",
        results / "iam_multipage_minpages=02_gpt-4o_split=0.20_seed=00.pkl",
        results / "iam_multipage_minpages=02_gemma-3-27b-it_split=0.20_seed=00.pkl",
        results / "laia_iam_multipage_minpages=02_split=0.20_seed=00_with_ocr.pkl",
    ],
    "bentham": [
        results / "bentham_multipage_consecutive_gemini-2.5-pro_split=1.00_seed=00.pkl",
        results / "bentham_multipage_consecutive_gpt-4o_split=1.00_seed=00.pkl",
        results / "bentham_multipage_consecutive_gemma-3-27b-it_split=1.00_seed=00.pkl",
    ],
    # Rebuttals
    "mhills_5p+": [
        results_dir
        / "malvern_hills_multipage_min_5p_gemini-2.5-pro_split=1.00_seed=00.pkl",
        results_dir / "malvern_hills_multipage_min_5p_gpt-4o_split=1.00_seed=00.pkl",
        results_dir
        / "malvern_hills_multipage_min_5p_gemma-3-27b-it_split=1.00_seed=00.pkl",
    ],
    "mhills_10p+": [
        results_dir
        / "malvern_hills_multipage_min_10p_gemini-2.5-pro_split=1.00_seed=00.pkl",
        results_dir / "malvern_hills_multipage_min_10p_gpt-4o_split=1.00_seed=00.pkl",
    ],
    "casia": [
        results_dir / "casia_multipage_gemini-2.5-pro_split=0.50_seed=00.pkl",
        results_dir / "casia_multipage_gpt-4o_split=0.50_seed=00.pkl",
    ],
    "iam_multipage_exactpages=05": [
        results_dir
        / "iam_multipage_exactpages=05_gemini-2.5-pro_split=0.23_seed=00.pkl",
        results_dir / "iam_multipage_exactpages=05_gpt-4o_split=0.23_seed=00.pkl",
        results_dir
        / "iam_multipage_exactpages=05_gemma-3-27b-it_split=0.23_seed=00.pkl",
    ],
    "iam_multipage_random_exactpages=05": [
        results_dir
        / "iam_multipage_random_exactpages=05_gemini-2.5-pro_split=1.00_seed=00.pkl",
        results_dir
        / "iam_multipage_random_exactpages=05_gpt-4o_split=1.00_seed=00.pkl",
        results_dir
        / "iam_multipage_random_exactpages=05_gemma-3-27b-it_split=1.00_seed=00.pkl",
    ],
}

get_ocr_engine = lambda dataset: "google_ocr" if "casia" in dataset else "azure_ocr"

get_alt_methods = lambda dataset: (
    ["docowl2_ocr", "docowl2_lines_ocr", "trocr_ocr"]
    if dataset in ["mhills_5p+"]
    else []
)

dfs_all = {}
for dataset, run_filepaths in run_filepaths_all.items():

    dfs = [pd.read_pickle(run_filepath) for run_filepath in run_filepaths]

    # Concatenate along columns. For columns that are the same, keep leftmost
    df = pd.concat(dfs, axis=1)

    # Rename all vision-pbp to just the base model name
    df = df.rename(columns={col: col.replace("vision-pbp->", "") for col in df.columns})

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Add zero-cost columns for alt OCR methods
    for alt_method in get_alt_methods(dataset):
        df[f"{alt_method}_cost"] = 0.0
        df[f"{alt_method}_input_cost"] = 0.0
        df[f"{alt_method}_output_cost"] = 0.0
        df[f"{alt_method}_ocr_cost"] = 0.0
        df[f"{alt_method}_input_img_tokens"] = 0
        df[f"{alt_method}_input_txt_tokens"] = 0
        df[f"{alt_method}_output_tokens"] = 0
        df[alt_method] = df.apply(
            lambda row: {
                f"{page_id:03d}": page_txt
                for page_id, page_txt in zip(row["page_id"], row[alt_method])
            },
            axis=1,
        )
        pass

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
# Failed call type


def check_error_types(failed_calls: list[str]) -> dict[str, int]:
    res = {
        "disconnected": 0,
        "resource_exhausted": 0,
        "failed": 0,
        "incomplete": 0,
        "invalid_json_schema": 0,
    }
    for f in failed_calls:
        if "Server disconnected without sending a response" in f:
            res["disconnected"] += 1
        elif "RESOURCE_EXHAUSTED" in f:
            res["resource_exhausted"] += 1
        elif "response_failed: True" in f:
            # count as disconnected
            res["failed"] += 1
        elif "response_incomplete: True" in f:
            res["incomplete"] += 1
        elif "invalid_json_schema: True" in f:
            res["invalid_json_schema"] += 1
        elif "'NoneType' object has no attribute 'parts'" in f:
            # gemini api returned an an empty candidate due to an unfinished response, truncation etc.
            res["incomplete"] += 1
        elif "Failed to parse JSON" in f:
            res["invalid_json_schema"] += 1
        else:
            raise NotImplementedError(f"Unknown failed call message: {f}")
    return res


# %% [markdown]
### Get all scores
results_all = {}
for dataset, df in dfs_all.items():
    modes = [
        col.replace("_input_cost", "")
        for col in df.columns
        if col.endswith("_input_cost")
    ]

    baseline_mode, baseline_scores = get_ocr_engine(dataset), {}
    modes = [baseline_mode] + [m for m in modes if m != baseline_mode]

    cers, results = {}, {}
    for mode in modes:
        logger.info(f"Logging runs for mode {mode}...")
        agg_scores = []
        for id, row in tqdm(df.iterrows(), total=len(df)):
            page_ids = row["page_id"]
            pred = row[mode]
            if not pred:
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

            scores["input_cost/kdoc"] = row[f"{mode}_input_cost"] * 1000
            scores["output_cost/kdoc"] = row[f"{mode}_output_cost"] * 1000
            scores["ocr_cost/kdoc"] = row[f"{mode}_ocr_cost"] * 1000
            scores["cost/kdoc"] = (
                scores["input_cost/kdoc"]
                + scores["output_cost/kdoc"]
                + scores["ocr_cost/kdoc"]
            )
            scores["input_img_tokens"] = row[f"{mode}_input_img_tokens"]
            scores["input_txt_tokens"] = row[f"{mode}_input_txt_tokens"]
            scores["output_tokens"] = row[f"{mode}_output_tokens"]
            if f"{mode}_time" in row:
                scores["time"] = row[f"{mode}_time"]
            else:
                if mode != baseline_mode:
                    logger.warning(f"Missing time for mode {mode} at id {id}.")
                scores["time"] = 0.0
            if f"{mode}_failed_calls" in row:
                error_type_counts = check_error_types(row[f"{mode}_failed_calls"])
                for error_type, count in error_type_counts.items():
                    scores[f"failed_calls_{error_type}"] = count
                scores["failed_calls"] = len(row[f"{mode}_failed_calls"])
            else:
                if mode != baseline_mode:
                    logger.warning(f"Missing failed calls for mode {mode} at id {id}.")
                scores["failed_calls"] = 0
                scores["failed_calls_disconnected"] = 0
                scores["failed_calls_resource_exhausted"] = 0
                scores["failed_calls_failed"] = 0
                scores["failed_calls_incomplete"] = 0
                scores["failed_calls_invalid_json_schema"] = 0
            agg_scores.append(scores)

        cers[mode] = [c["CER"] for c in agg_scores]
        results[mode] = pd.DataFrame(agg_scores).mean().to_dict()

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
        if ocr in {"gemini-2.5-pro", "gpt-4o", "gemma-3-27b-it"}:
            postproc, ocr = ocr, ""  # treat LLM as postproc to nothing, not as OCR
        res["ocr"].append(ocr)
        res["extra"].append(extra)
        res["postproc"].append(postproc)

    # Add new columns
    results["ocr"] = pd.Series(res["ocr"], index=results.index)
    results["extra"] = pd.Series(res["extra"], index=results.index)
    results["postproc"] = pd.Series(res["postproc"], index=results.index)
    results = results[
        [
            "ocr",
            "extra",
            "postproc",
            "CER",
            "ocr_cost/kdoc",
            "input_cost/kdoc",
            "output_cost/kdoc",
            "cost/kdoc",
            "input_img_tokens",
            "input_txt_tokens",
            "output_tokens",
            "time",
            "failed_calls",
            "failed_calls_disconnected",
            "failed_calls_resource_exhausted",
            "failed_calls_failed",
            "failed_calls_incomplete",
            "failed_calls_invalid_json_schema",
        ]
    ]

    results_all[dataset] = results

# %% [markdown]
# ### Save results
# The below cells save pickled dataframes for each dataset's results table

from judge_htr import results as tables_dir

tables_dir = results_dir

# %% [markdown]
# # Malvern-Hills results
results = results_all["mhills"]

results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])

results.to_pickle(tables_dir / "mhills_results.pkl")

dfs_all["mhills"].to_pickle(tables_dir / "all_results_mhills.pkl")

logger.info("\n=== Malvern Hills results ===\n")
print(results.reset_index(drop=True))

# %% [markdown]
# # IAM results
results = results_all["iam"]

results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])
results.to_pickle(tables_dir / "iam_results.pkl")

dfs_all["iam"].to_pickle(tables_dir / "all_results_iam.pkl")

logger.info("\n=== IAM results ===\n")
print(results.reset_index(drop=True))

# %% [markdown]
# # Bentham results
results = results_all["bentham"]

results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])
results.to_pickle(tables_dir / "bentham_results.pkl")

dfs_all["bentham"].to_pickle(tables_dir / "all_results_bentham.pkl")

logger.info("\n=== Bentham results ===\n")
print(results.reset_index(drop=True))

# %% [markdown]
# # IAM 5+ pages results
results = results_all["iam_multipage_exactpages=05"]

results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])
results.to_pickle(tables_dir / "iam_multipage_exactpages=05_results.pkl")

dfs_all["iam_multipage_exactpages=05"].to_pickle(
    tables_dir / "all_results_iam_multipage_exactpages=05.pkl"
)

logger.info("\n=== IAM (5+) results ===\n")
print(results.reset_index(drop=True))


# %% [markdown]
# # IAM-5-Random pages results
results = results_all["iam_multipage_random_exactpages=05"]

results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])
results.to_pickle(tables_dir / "iam_multipage_random_exactpages=05_results.pkl")
dfs_all["iam_multipage_random_exactpages=05"].to_pickle(
    tables_dir / "all_results_iam_multipage_random_exactpages=05.pkl"
)

logger.info("\n=== IAM-5-Random results ===\n")
print(results.reset_index(drop=True))


# %% [markdown]
results = results_all["casia"]
results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])
logger.info("\n=== CASIA 5 pages results ===\n")
print(results.reset_index(drop=True))

results = results[results["ocr"] != "tesseract_ocr"]

results.to_pickle(tables_dir / "casia_results.pkl")

# %% [markdown]
results = results_all["mhills_5p+"]
results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])
logger.info("\n=== Malvern Hills min 5 pages results ===\n")
print(results.reset_index(drop=True))

results.to_pickle(tables_dir / "mhills_min_5p_results.pkl")

# %% [markdown]
results = results_all["mhills_10p+"]
results = results.sort_values(by=["postproc", "CER"], ascending=[False, False])
logger.info("\n=== Malvern Hills min 10 pages results ===\n")
print(results.reset_index(drop=True))

results.to_pickle(tables_dir / "mhills_min_10p_results.pkl")


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
