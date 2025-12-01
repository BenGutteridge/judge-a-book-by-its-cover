# %% [markdown]
# Get per-dociment type and per-document feature ablations for Malvern-Hills-5+ dataset.

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
    "mhills_5p+": [
        results_dir
        / "malvern_hills_multipage_min_5p_gemini-2.5-pro_split=1.00_seed=00.pkl",
    ]
}

get_ocr_engine = lambda dataset: "google_ocr" if "casia" in dataset else "azure_ocr"
get_alt_methods = lambda _: []

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
# Add image types
from judge_htr import data

df = dfs_all.pop("mhills_5p+")

image_types = pd.read_pickle(
    data / "malvern_hills_trust" / "malvern_hills_image_types.pkl"
)

img_types_col = {}
for _, row in df.iterrows():
    doc_id = row["doc_id"]
    img_types = set()
    for page_id in row["page_id"]:
        full_page_id = f"{doc_id}_{page_id:03d}"
        img_type = image_types.loc[full_page_id, "primary_type"]
        img_types.add(img_type)
    img_types_col[row.name] = img_types

df["doc_primary_types"] = pd.Series(img_types_col)
df["doc_primary_types"] = df["doc_primary_types"].apply(lambda x: ",".join(sorted(x)))

df["doc_primary_types"].value_counts()


# %% [markdown]
# Convert dfs_all into ablations of different features of mhills rather than different datasets


for trait in ["tabular", "archaic", "multiple_hands"]:
    dfs_all[trait] = df[df[trait]]
    dfs_all[f"no_{trait}"] = df[~df[trait]]
    logger.info(
        f"{trait}: {len(dfs_all[trait])} vs no_{trait}: {len(dfs_all[f'no_{trait}'])}"
    )

doc_types = df["doc_primary_types"].unique()
for doc_type in doc_types:
    dfs_all[doc_type] = df[df["doc_primary_types"] == doc_type]
    logger.info(f"type_{doc_type}: {len(dfs_all[doc_type])}")


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

from judge_htr import results as tables_dir

tables_dir = results_dir

res = {}

for thing in [
    "tabular",
    "no_tabular",
    "archaic",
    "no_archaic",
    "multiple_hands",
    "no_multiple_hands",
] + list(doc_types):
    res[thing] = results_all[thing][["CER"]].sort_values(by="CER", ascending=False)
    res[thing]["CER"] = res[thing]["CER"].map(lambda x: f"{x*100:.2f}")

res_md = "\n\n".join(
    [f"#### {thing}\n\n" + table.to_markdown() for thing, table in res.items()]
)

# %% Save to file

res = {}
for doc_type in doc_types:
    df_type = dfs_all[doc_type][["tabular", "archaic", "multiple_hands"]].mean()
    # all 3 cols to %s, *100, 1 dp
    for col in df_type.index:
        df_type[col] = f"{df_type[col]*100:.1f}"
    res[doc_type] = df_type

    if "inventory" not in doc_type:
        all_years = []
        max_year, min_year = 0, 9999
        for _, row in dfs_all[doc_type].iterrows():
            written_year = row["written_year"]
            if None in written_year:
                continue
            for k, v in written_year.items():
                all_years.extend([k for _ in range(v)])
            max_year = max(max_year, max(written_year.keys()))
            min_year = min(min_year, min(written_year.keys()))
        # written year: median, min, max
        df_type["written_year min \| median \| max"] = (
            f"{min_year} \| {int(pd.Series(all_years).median())} \| {max_year}"
        )
    else:
        df_type["written_year min \| median \| max"] = "---"
    df_type["num_docs"] = len(dfs_all[doc_type])
    df_type["num_pages"] = dfs_all[doc_type]["page_id"].apply(len).sum()
res_df = pd.DataFrame(res).T

# %%
