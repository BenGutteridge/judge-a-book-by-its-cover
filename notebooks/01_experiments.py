# %% [markdown]
# # Multi-page transcription experiments notebook

# # # # For IAM dataset
# task = "iam_multipage_minpages=02"
# task = "iam_multipage_random_exactpages=05" # rebuttal
# task = "iam_multipage_exactpages=05" # rebuttal

# For Malvern-Hills dataset
# task = "malvern_hills_multipage"
# task = "malvern_hills_multipage_min_5p" # rebuttal
# task = "malvern_hills_multipage_min_10p" # rebuttal

# # For Bentham dataset
# task = "bentham_multipage_consecutive"

# # For CASIA dataset
# task = "casia_multipage" # rebuttal

# %% [markdown]
# ## Setup
from judge_htr import data, configs, results
from judge_htr.mllm.caller import MLLM, api_call
from judge_htr.postprocessing import (
    make_dummy_response,
    json_loads_safe,
    approximate_gemini_token_counter,
)
from loguru import logger
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from judge_htr.postprocessing import mllm_output_postprocessing as str_proc
from typing import Optional
from omegaconf import OmegaConf
import sys
import random

from judge_htr.prompts import (
    make_page_selection_schema,
    make_paged_transcription_json,
    make_paged_transcription_schema,
)

try:
    logger.level("COST", no=35, color="<yellow>")
except:
    logger.info("Logger already set up.")


configs_dir = configs
results_dir = results
results_dir.mkdir(exist_ok=True)


load_dotenv()
tqdm.pandas()
args = {
    k: v for k, v in zip(sys.argv[1::2], sys.argv[2::2])
}  # command line args - override config/defaults

task = args.get("task", task)
cfg = dict(OmegaConf.load(configs_dir / f"{task}.yaml")) | args
logger.info("\n" + OmegaConf.to_yaml(cfg))

# Import correct task-specific prompts
from judge_htr.prompts import prompts


# OCR costs $ / doc
ocr_costs = {
    "azure_ocr": 1e-3,
    "tesseract_ocr": 0.0,
    "google_ocr": 1.5e-3,
}


model = cfg["model"]
# OCR engines to use
ocr_engines = cfg["ocr_engines"]

mllm = MLLM(
    model=model,
    api_call=api_call,
    verbose=False,
    cache_path=(data / cfg["mllm_cache_path"]),
    update_cache_every_n_calls=5,
)

api_call_kwargs = {
    "top_logprobs": 5,
    "temperature": 0,
    # "load_from_cache": False,
    "load_failed_calls_from_cache": True,
}


# %% [markdown]
# ### Load dataset and set split

dataset_path = data / cfg["dataset_dir"] / f"{task}.pkl"
split, seed = float(cfg["split"]), int(cfg["seed"])
df = pd.read_pickle(data / dataset_path)
logger.info(f"\nLoaded {len(df)} docs from {dataset_path}")

assert df.index.is_unique, "Index not unique"

df = df.sample(frac=split, random_state=seed)
random.seed(seed)
logger.info(f"\nUsing split of size {split} ({len(df)} docs)")


page_counts = df["gt"].apply(len)
logger.info(f"\nDistribution of page counts:\n{page_counts.value_counts()}")

df["page_id"] = df["page_id"].apply(
    lambda page_id_list: [int(i) for i in page_id_list]
)  # turn page id list into ints

df["indexed_img_path"] = df.apply(
    lambda row: {pid: row["img_path"][i] for i, pid in enumerate(row["page_id"])},
    axis=1,
)

# Index gt with the same page IDs as indexed_img_path
df["indexed_gt"] = df.apply(
    lambda row: {pid: row["gt"][i] for i, pid in enumerate(row["page_id"])}, axis=1
)

# OCR also in dict format
for ocr_engine in ocr_engines:
    if ocr_engine in df.columns:
        df[ocr_engine] = df.apply(
            lambda row: {
                pid: row[ocr_engine][i] for i, pid in enumerate(row["page_id"])
            },
            axis=1,
        )


# Set up output files and batch API files if necessary
output_filename = f"{task}_{model}_split={split:.2f}_seed={seed:02d}"
if cfg["batch_api"]:
    batch_api_dir = data / "batch_api"
    batch_api_dir.mkdir(exist_ok=True)
    batch_api_filepath = batch_api_dir / (output_filename + "_batch_api.jsonl")
    # Reset files
    with open(batch_api_filepath, "w") as f:
        f.write("")
    with open(batch_api_filepath.with_suffix(".hash"), "w") as f:
        f.write("")
    batch_api_custom_id = output_filename
    logger.info(
        f"\nSaving calls to batch API file with custom ID: {batch_api_custom_id}"
    )
else:
    batch_api_filepath, batch_api_custom_id = None, None
output_filename += ".pkl"

# %% [markdown]
# Running all transcription methods
### Make all LLM calls for given dataset

all_modes = []

# %% [markdown]
# ### Helper functions


def extract(
    ocr_text: str,
    prompt: str,
    img_paths: list[Path] | dict[int, Path] = None,
    batch_api_custom_id: Optional[str] = None,
    **kwargs,
) -> tuple[list[str], float]:
    """Returns both the output from a MLLM call and its cost"""
    res = mllm.call(
        user_prompt=ocr_text,
        system_prompt=prompt,
        image_paths=img_paths,
        batch_api_filepath=batch_api_filepath,
        batch_api_custom_id=batch_api_custom_id,
        **(api_call_kwargs | kwargs),
    )

    if "text" in kwargs and kwargs["text"]["format"].get("type") == "json_schema":
        out = json_loads_safe(
            res["response"], return_dummy=res.get("dummy_response", False)
        )
        if out is None:
            raise Exception("Failed to parse JSON response")
        out = {k: str_proc(v) for k, v in out.items()}
    else:
        out = res["response"]

    # Dealing with gemini image token count issue
    if "gem" in mllm.model:
        input_tokens = approximate_gemini_token_counter(
            model=mllm.model,
            urls=img_paths or [],
            prompts=[ocr_text, prompt],
        )
    else:
        input_tokens = {
            "total": res["usage"]["input_tokens"],
            "text": mllm.count_tokens(prompt + ocr_text),
            "image": mllm.get_token_count(image_paths=img_paths),
        }

    output_tokens = res["usage"]["output_tokens"] if "usage" in res else None
    costs = (
        mllm.cost_tracker.calculate_cost(
            input_tokens=input_tokens["total"],
            output_tokens=output_tokens,
            cached_tokens=res["usage"].get("cached_tokens", 0),
        )
        if "usage" in res
        else {"input": 0.0, "output": 0.0, "total": 0.0}
    )

    return {
        "output": out,
        "input_cost": costs["input"],
        "output_cost": costs["output"],
        "total_cost": costs["total"],
        "input_img_tokens": input_tokens["image"],
        "input_txt_tokens": input_tokens["text"],
        "output_tokens": output_tokens,
        "failed_calls": res.get("failed_calls", []),
        "time": res.get("time", 0.0),
        "dummy_response": res.get("dummy_response", False),
    }


def extract_pbp(
    ocr_text: list[str],
    prompt: str,
    img_paths: list[Path] | dict[int, Path] | None = None,
    batch_api_custom_id: Optional[str] = None,
) -> tuple[list[str], float]:
    """Returns both the output from a MLLM call and its cost, processing PAGE-BY-PAGE."""
    output = []
    input_cost, output_cost = 0.0, 0.0
    total_input_img_tokens = 0
    total_input_txt_tokens = 0
    total_output_tokens = 0
    total_failed_calls = []
    total_time = 0.0
    any_dummy_response = False

    for i, page in enumerate(ocr_text):
        batch_api_custom_id = (
            (batch_api_custom_id + f"_page_{i:02d}") if batch_api_custom_id else None
        )
        image_paths = [img_paths[i]] if img_paths else None
        res = mllm.call(
            user_prompt=page,
            system_prompt=prompt,
            image_paths=image_paths,
            batch_api_filepath=batch_api_filepath,
            batch_api_custom_id=batch_api_custom_id,
            **api_call_kwargs,
        )
        output.append(str_proc(res["response"]))

        # Dealing with gemini image token count issue
        if "gem" in mllm.model:
            input_tokens = approximate_gemini_token_counter(
                model=mllm.model,
                urls=image_paths or [],
                prompts=[page, prompt],
            )
        else:
            input_tokens = {
                "total": res["usage"]["input_tokens"],
                "text": mllm.count_tokens(prompt + page),
            }
            input_tokens["image"] = input_tokens["total"] - input_tokens["text"]

        per_page_input_tokens = (input_tokens["image"] or 0.0) + (
            input_tokens["text"] or 0.0
        )
        costs = (
            mllm.cost_tracker.calculate_cost(
                input_tokens=per_page_input_tokens,
                output_tokens=res["usage"]["output_tokens"],
                cached_tokens=res["usage"].get("cached_tokens", 0),
            )
            if "usage" in res
            else 0.0
        )

        input_cost += costs["input"]
        output_cost += costs["output"]
        total_input_img_tokens += input_tokens["image"] or 0.0
        total_input_txt_tokens += input_tokens["text"] or 0.0
        total_output_tokens += res["usage"]["output_tokens"] if "usage" in res else 0
        total_failed_calls.extend(res.get("failed_calls", []))
        total_time += res.get("time", 0.0)
        if res.get("dummy_response", False):
            any_dummy_response = True

    return {
        "output": output,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "input_img_tokens": total_input_img_tokens,
        "input_txt_tokens": total_input_txt_tokens,
        "output_tokens": total_output_tokens,
        "failed_calls": total_failed_calls,
        "time": total_time,
        "dummy_response": any_dummy_response,
    }


def extract_pbp_vision(row: pd.Series) -> tuple[str, float]:
    """Extracts MLLM Vision output PAGE-BY-PAGE and combines across pages."""
    (
        input_cost,
        output_cost,
    ) = (
        0.0,
        0.0,
    )
    input_img_tokens, input_txt_tokens, output_tokens = 0, 0, 0
    total_failed_calls = []
    total_time = 0.0
    any_dummy_response = False
    res = {}

    for i, img_path in row["indexed_img_path"].items():
        run_data = extract(
            ocr_text="",
            prompt=prompts[prompt_name],
            img_paths=[img_path],
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}_page_{i:02d}"
                if batch_api_custom_id
                else None
            ),
        )
        res[i] = str_proc(run_data["output"])

        # Costs
        input_cost += run_data["input_cost"] or 0.0
        output_cost += run_data["output_cost"] or 0.0

        # Tokens
        if input_img_tokens is not None and run_data["input_img_tokens"] is not None:
            input_img_tokens += run_data["input_img_tokens"]
        if input_txt_tokens is not None and run_data["input_txt_tokens"] is not None:
            input_txt_tokens += run_data["input_txt_tokens"]
        if output_tokens is not None and run_data["output_tokens"] is not None:
            output_tokens += run_data["output_tokens"]
        total_failed_calls.extend(run_data.get("failed_calls", []))
        total_time += run_data.get("time", 0.0)
        if run_data.get("dummy_response", False):
            any_dummy_response = True
    return (
        res,
        input_cost,
        output_cost,
        input_img_tokens,
        input_txt_tokens,
        output_tokens,
        total_failed_calls,
        total_time,
        any_dummy_response,
    )


# Helpers for passing run outputs
field_defaults = {
    "input_cost": 0.0,
    "output_cost": 0.0,
    "input_img_tokens": 0,
    "input_txt_tokens": 0,
    "output_tokens": 0,
    "failed_calls": [],
    "time": 0.0,
    "dummy_response": False,
}


def get_fields_per_mode(mode: str) -> list[str]:
    """Get list of expected dataframe fields for a given mode."""
    return ["mode"] + [mode + "_" + field for field in field_defaults.keys()]


def tuple_from_run_data(run_data: dict[str, int | float | str]) -> tuple:
    """Extract tuple of relevant fields from run_data dict."""
    return (
        run_data["output"],
        run_data["input_cost"],
        run_data["output_cost"],
        run_data["input_img_tokens"],
        run_data["input_txt_tokens"],
        run_data["output_tokens"],
        run_data.get("failed_calls", []),
        run_data.get("time", 0.0),
        run_data["dummy_response"],
    )


# %% [markdown]
# Get LLM-based OCR transcriptions. Use pbp

prompt_name = "vision-pbp"
for ocr_engine in ocr_engines:
    if ocr_engine.endswith("_ocr"):
        continue
    mllm.model = ocr_engine
    mode = ocr_engine
    logger.info(f"\n\nProcessing LLM-as-OCR transcription, {mode}...")
    fields = get_fields_per_mode(mode)
    df[fields] = df.progress_apply(
        extract_pbp_vision,
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)

mllm.model = model  # reset

# %% [markdown]
# ## vision-pbp->mllm
# ### Full MLLM on all pages, one at a time, and concatenated back together afterwards (no OCR text)

mode = f"vision-pbp->{mllm.model}"
prompt_name = "vision-pbp"
if "vision-pbp" in prompts:
    logger.info(f"\n\nProcessing {mode}...")
    fields = get_fields_per_mode(mode)
    df[fields] = df.progress_apply(
        extract_pbp_vision,
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)
    mllm.model = model  # reset

else:
    logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")


# %% [markdown]
# ## vision->mllm
# ### Full MLLM end-to-end on all pages at once (no OCR text)


def extract_from_row(row: pd.Series) -> tuple[str, float]:
    """Extract wrapper for a dataframe row."""
    run_data = extract(
        ocr_text="",
        prompt=prompts[prompt_name],
        img_paths=row["indexed_img_path"],
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
        **{
            "text": make_paged_transcription_schema(
                page_ids=list(row["indexed_img_path"].keys())
            )  # specify JSON schema for output
        },
    )
    return tuple_from_run_data(run_data)


prompt_name = "vision"
if prompt_name in prompts:
    mode = f"vision->{mllm.model}"
    logger.info(f"\n\nProcessing {mode}...")
    fields = get_fields_per_mode(mode)
    df[fields] = df.progress_apply(
        extract_from_row,
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)
else:
    logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")


# %% [markdown]
# ## ocr-pbp->mllm
# ### MLLM on raw OCR outputs only, one page at a time


def extract_from_row(row: pd.Series) -> tuple[str, float]:
    """Extract wrapper for a dataframe row."""
    run_data = extract_pbp(
        ocr_text=list(row[ocr_engine].values()),
        prompt=prompt,
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
    )
    return tuple_from_run_data(run_data)


prompt_name = "ocr-pbp"
for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}+pbp->{mllm.model}"
    prompt = prompts.get(prompt_name, None)
    if prompt is None:
        logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")
        continue
    logger.info(f"\n\nProcessing {mode}...")
    df[get_fields_per_mode(mode)] = df.progress_apply(
        extract_from_row,
        axis=1,
        result_type="expand",
    )
    # Convert list output to dict with page IDs as keys
    df[mode] = df.apply(
        lambda row: {str(pid): text for pid, text in zip(row["page_id"], row[mode])},
        axis=1,
    )
    all_modes.append(mode)


# %% [markdown]
# ## ocr+all-pages-pbp->mllm
# ### One page at a time, OCR output *and* image


def extract_from_row(row: pd.Series) -> tuple[str, float]:
    """Extract wrapper for a dataframe row."""
    run_data = extract_pbp(
        ocr_text=list(row[ocr_engine].values()),
        prompt=prompt,
        img_paths=row["img_path"],  # don't need indexing here
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
    )
    return tuple_from_run_data(run_data)


prompt_name = "all-pages-pbp"
for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}+all-pages-pbp->{mllm.model}"
    prompt = prompts.get(prompt_name, None)
    if prompt is None:
        logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")
        continue
    logger.info(f"\n\nProcessing {mode}...")
    df[get_fields_per_mode(mode)] = df.progress_apply(
        extract_from_row,
        axis=1,
        result_type="expand",
    )
    # Convert list output to dict with page IDs as keys
    df[mode] = df.apply(
        lambda row: {str(pid): text for pid, text in zip(row["page_id"], row[mode])},
        axis=1,
    )

    all_modes.append(mode)

# %% [markdown]
# ## ocr+page1->mllm
# ### Giving the entire OCR output as well as just the **first page** image of the doc


def extract_from_row(row: pd.Series) -> tuple[str, float]:
    """Extract wrapper for a dataframe row."""
    run_data = extract(
        ocr_text=make_paged_transcription_json(
            page_ids=row["page_id"], pages=list(row[ocr_engine].values())
        ),
        prompt=prompt,
        img_paths={row["page_id"][0]: row["indexed_img_path"][row["page_id"][0]]},
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
        **{
            "text": make_paged_transcription_schema(
                page_ids=list(row["indexed_img_path"].keys())
            )  # specify JSON schema for output
        },
    )
    return tuple_from_run_data(run_data)


prompt_name = "page1"
for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}+page1->{mllm.model}"
    prompt = prompts.get(prompt_name, None)
    if prompt is None:
        logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")
        continue
    prompt = prompt.get(task, prompt["default"])
    logger.info(f"\n\nProcessing {mode}...")
    df[get_fields_per_mode(mode)] = df.progress_apply(
        extract_from_row,
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)


# %% [markdown]
# ## ocr+pageN->mllm
# ### As +page1, but allow MLLM to *choose* which page to have the image of at based on the OCR output (as opposed to page 1 by default).


mllm.model = "gemma-3-27b-it"  # cheap

PAGE_IDS_FLAG = True


def choose_page_id(
    ocr_text: dict[int, str],
    ocr_engine: str,
    batch_api_custom_id: Optional[str] = None,
) -> tuple[int, float]:
    """Returns a choice of page ID from a MLLM call and the cost of the call."""
    prompt_name = "choose_page_id"
    prompt = prompts[prompt_name]
    page_ids, ocr_text = list(ocr_text.keys()), list(ocr_text.values())
    res = mllm.call(
        user_prompt=make_paged_transcription_json(page_ids=page_ids, pages=ocr_text),
        system_prompt=prompt,
        batch_api_filepath=batch_api_filepath,
        batch_api_custom_id=batch_api_custom_id,
        **(api_call_kwargs | {"text": make_page_selection_schema(page_ids=page_ids)}),
    )
    # Post-process and check page id str->int
    try:
        res["response"] = json_loads_safe(res["response"])
        page_id = int(res["response"]["page_id"])
        reasoning = res["response"]["reasoning"]

    except Exception as e:
        logger.error(
            f"\n\nFailed to parse JSON response:\n{res['response']}\n{e}, defaulting to first page."
        )
        page_id = page_ids[0]
        reasoning = "None (defaulted to first page)"

    if res.get("batch_call", False) == True:
        # Don't run pageN if we don't have page IDs for the prompts
        global PAGE_IDS_FLAG
        PAGE_IDS_FLAG = False
        page_id = -1

    costs = (
        mllm.cost_tracker.calculate_cost(
            input_tokens=res["usage"]["input_tokens"],
            output_tokens=res["usage"]["output_tokens"],
            cached_tokens=res["usage"].get("cached_tokens", 0),
        )
        if "usage" in res
        else 0.0
    )

    run_data = {
        "page_id": page_id,
        "reasoning": reasoning,
        "input_cost": costs["input"],
        "output_cost": costs["output"],
        "total_cost": costs["total"],
        "input_img_tokens": 0,  # just from ocr text
        "input_txt_tokens": res["usage"]["input_tokens"] if "usage" in res else 0,
        "output_tokens": res["usage"]["output_tokens"] if "usage" in res else 0,
        "failed_calls": res.get("failed_calls", []),
        "time": res.get("time", 0.0),
        "dummy_response": res.get("dummy_response", False),
    }

    return run_data


def extract_from_row(row: pd.Series) -> tuple[str, float]:
    run_data = choose_page_id(
        ocr_text=row[ocr_engine],
        ocr_engine=ocr_engine,
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
    )
    return tuple_from_run_data(run_data)


for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}_page_id"
    if "choose_page_id" not in prompts:
        logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")
        continue
    logger.info(f"\n\nProcessing {mode}...")
    df[get_fields_per_mode(mode)] = df.progress_apply(
        extract_from_row,
        axis=1,
        result_type="expand",
    )
    df[mode] = df[mode].astype(int)
mllm.model = model  # reset


def extract_from_row(row: pd.Series) -> tuple[str, float]:
    """Extract wrapper for a dataframe row."""
    run_data = extract(
        ocr_text=make_paged_transcription_json(
            page_ids=row["page_id"], pages=list(row[ocr_engine].values())
        ),
        prompt=prompts[prompt_name],  # + row[f"{ocr_engine}_page_id_reasoning"],
        img_paths={
            (chosen_page_id := row[f"{ocr_engine}_page_id"]): row["indexed_img_path"][
                chosen_page_id
            ]
        },
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_page={chosen_page_id:02d}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
        **{
            "text": make_paged_transcription_schema(
                page_ids=list(row["indexed_img_path"].keys())
            )  # specify JSON schema for output
        },
    )
    return tuple_from_run_data(run_data)


# Give MLLM OCR text and chosen page
if PAGE_IDS_FLAG == False:
    logger.warning(
        "\n\nSkipping pageN as no page IDs were found in the prompt. "
        "(Likely because calls were written to batch file.)"
    )
prompt_name = "pageN"
for ocr_engine in ocr_engines if PAGE_IDS_FLAG else []:
    mode = f"{ocr_engine}+pageN->{mllm.model}"
    if "pageN" not in prompts:
        logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")
        continue
    logger.info(f"\n\nProcessing {mode}...")
    df[get_fields_per_mode(mode)] = df.progress_apply(
        extract_from_row,
        axis=1,
        result_type="expand",
    )
    # Add page_id request cost to transcription cost
    df[mode + "_input_cost"] += df[f"{ocr_engine}_page_id_input_cost"]
    df[mode + "_output_cost"] += df[f"{ocr_engine}_page_id_output_cost"]
    df[mode + "_time"] += df[f"{ocr_engine}_page_id_time"]
    df.drop(
        columns=[
            f"{ocr_engine}_page_id_input_cost",
            f"{ocr_engine}_page_id_output_cost",
            f"{ocr_engine}_page_id_time",
        ],
        inplace=True,
    )
    all_modes.append(mode)


# %% [markdown]
# Page R -- same prompt as pageN, but choosing a RANDOM page instead of via MLLM
# Give MLLM OCR text and chosen page

mode_name = "pageR"
prompt_name = "pageN"

if (
    "bentham" in task
    or "iam_multipage_minpages=02" in task
    or task == "malvern_hills_multipage"
):
    pass
else:
    for ocr_engine in ocr_engines:

        df[f"{ocr_engine}_page_id_pageN"] = df[f"{ocr_engine}_page_id"]  # save original
        df[f"{ocr_engine}_page_id"] = df["page_id"].apply(
            lambda x: random.choice(x)
        )  # set random page

        mode = f"{ocr_engine}+{mode_name}->{mllm.model}"
        if prompt_name not in prompts:
            logger.warning(f"\n\nNo prompt found for {mode}. Skipping this mode.")
            continue
        logger.info(f"\n\nProcessing {mode}...")
        df[get_fields_per_mode(mode)] = df.progress_apply(
            extract_from_row,
            axis=1,
            result_type="expand",
        )
        all_modes.append(mode)


mllm.update_cache()

logger.info(f"\n\nCompleted all modes: {all_modes}!")

# %% [markdown]
# ### Aggregating cost sources

# Standard OCR
for ocr_engine in ocr_engines:
    if not ocr_engine.endswith("_ocr"):
        continue
    for field, default_value in field_defaults.items():
        df[f"{ocr_engine}_{field}"] = default_value
    # OCR costs
    df[ocr_engine + "_ocr_cost"] = df[ocr_engine].apply(
        lambda x: len(x) * ocr_costs[ocr_engine]
    )  # fixed cost per page

    for mode in all_modes:  # MLLM costs
        if ocr_engine in mode:
            df[mode + "_ocr_cost"] = df[ocr_engine + "_ocr_cost"]
        else:
            df[mode + "_ocr_cost"] = 0.0
    logger.info(f"\nAdded OCR costs for {ocr_engine}.")


# %% [markdown]
# ### Save outputs of experiments for eval in a separate notebook

df.to_pickle(results_dir / output_filename)
logger.info(f"\nSaved outputs to {results_dir / output_filename}")

mllm.update_cache()

# %% [markdown]
# mllm.backup_cache()

# %%
