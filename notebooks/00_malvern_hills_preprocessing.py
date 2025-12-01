# %% [markdown]
# # Converts Malvern-Hills dataset into pandas dataframes and save for downstream experiment notebooks.

# %% [markdown]
# ### Setup
from judge_htr import data
from loguru import logger
from judge_htr.ocr_wrappers import azure_img2txt
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import pickle
import matplotlib.pyplot as plt
import time
from datetime import datetime
from judge_htr.preprocessing import collapse_whitespace_and_newlines


load_dotenv()
tqdm.pandas()

datetime_str = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

data_dir = data / "malvern_hills_trust"

ocr_engines = [
    "azure_ocr",
]
non_mllm_methods = ["trocr_ocr", "docowl2_ocr", "docowl2_lines_ocr"]

# multipage_doc_filename = "_files_per_multipage_doc.txt"
# output_filename = "malvern_hills_multipage.pkl"

multipage_doc_filename = "_files_per_multipage_min_5p_doc.txt"
output_filename = "malvern_hills_multipage_min_5p.pkl"

# %% [markdown]
# Load data into dataframe
data = []
for doc_path in sorted(data_dir.iterdir()):
    if not doc_path.is_dir():
        continue
    doc_id = doc_path.stem
    logger.info(f"Processing doc {doc_id} at {doc_path}...")
    multipage_doc_file = doc_path / multipage_doc_filename
    with open(multipage_doc_file, "r") as f:
        lines = f.readlines()

    # Get years of writing
    years = [
        y.stem.replace("year_", "")
        for y in doc_path.iterdir()
        if y.stem.startswith("year_")
    ]
    if len(years) > 1:
        print(years)
    years_map = {}
    for year in sorted(years, key=len):
        if year.startswith("original_"):
            years_map["original"] = int(year.replace("original_", ""))
        elif len(year.split("_")) == 2:
            page_ids = year.split("_")[1].split("-")
            years_map.update(
                {
                    int(pid): int(year.split("_")[0])
                    for pid in range(int(page_ids[0]), int(page_ids[1]) + 1)
                }
            )
            years_map.update(
                {
                    f"{int(pid)}_original": int(year.split("_")[0])
                    for pid in range(int(page_ids[0]), int(page_ids[1]) + 1)
                }
            )
        elif year == "":
            assert not None in years_map
            years_map[None] = None
        else:
            assert not None in years_map
            years_map[None] = int(year)
        if "original" not in years_map:
            years_map["original"] = years_map[None]

    # Get metadata
    metadata_files = [
        f for f in sorted(doc_path.iterdir()) if f.stem.startswith("metadata")
    ]

    metadata = pd.concat(
        [pd.read_csv(meta_f, index_col=0) for meta_f in metadata_files], axis=1
    )

    # Get author IDs
    author_id_file = doc_path / "author_ids.csv"
    author_ids = pd.read_csv(author_id_file, index_col=0, dtype=str)
    author_ids["author_ids"] = author_ids.apply(
        lambda row: [str(row[c]) for c in author_ids.columns if pd.notna(row[c])],
        axis=1,
    )

    page_ids_to_multipage_doc_id = {}
    for i, line in enumerate(lines):
        page_ids = line.strip().split()
        for page_id in page_ids:
            page_ids_to_multipage_doc_id[page_id] = i

    for page_path in tqdm(sorted(doc_path.iterdir())):
        if not page_path.is_file() or page_path.suffix != ".jpeg":
            continue
        page_id = page_path.stem
        gt_path = doc_path / f"{page_id}_gt.txt"
        # Load gt str from file
        with open(gt_path, "r") as f:
            gt_str = f.read()

        # Format str: strip leading/trailing whitespace from each line, collapse multiple spaces into one, collapse multiple newlines into one
        gt_str = collapse_whitespace_and_newlines(gt_str)

        if not gt_str:
            print(gt_str)

        # Append dict to data list
        if page_id not in page_ids_to_multipage_doc_id:
            logger.warning(
                f"Page ID {page_id} not found in {multipage_doc_file}, skipping..."
            )
            continue
        multipage_doc_id = f"{doc_id}_{page_ids_to_multipage_doc_id[page_id]:02d}"
        written_year = years_map.get(int(page_id), years_map.get(None, "unknown"))
        original_year = years_map.get(
            f"{int(page_id)}_original", years_map.get("original", "unknown")
        )
        data.append(
            {
                "id": f"{doc_id}_{page_id}",
                "doc_id": doc_id,
                "page_id": int(page_id),
                "multipage_doc_id": multipage_doc_id,
                "img_path": page_path,
                "gt": gt_str,
                "written_year": written_year,
                "original_year": original_year,
                "author_ids": author_ids.loc[str(page_path), "author_ids"],
            }
            | metadata.T.to_dict()[str(page_path)]
        )
# Make unique ID from doc_id and page_id and set as index
df = pd.DataFrame(data).set_index("id")

# Check docs with mulitple authors match their author IDs
author_id_counts = {}
for id, row in df.iterrows():
    if row["multiple_hands"]:
        assert (
            len(row["author_ids"]) > 1
        ), f"Doc {id} marked as multiple authors but has author_ids {row['author_ids']}"
    else:
        assert (
            len(row["author_ids"]) == 1
        ), f"Doc {id} marked as single author but has author_ids {row['author_ids']}"
    for author_id in row["author_ids"]:
        author_id_counts[author_id] = author_id_counts.get(author_id, 0) + 1

print(f"Author ID counts:\n{author_id_counts}")

# %% [markdown]
# Add OCR transcriptions

# Get OCR transcriptions from images
ocr_outputs = {engine: {} for engine in ocr_engines}

ocr_to_text_funcs = {
    "azure_ocr": azure_img2txt,
}

# Looping allows recovery if interrupted
try:
    for engine in ocr_engines:
        logger.info(f"\nRunning {engine} OCR on images...")
        saved_path = data_dir / f"ocr_{engine}.pkl"
        if saved_path.exists():
            with open(saved_path, "rb") as f:
                ocr_outputs[engine] = pickle.load(f)
            logger.info(
                f"Loaded {len(ocr_outputs[engine])} doc OCR outputs from {saved_path}"
            )
            if len(ocr_outputs[engine]) == len(df):
                logger.info(f"All {engine} OCR outputs already exist, skipping...")
                continue
        for id, row in tqdm(df.iterrows(), total=len(df)):
            if id in ocr_outputs[engine]:
                continue
            text, bboxes = ocr_to_text_funcs[engine](row["img_path"])
            ocr_outputs[engine][id] = {
                "text": text,
                "bboxes": bboxes,
            }
        # Pickle the dict just in case
        with open(saved_path, "wb") as f:
            pickle.dump(ocr_outputs[engine], f)
        logger.info(f"Saved {engine} OCR outputs to {saved_path}")
except Exception:
    logger.exception("OCR processing interrupted, saving progress...")
    for engine in ocr_engines:
        saved_path = data_dir / f"ocr_{engine}.pkl"
        with open(saved_path, "wb") as f:
            pickle.dump(ocr_outputs[engine], f)
        logger.info(f"Saved {engine} OCR outputs to {saved_path}")
    raise

# Add dfs to main df
# FIx below so apply works when id is the index, not a col:
for engine in ocr_engines:
    df[engine] = df.index.map(lambda x: ocr_outputs[engine][x]["text"])


# %% [markdown]
# Add TrOCR results
trocr_ocr_col, trocr_time_col = [], []
for i, row in df.iterrows():
    img_path = row["img_path"]
    doc_dir = (
        img_path.parent / "lines" / img_path.stem / "trocr-large-handwritten_output.txt"
    )
    with open(doc_dir, "r") as f:
        trocr_text = f.read()
    trocr_ocr_col.append(trocr_text)
    # Get time
    trocr_time = None
    for file in doc_dir.parent.iterdir():
        if file.stem.startswith(
            "trocr-large-handwritten_output_"
        ) and file.stem.endswith("ms"):
            trocr_time = int(str(file).split("_")[-1][:-2])
            break
    if not trocr_time:
        raise ValueError(f"TrOCR time file not found for {doc_dir}")
    trocr_time_col.append(trocr_time / 1000)  # seconds

df["trocr_ocr"] = trocr_ocr_col
df["trocr_ocr_time"] = trocr_time_col

# %% [markdown]
# Load docowl 2 results
from judge_htr import results

docowl2_results_path = results / "malvern_hills_trust_docowl2_outputs"

from pathlib import Path

docowl2_col, docowl2_time_col = [], []
for i, row in df.iterrows():
    img_path = row["img_path"]
    docowl_txt_path = Path(
        str(img_path)
        .replace("malvern_hills_trust", "malvern_hills_trust_docowl2_outputs")
        .replace("data/", "results/")
    ).with_suffix(".txt")
    docowl2_text = docowl_txt_path.read_text(encoding="utf-8")
    docowl2_col.append(docowl2_text)
    # Get time
    docowl2_time = None
    for file in docowl_txt_path.parent.iterdir():
        if file.stem.startswith(docowl_txt_path.stem + "_") and file.stem.endswith(
            "ms"
        ):
            docowl2_time = int(str(file).split("_t=")[-1][:-2])
            break
    if not docowl2_time:
        raise ValueError(f"Docowl2 time file not found for {docowl_txt_path}")
    docowl2_time_col.append(docowl2_time / 1000)  # seconds

df["docowl2_ocr"] = docowl2_col
df["docowl2_ocr_time"] = docowl2_time_col


# %% [markdown]
# Load DocOwl 2 line-level results
docowl2_col, docowl2_time_col = [], []
for i, row in df.iterrows():
    img_path = row["img_path"]
    lines_dir = Path(
        str(img_path)
        .replace("malvern_hills_trust", "malvern_hills_trust_docowl2_outputs")
        .replace("data/", "results/")
        .replace(".jpeg", "")  # dir
        .replace(row["doc_id"], f"{row['doc_id']}/lines")
    )
    page_text, page_time = "", 0
    for line_txt_path in sorted(lines_dir.glob("*.txt")):
        line_text = line_txt_path.read_text(encoding="utf-8")
        page_text += line_text + "\n"
    for time_file in lines_dir.iterdir():
        if not time_file.stem.endswith("ms"):
            continue
        time_ms = int(str(time_file).split("_t=")[-1][:-2])
        page_time += time_ms
    docowl2_col.append(page_text.strip())
    docowl2_time_col.append(page_time / 1000)  # seconds


df["docowl2_lines_ocr"] = docowl2_col
df["docowl2_lines_ocr_time"] = docowl2_time_col

import math


def count(x: list[float]):
    res = {}
    for i in x:
        if math.isnan(i):
            res[None] = res.get(None, 0) + 1
        else:
            res[int(i)] = res.get(i, 0) + 1
    return res


# %% Make multi-page doc df
df_mp = df.groupby("multipage_doc_id").agg(
    {
        "doc_id": "first",
        "page_id": lambda x: list(x),
        "img_path": lambda x: list(x),
        "gt": lambda x: list(x),
        "written_year": count,
        "original_year": count,
        "tabular": any,
        "margin_notes": any,
        "distractors": any,
        "non_standard_structure": any,
        "archaic": any,
        "poor_quality": any,
        "multiple_hands": any,
        "crossings_out": any,
    }
    | {engine: lambda x: list(x) for engine in ocr_engines + non_mllm_methods}
    | {f"{method}_time": lambda x: sum(x) for method in non_mllm_methods}
)


# %% [markdown]
# Pickle df
df_path = data_dir / output_filename
df_mp.to_pickle(df_path)
logger.info(f"Saved df to {df_path}")


# %%
# Stats
df["word_count"] = df["gt"].apply(lambda x: len(x.split()))
df["char_count"] = df["gt"].apply(lambda x: len(x))
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].hist(df["word_count"], bins=10)
axs[0].set_title("Page Lengths in Words")
axs[1].hist(df["char_count"], bins=10)
axs[1].set_title("Page Lengths in Characters")
axs[2].bar(
    df["doc_id"].value_counts().index.sort_values(),
    df["doc_id"].value_counts().sort_index(),
)
axs[2].set_title("Number of Pages per Document")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
# plt.show()

# %%
# Metadata
# word count, char count, tabular, margin_notes, distractors, non_standard_structure, archaic, poor_quality, multiple_hands, crossings_out
# mean for all of these
metadata_stats = df.agg(
    {
        "word_count": ["mean", "std", "min", "max"],
        "char_count": ["mean", "std", "min", "max"],
        "tabular": "mean",
        "margin_notes": "mean",
        "distractors": "mean",
        "non_standard_structure": "mean",
        "archaic": "mean",
        "poor_quality": "mean",
        "multiple_hands": "mean",
        "crossings_out": "mean",
    }
)
ms = metadata_stats.loc[["mean", "std"]]

row = {
    "word_count": f"{int(round(ms.at['mean', 'word_count']))} ± {int(round(ms.at['std', 'word_count']))}",
    "char_count": f"{int(round(ms.at['mean', 'char_count']))} ± {int(round(ms.at['std', 'char_count']))}",
}

# all other columns → percentage strings with 1 dp (from the mean row)
for c in [c for c in ms.columns if c not in ["word_count", "char_count"]]:
    row[c] = f"{ms.at['mean', c] * 100:.1f}%"

formatted_df = pd.DataFrame([row])

formatted_df

# %%
# Pages by year
results = df["written_year"].value_counts(dropna=False).sort_index()
results

written_year_counts = results.to_dict()

# %%
# And original year
results = df["original_year"].value_counts(dropna=False).sort_index()
results

original_year_counts = results.to_dict()

# %%
# Re-indexed author ID counts table
author_id_counts_df = (
    pd.DataFrame.from_dict(author_id_counts, orient="index", columns=["Count"])
    .sort_values(by="Count", ascending=False)
    .reset_index(drop=True)
)
author_id_counts_df.index.name = "Author ID"
author_id_counts_df


# %% [markdown]

doc_year_counts = pd.DataFrame.from_dict(
    original_year_counts, orient="index", columns=["Original"]
)
doc_year_counts["Written"] = pd.DataFrame.from_dict(
    written_year_counts, orient="index", columns=["Written"]
)

doc_year_counts = doc_year_counts.fillna(0).astype(int).sort_index()
# index to ints, nan to "Unknown"
doc_year_counts.index = doc_year_counts.index.map(
    lambda x: int(x) if pd.notna(x) else "Unknown"
)
doc_year_counts

# %%

df_doc_types = df[["doc_id", "tabular"]]


# %%
import yaml
import pandas as pd

# ==== Set this to your YAML path ====
YAML_PATH = data_dir / "image_types.yaml"

# ---- Load YAML ----
with open(YAML_PATH, "r", encoding="utf-8") as f:
    raw_map = yaml.safe_load(f) or {}

# ---- Build lookup maps ----
doc_level_map: dict[str, str] = {}
page_level_map: dict[tuple[str, str], str] = {}

for doc_key, val in raw_map.items():
    if isinstance(val, dict):  # page-specific classifications
        for page_key, type_str in val.items():
            if type_str is None:
                continue
            # Normalize page index to zero-padded 3 chars if it looks numeric
            page_norm = str(page_key).strip()
            if page_norm.isdigit():
                page_norm = page_norm.zfill(3)
            page_level_map[(str(doc_key), page_norm)] = str(type_str).strip()
    elif val is None:
        continue
    else:  # whole-document classification
        doc_level_map[str(doc_key)] = str(val).strip()


# ---- Helper: parse "<primary> -> <sec1 & sec2 & ...>" into components ----
def parse_type_string(s: str) -> tuple[str, list[str]]:
    s = s.strip()
    if "->" not in s:
        return s, []
    left, right = s.split("->", 1)
    primary = left.strip()
    right = right.strip()
    if not right:
        return primary, []
    # Split on '&' into multiple, trimming whitespace
    secs = [p.strip() for p in right.split("&") if p.strip()]
    return primary, secs


# ---- Resolve classification per row ----
def resolve_row(row) -> tuple[str | None, list[str]]:
    doc_id = str(row["doc_id"])
    row_id = str(row.name)
    # Extract page suffix if present: e.g., "1889-05--1909-10_006" -> "006"
    page_suffix = None
    if "_" in row_id:
        page_suffix = row_id.rsplit("_", 1)[-1].strip()
        if page_suffix.isdigit():
            page_suffix = page_suffix.zfill(3)

    # Prefer page-specific classification if available
    type_str = None
    if page_suffix is not None and (doc_id, page_suffix) in page_level_map:
        type_str = page_level_map[(doc_id, page_suffix)]
    elif doc_id in doc_level_map:
        type_str = doc_level_map[doc_id]

    # If nothing found, return empty defaults
    if not type_str:
        primary, secondary = None, []
    else:
        primary, secondary = parse_type_string(type_str)

    # If tabular True, append "tabular" to secondary types
    try:
        if bool(row.get("tabular", False)):
            if "tabular" not in secondary:
                secondary = [*secondary, "tabular"]
    except Exception:
        # If tabular column not present or non-boolean, ignore
        pass

    return primary, secondary


# ---- Apply to dataframe ----
primary_types = []
secondary_types = []

for _, r in df_doc_types.iterrows():
    p, s = resolve_row(r)
    primary_types.append(p)
    secondary_types.append(s)

df_doc_types["primary_type"] = primary_types
df_doc_types["secondary_types"] = secondary_types

df_doc_types.to_pickle(data_dir / "malvern_hills_image_types.pkl")

# %% [markdown]
