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

# %% [markdown]
# Load data into dataframe
data = []
for doc_path in sorted(data_dir.iterdir()):
    if not doc_path.is_dir():
        continue
    doc_id = doc_path.stem
    logger.info(f"Processing doc {doc_id} at {doc_path}...")
    multipage_doc_file = doc_path / "_files_per_multipage_doc.txt"
    with open(multipage_doc_file, "r") as f:
        lines = f.readlines()
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

        # Append dict to data list
        multipage_doc_id = f"{doc_id}_{page_ids_to_multipage_doc_id[page_id]:02d}"
        data.append(
            {
                "id": f"{doc_id}_{page_id}",
                "doc_id": doc_id,
                "page_id": int(page_id),
                "multipage_doc_id": multipage_doc_id,
                "img_path": page_path,
                "gt": gt_str,
            }
        )
# Make unique ID from doc_id and page_id and set as index
df = pd.DataFrame(data).set_index("id")


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


# %% Make multi-page doc df
df_mp = df.groupby("multipage_doc_id").agg(
    {
        "doc_id": "first",
        "page_id": lambda x: list(x),
        "img_path": lambda x: list(x),
        "gt": lambda x: list(x),
    }
    | {engine: lambda x: list(x) for engine in ocr_engines}
)


# %% [markdown]
# Pickle df
df_path = data_dir / f"malvern_hills_multipage.pkl"
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
