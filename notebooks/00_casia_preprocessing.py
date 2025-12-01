# %% [markdown]
# # Converts CASIA HWDB (Chinese Academy of Sciences Institute of Automation) dataset into pandas dataframes and save for downstream experiment notebooks.

# %% [markdown]
# ### Setup
import sys
from judge_htr import data
from loguru import logger
from judge_htr.ocr_wrappers import (
    azure_img2txt,
    googleocr_img2txt,
    tesseract_img2txt,
    textract_img2txt,
)
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import pickle
import matplotlib.pyplot as plt
import time
from datetime import datetime
from judge_htr.preprocessing import collapse_whitespace_and_newlines
from judge_htr.preproc_casia import save_page_and_lines


load_dotenv()
tqdm.pandas()

datetime_str = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

data_dir = data / "CASIA"
raw_data_dir = data_dir / "HWDB2.1Test"
processed_data_dir = data_dir / "processed"
processed_data_dir.mkdir(parents=True, exist_ok=True)

ocr_engines = [
    "google_ocr",
]

multipage_doc_filename = "_files_per_multipage_doc.txt"
output_filename = "casia_multipage.pkl"

# %% [markdown]
# Convert CASIA .dgrl files to images and line data
for filepath in tqdm(sorted(raw_data_dir.iterdir())):
    if filepath.suffix == ".dgrl":
        if (processed_data_dir / (filepath.stem + ".png")).exists():
            continue
        save_page_and_lines(
            dgrl_path=filepath,
            out_dir=processed_data_dir,
        )

data = []
for img_path in tqdm(sorted(processed_data_dir.iterdir())):
    if img_path.suffix != ".png":
        continue
    id = img_path.stem.split("_")[0]
    author_id, page_id = int(id.split("-")[0]), int(id.split("-P")[1])
    lines_dir = processed_data_dir / id
    gt = ""
    for line_idx in sorted(lines_dir.iterdir()):
        if line_idx.suffix == ".txt":
            with open(line_idx, "r", encoding="utf-8") as f:
                gt += f.read() + "\n"
    data.append(
        {
            "id": id,
            "author_id": author_id,
            "page_id": page_id,
            "img_path": img_path,
            "gt": gt.strip(),
        }
    )
# Make unique ID from doc_id and page_id and set as index
df = pd.DataFrame(data).set_index("id")


# %% [markdown]
# Add OCR transcriptions

# Get OCR transcriptions from images
ocr_outputs = {engine: {} for engine in ocr_engines}

tesseract_fn = (
    tesseract_img2txt
    if "CASIA" not in str(data_dir)
    else lambda x: tesseract_img2txt(x, lang="chi_sim")
)

ocr_to_text_funcs = {
    # "azure_ocr": azure_img2txt,
    "google_ocr": googleocr_img2txt,
    "tesseract_ocr": tesseract_fn,
    "textract": textract_img2txt,
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
except Exception as e:
    logger.error(f"Exception during OCR processing: {e}")
    logger.exception("OCR processing interrupted, saving progress...")
    for engine in ocr_engines:
        saved_path = data_dir / f"ocr_{engine}.pkl"
        with open(saved_path, "wb") as f:
            pickle.dump(ocr_outputs[engine], f)
        logger.info(f"Saved {engine} OCR outputs to {saved_path}")
    raise

for engine in ocr_engines:
    df[engine] = df.index.map(lambda x: ocr_outputs[engine][x]["text"])


# %% Make multi-page doc df
df_mp = df.groupby("author_id").agg(
    {
        "page_id": lambda x: list(x),
        "img_path": lambda x: list(x),
        "gt": lambda x: list(x),
    }
    | {engine: lambda x: list(x) for engine in ocr_engines}
)
# Rename index from author_id to multipage_doc_id
df_mp.index.name = "multipage_doc_id"

if "casia" in output_filename.lower():
    # All the same, unlabelled, no reason not to start from 0
    df_mp["original_page_id"] = df_mp["page_id"]
    df_mp["page_id"] = [list(range(5)) for _ in range(len(df_mp))]

# %% [markdown]
# Pickle df
df_path = data_dir / output_filename
df_mp.to_pickle(df_path)
logger.info(f"Saved df to {df_path}")

if "casia" in output_filename.lower():
    sys.exit(0)


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
