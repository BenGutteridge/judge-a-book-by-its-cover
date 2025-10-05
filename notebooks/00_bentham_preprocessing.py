# %% [markdown]
# # Convert Bentham dataset into pandas dataframes for downstream experiments notebooks

# %% [markdown]
# ### Setup
from judge_htr import data
from loguru import logger
from judge_htr.ocr_wrappers import azure_img2txt
from judge_htr.preprocessing import combine_gt_txt_files, txt2str
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import pickle

load_dotenv()
tqdm.pandas()

data_dir = data / "bentham"

ocr_engines = [
    "azure_ocr",
]

# %% [markdown]
# Pre-process data
raw_transcriptions_dir = data_dir / "BenthamDatasetR0-GT/Transcriptions"
transcriptions_dir = data_dir / "gt"
pages_dir = data_dir / "BenthamDatasetR0-Images" / "Images" / "Pages"

combine_gt_txt_files(
    input_dir=raw_transcriptions_dir,
    output_dir=transcriptions_dir,
)


# %% [markdown]
# Make a df out of the image paths and the raw ground-truth text
df_page = pd.DataFrame(
    list({x.stem: x for x in pages_dir.glob("*.jpg")}.items()),
    columns=["id", "img_path"],
)

df_page_gt = pd.DataFrame(
    list({x.stem: txt2str(x) for x in transcriptions_dir.glob("*.txt")}.items()),
    columns=["id", "gt"],
)
df_page_gt["gt"] = df_page_gt["gt"].apply(lambda s: s.replace("=\n : ", "-\n"))

df = df_page_gt.merge(df_page, on="id", how="left")
assert len(df_page_gt) == len(df_page) and df.isna().sum().sum() == 0
del df_page_gt, df_page

# df["id"] is in the format '000_000_000' - split into id1 id2 id3 columns
df[["id1", "id2", "id3"]] = df["id"].str.split("_", expand=True)

# %% [markdown]
# Get OCR transcriptions from images

ocr_outputs = {engine: {} for engine in ocr_engines}

# # Identical to process below but nonrecoverable if interrupted
# df["tesseract"] = df["img_path"].progress_apply(tesseract_img2txt)
# df["google_ocr"] = df["img_path"].progress_apply(googleocr_img2txt)
# df["azure_ocr"] = df["img_path"].progress_apply(azure_img2txt)

# ALT: looping allows recovery if interrupted
ocr_to_text_funcs = {
    "azure_ocr": azure_img2txt,
}

for engine in ocr_engines:
    logger.info(f"\nRunning {engine} OCR on images...")
    saved_path = data_dir / f"ocr_{engine}.pkl"
    if saved_path.exists():
        with open(saved_path, "rb") as f:
            ocr_outputs[engine] = pickle.load(f)
        continue
    for i, row in tqdm(df.iterrows(), total=len(df)):
        ocr_outputs[engine][row["id"]] = ocr_to_text_funcs[engine](row["img_path"])
    # Pickle the dict just in case
    with open(data_dir / f"{engine}.pkl", "wb") as f:
        pickle.dump(ocr_outputs[engine], f)

# Add dfs to main df
for engine in ocr_engines:
    df[engine] = df["id"].apply(lambda x: ocr_outputs[engine][x])

# %% [markdown]
# Save the df
df_filepath = data_dir / "bentham_page_df.pkl"
df.to_pickle(df_filepath)
logger.info(f"\nSaved df to {df_filepath}")

# %% [markdown]
# # Multi-page mode
#
# We have per-page OCR transcriptions, but we want to work with multi-page documents.
# So lets construct some by combining across pages, grouping by writer_id.
#
# Having looked at the UCL file naming convention,
# the first 3 digits indicate the 'box', and seem to share similar handwriting/paper/pen style,
# and perhaps some similarity in content.
# The next 3 digits indicate some split within the box —
# possibly a single document (consisting of one or more pages)
# The final 3 digits are a page number.
# Most of the documents in this dataset are not multi-page, and not consecutive.

# How many are of each id1?
logger.info("\nHow many are of each id1?\n" f"Per id1:\n{df['id1'].value_counts()}")

# How many are of each id1-id2 pairing?
logger.info(
    "\nHow many are of each id1-id2 pairing?"
    f"Per id1-id2:\n{(df['id1'] + '_' + df['id2']).value_counts()}"
)

# %% [markdown]
# How many multi-page docs are there?
# Either strictly consecutive or just the same source
df["id3"] = df["id3"].astype(int)
df = df.sort_values(by=["id1", "id2", "id3"]).reset_index(drop=True)
agg_dict = {"id3_list": ("id3", list)} | {
    col: (col, list) for col in df.columns if "id" not in col
}
df_multipage = df.groupby(["id1", "id2"]).agg(**agg_dict).reset_index()

is_consecutive = lambda nums: (
    len(nums) == len(set(nums)) and max(nums) - min(nums) == len(nums) - 1
)
df_multipage = df_multipage[
    df_multipage["id3_list"].apply(lambda x: len(x) > 1)
].reset_index(drop=True)

df_multipage_consecutive = df_multipage[df_multipage["id3_list"].apply(is_consecutive)]

logger.info(
    "\nHow many multipage docs are there, and how long are they?\n"
    f"{df_multipage['id3_list'].apply(len).value_counts()}"
)
logger.info(
    "\nHow many multipage *consecutive* docs are there, and how long are they?\n"
    f"{df_multipage_consecutive['id3_list'].apply(len).value_counts()}"
)

# %% [markdown]
# Save the multi-page dfs
df_multipage["page_id"] = df_multipage["id3_list"]
df_multipage_consecutive["page_id"] = df_multipage_consecutive["id3_list"]

df_multipage.to_pickle(data_dir / "bentham_multipage.pkl")
df_multipage_consecutive.to_pickle(data_dir / "bentham_multipage_consecutive.pkl")

# %% [markdown]
