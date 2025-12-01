# %% [markdown]
# Script for generating per-line images from whole-page document images.
# Use upstream with non-MLLM models (TrOCR, PyLaia, DocOwl2), see 01_{model}_experiments.py notebooks in `notebooks/additional_models/`

# 1. write json files for every image with kraken
# 2. crop existing images and save line images with crop_lines.py
# 3. notebook running trocr large handwritten on each line
# 4. recombine and reorganise to work with our eval setup

# N.B. run this notebook inside a separate kraken env; use
# `source ~/.venvs/kraken-lines/bin/activate`

from judge_htr import data
from judge_htr.crop_lines import crop_lines
from tqdm import tqdm
import os

# %% [markdown]
# # 1. Get line info with kraken
# Use `kraken -i <IMG_PATH> <JSON_OUTPUT_PATH> segment -bl`

data_dir = data / "malvern_hills_trust"

lines_dirs = []
for root in sorted(data_dir.iterdir()):
    if not root.is_dir():
        continue

    # Make lines subdir
    print(f"Processing dir: {root}")
    lines_dir = root / "lines"
    lines_dir.mkdir(exist_ok=True)
    lines_dirs.append(lines_dir)

    for img_path in tqdm(sorted(root.glob("*.jpeg"))):
        json_path = lines_dir / f"{img_path.stem}.json"
        if json_path.exists():
            print(f" - JSON exists: {json_path}")
            continue

        cmd = f"kraken -i {img_path} {json_path} segment -bl"
        os.system(cmd)

# %% [markdown]
# # 2. Crop line images
# Use `python crop_lines.py /path/to/page.jpg /path/to/page_lines.json /path/to/out_dir --pad 6 --prefix line_ --ext png`
for line_dir in lines_dirs:
    print(f"Cropping lines in dir: {line_dir}")
    for json_path in tqdm(sorted(line_dir.glob("*.json"))):
        page_id = json_path.stem
        out_dir = line_dir / page_id
        if out_dir.exists():
            print(f" - output dir exists: {out_dir}")
            continue
        out_dir.mkdir(exist_ok=True)
        img_path = str(json_path.with_suffix(".jpeg")).replace("lines/", "")

        # Crop images and generate new cropped image files
        crop_lines(
            img_path=img_path, json_path=json_path, out_dir=out_dir, pad=0, ext="png"
        )
