#
import pandas as pd
from tqdm import tqdm
from judge_htr import data, results
from judge_htr.laia_inference import LAIAInference
from time import time
from pathlib import Path

laia_dir = data / "laia_rerun"
laia_dir.mkdir(exist_ok=True)

df = pd.read_pickle(results / "laia_iam_multipage_minpages=02_split=0.20_seed=00.pkl")

laia_model = LAIAInference(model_spec="Teklia/pylaia-iam")

laia_col, laia_col_times = [], []
for i, row in tqdm(df.iterrows(), total=len(df)):
    page_ids, line_img_paths = row["page_id"], row["line_img_paths"]
    # Call laia_inference.py on each line image
    res, time_per_doc = [], []
    for page_id, page in tqdm(zip(page_ids, line_img_paths)):
        output_page_file = laia_dir / f"{i:03d}_{page_id:02d}.txt"
        if output_page_file.exists():
            print(f"OCR output exists: {output_page_file}")
            page_txt = output_page_file.read_text(encoding="utf-8")
            res.append(page_txt)
            # Get time file
            page_time = None
            for file in output_page_file.parent.iterdir():
                if file.stem.startswith(
                    output_page_file.stem + "_"
                ) and file.stem.endswith("ms"):
                    page_time = int(str(file).split("_")[-1][:-2])
                    break
            if not page_time:
                raise ValueError
            time_per_doc.append(page_time)
            continue
        txt_lines, time_per_line = [], []
        for line_img_path in page:
            line_txt, t = laia_model.run(
                image_path=line_img_path,
            )
            print("\n")
            print(line_txt)
            txt_lines.append(line_txt)
            time_per_line.append(t)
        page_txt = "\n".join(txt_lines)
        page_time = sum(time_per_line)
        # write to a file with name i to 3sf
        output_page_file.write_text(page_txt, encoding="utf-8")
        Path(str(output_page_file).replace(".txt", f"_{int(page_time*1000)}ms")).touch()
        res.append(page_txt)
        time_per_doc.append(page_time)
    laia_col.append(res)
    laia_col_times.append(sum(time_per_doc))
df["laia_ocr"] = laia_col
df["laia_ocr_time"] = laia_col_times
breakpoint()
# turn into dict mapping page ID to col contents
df["laia_ocr"] = df.apply(
    lambda row: dict(zip(row["page_id"], row["laia_ocr"])),
    axis=1,
)

for v in [
    "input_cost",
    "output_cost",
    "input_img_tokens",
    "output_img_tokens",
    "input_txt_tokens",
    "output_tokens",
    "ocr_cost",
]:
    df[f"laia_ocr_{v}"] = 0

df.to_pickle(results / "laia_iam_multipage_minpages=02_split=0.20_seed=00_with_ocr.pkl")
