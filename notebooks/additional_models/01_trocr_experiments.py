from judge_htr import data, results
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm import tqdm
import time
from pathlib import Path

model_name = "trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(f"microsoft/{model_name}")
model = VisionEncoderDecoderModel.from_pretrained(f"microsoft/{model_name}")


data_dir = data / "malvern_hills_trust"
output_dir = results / "additional_model_outputs" / "malvern_hills_trust_trocr"

for subdir in [
    "1889-05--1909-10",
    "1910-01--1919-11",
    "1923-10--1928-02",
    "1931-11--1935-11",
    "1936-01--1938-03",
    "202505-000",
    "202505-001",
    "202505-002",
    "202508-000",
    "202508-001",
]:
    lines_dir = data_dir / subdir / "lines"
    for lines_path in tqdm(sorted(lines_dir.iterdir())):
        if not lines_path.is_dir():
            continue

        # Create a txt file and write each line to it
        output_lines_path = Path(
            str(lines_path).replace(str(data_dir), str(output_dir))
        )

        txt_file_path = output_lines_path / f"{model_name}_output.txt"
        time_file_path = output_lines_path / f"{model_name}_output"
        if txt_file_path.exists():
            print(f"OCR output exists: {txt_file_path}")
            continue
        print(f"Processing {lines_path}")
        t0 = time.time()
        with open(txt_file_path, "w") as txt_file:
            for line_img_path in tqdm(sorted(lines_path.glob("*.png"))):
                image = Image.open(line_img_path).convert("RGB")

                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                txt_file.write(generated_text + "\n")
        t = time.time() - t0
        # write time to a file
        time_file_path = lines_path / (time_file_path.stem + f"_{int(t*1000)}ms")
        time_file_path.touch()

        print(f"Wrote OCR output to: {txt_file_path}")
