import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time
from pathlib import Path


class DocOwlInfer:
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.init_processor(
            tokenizer=self.tokenizer,
            basic_image_size=504,
            crop_anchors="grid_12",
        )

    def inference(self, images, query):
        messages = [{"role": "USER", "content": "<|image|>" * len(images) + query}]
        answer = self.model.chat(
            messages=messages, images=images, tokenizer=self.tokenizer
        )
        return answer


def main(dir_to_process: str):

    docowl = DocOwlInfer(ckpt_path="mPLUG/DocOwl2")

    whole_page_prompt = "Given the above handwritten document page, transcribe the full text accurately. Return all of the text on the page in reading order and nothing else. Use correct capitalization. Use `\\n` to indicate new lines."

    output_dir = "malvern_hills_trust_docowl2_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # walk recursively through a directory to get all image paths
    for root, dirs, files in sorted(os.walk("malvern_hills_trust")):
        if root != dir_to_process:
            continue
        for file in sorted(files):
            if file.endswith(".jpeg"):
                img_path = os.path.join(root, file)
                output_path = Path(
                    img_path.replace(
                        "malvern_hills_trust/", "malvern_hills_trust_docowl2_outputs/"
                    ).replace(".jpeg", ".txt")
                )
                if output_path.exists():
                    print(f"Output already exists for {img_path}, skipping.")
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Processing {img_path}...")
                t0 = time.time()
                res: str = docowl.inference([img_path], query=whole_page_prompt)
                t1 = time.time() - t0
                print(f"Result:\n{res}")
                # Save to text file
                with open(output_path, "w") as f:
                    f.write(res)

                Path.touch(str(output_path).replace(".txt", f"_t={int(t1*1000):d}ms"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python run_docowl.py <dir_to_process>")
        sys.exit(1)
    dir_to_process = sys.argv[1]
    main(dir_to_process)
