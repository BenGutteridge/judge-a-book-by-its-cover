"""
Zero-shot line transcription with PyLaia + lexicon/LM decoding.

Usage (example, English IAM model + your domain corpus):
  python pylaia_zero_shot_infer.py \
      --image path/to/line.png \
      --model Teklia/pylaia-iam \
      --corpus_txt path/to/domain_corpus.txt

If your chosen HF repo already ships an ARPA (e.g., Teklia/pylaia-huginmunin),
you can skip --corpus_txt and it will use the bundled LM automatically.

Requires:
  pip install pylaia huggingface_hub
  # KenLM (for building LM): make sure `lmplz` is on PATH

This code was partially written by ChatGPT.
"""

import argparse, os, sys, shutil, subprocess, tempfile, gzip
from pathlib import Path
from PIL import Image
from time import time


# Optional: prefer HF API over git clone so you don't need git-lfs locally
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


def die(msg, code=1):
    print(f"[ERR] {msg}", file=sys.stderr)
    sys.exit(code)


def which(name):
    from shutil import which as _which

    return _which(name)


def ensure_model_dir(model_spec: str, cache_dir: Path) -> Path:
    """
    model_spec: HF repo id like 'Teklia/pylaia-iam' or a local path
    returns local directory containing:
      - 'model' (architecture file) and 'weights.ckpt'
      - 'syms.txt' (required mapping)
      - optionally: 'language_model.arpa' or '.arpa.gz', 'tokens.txt', 'lexicon.txt'
    """
    p = Path(model_spec)
    if p.exists():
        return p.resolve()

    if snapshot_download is None:
        die(
            "huggingface_hub not available; either pip install it or pass a local model path."
        )

    local_dir = Path(
        snapshot_download(
            repo_id=model_spec,
            local_dir=cache_dir,
        )
    )
    return local_dir


def find_file(root: Path, names):
    if isinstance(names, (str, Path)):
        names = [names]
    for n in names:
        cand = root / n
        if cand.exists():
            return cand
    # fallback search
    for n in names:
        hits = list(root.rglob(n))
        if hits:
            return hits[0]
    return None


def build_char_lm_with_kenlm(corpus_txt: Path, out_arpa: Path, order: int = 6):
    if not which("lmplz"):
        die(
            "KenLM not found. Install and ensure `lmplz` is on PATH (see https://github.com/kpu/kenlm)."
        )
    out_arpa.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lmplz",
        "--order",
        str(order),
        "--text",
        str(corpus_txt),
        "--arpa",
        str(out_arpa),
        "--discount_fallback",
    ]
    # print("[INFO] Building KenLM character LM:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def syms_to_tokens_and_lexicon(syms_txt: Path, tokens_txt: Path, lexicon_txt: Path):
    # tokens.txt = first column of syms.txt (order preserved)
    # lexicon (char-level): each token maps to itself
    toks = []
    with syms_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: "<token> <index>"
            parts = line.split()
            token = parts[0]
            toks.append(token)

    tokens_txt.write_text("\n".join(toks) + "\n", encoding="utf-8")

    with lexicon_txt.open("w", encoding="utf-8") as g:
        for t in toks:
            g.write(f"{t} {t}\n")


def choose_language_model(
    model_dir: Path, user_arpa: Path | None, corpus_txt: Path | None
) -> Path:
    # 1) user-provided ARPA wins
    if user_arpa:
        return user_arpa

    # 2) bundled LM in the model repo (often .arpa or .arpa.gz)
    lm = None
    for name in [
        "language_model.arpa",
        "language_model.arpa.gz",
        "model_words.arpa",
        "model_characters.arpa",
        "model_subwords.arpa",
        "language_model.arpa.txt",
    ]:
        hit = find_file(model_dir, name)
        if hit:
            lm = hit
            break

    if lm:
        # If it's gz, return gz path directly—PyLaia supports ARPA (gz) in docs.
        return lm

    # 3) build from corpus (character-level)
    if corpus_txt:
        out_arpa = model_dir / "language_model" / "model_characters.arpa"
        build_char_lm_with_kenlm(corpus_txt, out_arpa, order=6)
        return out_arpa

    die(
        "No language model found. Provide --lm_arpa or --corpus_txt to build one, "
        "or use a HF model that already includes an ARPA (e.g., Teklia/pylaia-huginmunin)."
    )


def decode_single_line(
    model_dir: Path,
    image_path: Path,
    lm_arpa: Path,
    join_string: str = "",
    convert_spaces: bool = True,
    device_gpus: int = 0,
) -> str:
    """
    Runs pylaia-htr-decode-ctc with LM + lexicon and returns plain text.
    """
    # Resolve required files
    syms = find_file(model_dir, "syms.txt") or die(
        "syms.txt not found in model directory."
    )
    model_file = find_file(model_dir, "model") or die(
        "model (architecture file) not found."
    )
    weights = find_file(model_dir, "weights.ckpt") or die("weights.ckpt not found.")

    # Ensure tokens/lexicon exist (build char-level ones if missing)
    tokens = find_file(model_dir, "tokens.txt")
    lexicon = find_file(model_dir, "lexicon.txt")
    if tokens is None or lexicon is None:
        tokens = model_dir / "tokens.txt"
        lexicon = model_dir / "lexicon.txt"
        syms_to_tokens_and_lexicon(syms, tokens, lexicon)

    # Prepare a temp workspace with an img_list and image copy
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        img_dir = tdir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # PyLaia expects img_ids and an --img_dirs; it will resolve by filename stem.
        stem = "input_line"
        ext = image_path.suffix
        target = img_dir / f"{stem}{ext}"

        # Load, convert to grayscale, and resize to the model's expected height.
        # For Teklia/pylaia-iam this is 128 px (fixed height, keep aspect ratio).
        img = Image.open(image_path)
        img = img.convert("L")  # grayscale

        target_h = 128  # IAM model: training images resized to height 128
        w, h = img.size
        if h != target_h:
            new_w = max(1, int(round(w * (target_h / h))))
            img = img.resize((new_w, target_h), Image.BILINEAR)

        img.save(target)

        (tdir / "img_list.txt").write_text(stem + "\n", encoding="utf-8")

        # Write a minimal YAML config for decoding
        cfg = tdir / "config_decode_model_lm.yaml"
        cfg.write_text(
            f"""syms: {syms}
img_list: {tdir/'img_list.txt'}
img_dirs:
  - {img_dir}
common:
  experiment_dirname: {model_dir}
  model_filename: {model_file}
  checkpoint: "{weights}"  # absolute path or glob OK
decode:
  include_img_ids: false
  join_string: "{join_string}"
  convert_spaces: {"true" if convert_spaces else "false"}
  use_language_model: true
  language_model_path: {lm_arpa}
  tokens_path: {tokens}
  lexicon_path: {lexicon}
  language_model_weight: 1.5
trainer:
  gpus: {device_gpus}
""",
            encoding="utf-8",
        )

        # Run the decoder
        cmd = ["pylaia-htr-decode-ctc", "--config", str(cfg)]
        print("[INFO] Running:", " ".join(cmd), file=sys.stderr)

        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("=== pylaia-htr-decode-ctc STDOUT ===", file=sys.stderr)
            print(e.stdout or "", file=sys.stderr)
            print("=== pylaia-htr-decode-ctc STDERR ===", file=sys.stderr)
            print(e.stderr or "", file=sys.stderr)
            die(f"pylaia-htr-decode-ctc failed with code {e.returncode}")

        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        if not lines:
            die("Decoder returned no output.")
        return lines[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image", required=True, help="Path to a single line image (png/jpg/...)."
    )
    ap.add_argument(
        "--model",
        required=True,
        help="HF repo id (e.g., Teklia/pylaia-iam) or a local path.",
    )
    ap.add_argument(
        "--lm_arpa",
        default=None,
        help="Path to an ARPA (or .gz). If omitted, tries bundled LM, else builds from --corpus_txt.",
    )
    ap.add_argument(
        "--corpus_txt",
        default=None,
        help="Plain text file to build a character LM with KenLM (used if no --lm_arpa found).",
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="0=CPU; n=use n GPUs (PyLaia Lightning flag).",
    )
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        die(f"Image not found: {image_path}")

    cache = Path.home() / ".cache" / "pylaia_models"
    model_dir = ensure_model_dir(args.model, cache)

    lm = choose_language_model(
        model_dir,
        Path(args.lm_arpa).resolve() if args.lm_arpa else None,
        Path(args.corpus_txt).resolve() if args.corpus_txt else None,
    )

    text = decode_single_line(
        model_dir=model_dir, image_path=image_path, lm_arpa=lm, device_gpus=args.gpus
    )

    print(text)
    return text


class LAIAInference:
    def __init__(
        self,
        model_spec: str,
        lm_arpa: Path | None = None,
        corpus_txt: Path | None = None,
        gpus: int = 0,
    ):
        self.model_dir, self.lm = get_language_model(model_spec, lm_arpa, corpus_txt)
        self.gpus = gpus

    def run(self, image_path: Path) -> str:
        t0 = time()
        text = decode_single_line(
            model_dir=self.model_dir,
            image_path=image_path,
            lm_arpa=self.lm,
            device_gpus=self.gpus,
        )
        return text, time() - t0


def get_language_model(model_spec, lm_arpa, corpus_txt):

    cache = Path.home() / ".cache" / "pylaia_models"
    model_dir = ensure_model_dir(model_spec, cache)

    lm = choose_language_model(
        model_dir,
        lm_arpa,
        corpus_txt,
    )

    return model_dir, lm


def run(
    image_path: Path,
    model_spec: str,
    lm_arpa: Path | None = None,
    corpus_txt: Path | None = None,
    gpus: int = 0,
) -> str:
    """
    Runs the inference pipeline as a function call.

    Args:
    - image_path (Path): Path to the line image.
    - model_spec (str): HF repo id or local path to the model.
    - lm_arpa (Path | None): Optional path to an ARPA LM.
    - corpus_txt (Path | None): Optional path to a corpus for building LM.
    - gpus (int): Number of GPUs to use (0 for CPU).

    Returns:
    - str: The transcribed text.
    """

    model_dir, lm = get_language_model(model_spec, lm_arpa, corpus_txt)

    text = decode_single_line(
        model_dir=model_dir, image_path=image_path, lm_arpa=lm, device_gpus=gpus
    )

    return text, time_in_seconds


if __name__ == "__main__":
    main()
