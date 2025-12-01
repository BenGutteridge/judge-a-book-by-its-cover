"""
Crop line images from a Kraken segmentation JSON.
Used in notebooks/00_line_preprocessing.py to generate line crops for TrOCR and other non-MLLM experiments that operate on line-level images.

Notes:
- Uses each line's polygon "boundary" to compute a tight bounding box (with padding).
- Falls back to "baseline" if no boundary exists.
- Saves crops as {prefix}{idx:05d}_{line_id}.{ext} and writes:
    - lines.lst  (newline-separated list of crop paths)
    - lines.csv  (id,x0,y0,x1,y1,filename)
- Sorting is top-to-bottom, then left-to-right.
"""

import json
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

from PIL import Image


def clamp_bbox(x0, y0, x1, y1, w, h):
    x0 = max(0, min(int(x0), w))
    y0 = max(0, min(int(y0), h))
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    # ensure non-empty
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def bbox_from_points(pts: Iterable[Tuple[float, float]], pad: int, w: int, h: int):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    return clamp_bbox(x0, y0, x1, y1, w, h)


def line_key_for_sort(line: Dict[str, Any]) -> Tuple[float, float]:
    """
    Sort lines top-to-bottom, then left-to-right (using boundary if available, else baseline).
    """

    def first_xy(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(ys) if ys else 0.0, min(xs) if xs else 0.0)

    if line.get("boundary"):
        return first_xy(line["boundary"])
    if line.get("baseline"):
        return first_xy(line["baseline"])
    return (0.0, 0.0)


def crop_lines(
    img_path: str,
    json_path: str,
    out_dir: str,
    pad: int = 4,
    prefix: str = "line_",
    ext: str = "png",
):

    img_path = Path(img_path)
    out_dir = Path(out_dir)
    json_path = Path(json_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img = Image.open(img_path).convert("RGB")
    W, H = img.width, img.height

    # Load JSON
    data = json.loads(json_path.read_text())
    lines = data.get("lines", [])
    if not lines:
        raise SystemExit("No 'lines' found in JSON.")

    # Sort lines top-to-bottom then left-to-right
    lines_sorted = sorted(lines, key=line_key_for_sort)

    lst_lines: List[str] = []
    csv_rows: List[str] = ["id,x0,y0,x1,y1,filename"]

    for idx, line in enumerate(lines_sorted):
        line_id = line.get("id", f"noid_{idx:05d}")
        boundary = line.get("boundary")
        baseline = line.get("baseline")

        if boundary and isinstance(boundary, list) and len(boundary) >= 2:
            pts = [(float(x), float(y)) for x, y in boundary]
        elif baseline and isinstance(baseline, list) and len(baseline) >= 2:
            # fallback: make a bbox around baseline with extra vertical slack (pad*3)
            pts = [(float(x), float(y)) for x, y in baseline]
        else:
            # nothing usable
            continue

        # If using baseline fallback, add extra vertical padding
        extra_vpad = pad * 3 if (not boundary and baseline) else 0
        x0, y0, x1, y1 = bbox_from_points(pts, pad, W, H)
        if extra_vpad:
            x0, y0, x1, y1 = clamp_bbox(x0, y0 - extra_vpad, x1, y1 + extra_vpad, W, H)

        crop = img.crop((x0, y0, x1, y1))

        # Save file
        fname = f"{prefix}{idx:03d}_{line_id}.{ext}"
        fpath = out_dir / fname
        # For JPEG ensure quality
        save_kwargs = {}
        if ext.lower() in ("jpg", "jpeg"):
            save_kwargs.update(dict(quality=95, optimize=True, subsampling=0))
        crop.save(fpath, **save_kwargs)

        lst_lines.append(str(fpath))
        csv_rows.append(f"{line_id},{x0},{y0},{x1},{y1},{fname}")

    # Write index files
    (out_dir / "lines.lst").write_text(
        "\n".join(lst_lines) + ("\n" if lst_lines else "")
    )
    (out_dir / "lines.csv").write_text("\n".join(csv_rows) + "\n")

    print(f"Cropped {len(lst_lines)} lines to: {out_dir}")
    print(f" - paths list: {out_dir / 'lines.lst'}")
    print(f" - bbox csv:   {out_dir / 'lines.csv'}")
