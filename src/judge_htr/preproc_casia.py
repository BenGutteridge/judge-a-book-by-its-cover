import os
import struct
from pathlib import Path
import numpy as np
from PIL import Image


def _read_int32(f):
    return int.from_bytes(f.read(4), byteorder="little", signed=False)


def _read_int16(f):
    return int.from_bytes(f.read(2), byteorder="little", signed=False)


def _decode_1bpp_to_uint8(h, w, buf):
    """Decode H * ceil(W/8) bytes of packed 1bpp into HxW uint8 (0/255)."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    # unpackbits gives bits MSB-first per byte; CASIA uses standard packing
    bits = np.unpackbits(arr).reshape(-1)
    # We have H * ceil(W/8) bytes => H * ceil(W/8) * 8 bits, take first H*W
    bits = bits[: h * ((w + 7) // 8) * 8].reshape(-1, 8)
    bits = bits.reshape(-1)[: h * w].reshape(h, w)
    # Map 1->0 (foreground) or 255? Spec allows B/W; we treat 1 as foreground black (0)
    # If your samples look inverted, swap the mapping.
    return np.uint8(np.where(bits == 1, 0, 255))


def read_dgrl(path):
    """
    Returns:
        page (np.uint8 HxW): reconstructed page (255=white)
        meta (dict): header/meta info
        lines (list[dict]): each with keys ['top','left','h','w','bitmap','label']
                            bitmap is np.uint8 (HxW)
    """
    with open(path, "rb") as f:
        header_size = _read_int32(f)  # 4
        dgrl_code = f.read(8)  # "DGRL" + padding
        illustr_len = header_size - 36  # 36 + strlen(illustr)
        _illustr = f.read(illustr_len)  # not needed for imaging
        code_type = f.read(20).split(b"\x00", 1)[0].decode("ascii", "ignore").strip()
        code_len = _read_int16(f)  # bytes per char label (often 2 for GBK)
        bits_pp = _read_int16(f)  # typically 8 (grayscale) or 1 (B/W)

        # Page record
        page_h = _read_int32(f)
        page_w = _read_int32(f)
        line_n = _read_int32(f)

        page = np.full((page_h, page_w), 255, dtype=np.uint8)
        lines = []

        for _ in range(line_n):
            char_n = _read_int32(f)
            label_bytes = f.read(code_len * char_n)

            # Try a best-effort GBK decode per spec; ignore errors.
            try:
                label = label_bytes.decode("gbk", errors="ignore")
            except Exception:
                label = ""

            top = _read_int32(f)
            left = _read_int32(f)
            h = _read_int32(f)
            w = _read_int32(f)

            if bits_pp == 8:
                buf = f.read(h * w)
                bitmap = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
            elif bits_pp == 1:
                packed = f.read(h * ((w + 7) // 8))
                bitmap = _decode_1bpp_to_uint8(h, w, packed)
            else:
                raise ValueError(f"Unsupported bits per pixel: {bits_pp}")

            # Paste into page with MIN to preserve darkest stroke in overlaps
            y1, y2 = max(0, top), min(page_h, top + h)
            x1, x2 = max(0, left), min(page_w, left + w)
            if y2 > y1 and x2 > x1:
                region = page[y1:y2, x1:x2]
                crop = bitmap[(y1 - top) : (y2 - top), (x1 - left) : (x2 - left)]
                page[y1:y2, x1:x2] = np.minimum(region, crop)

            lines.append(dict(top=top, left=left, h=h, w=w, bitmap=bitmap, label=label))

        meta = dict(
            code_type=code_type,
            code_len=code_len,
            bits_pp=bits_pp,
            page_h=page_h,
            page_w=page_w,
            line_n=line_n,
            dgrl_code=dgrl_code.decode("ascii", "ignore"),
        )
        return page, meta, lines


def save_page_and_lines(dgrl_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page, meta, lines = read_dgrl(dgrl_path)

    # Save page
    page_img = Image.fromarray(page, mode="L")
    page_img.save(out_dir / (Path(dgrl_path).stem + ".png"))

    # Save each line image (and a label text file)
    line_dir = out_dir / (Path(dgrl_path).stem)
    line_dir.mkdir(exist_ok=True)
    for i, ln in enumerate(lines):
        Image.fromarray(ln["bitmap"], mode="L").save(line_dir / f"{i:03d}.png")
        if ln["label"]:
            with open(line_dir / f"{i:03d}.txt", "w", encoding="utf-8") as fp:
                fp.write(ln["label"].replace("\x00", ""))  # remove null chars
