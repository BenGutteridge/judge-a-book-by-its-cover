"""
Post-processing and plotting utilities, as well as string mappings,
for the 01_experiments.py and succeeding notebooks.
"""

import re
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
from typing import Optional
import PIL
from google import genai
from loguru import logger
from dotenv import load_dotenv
import json
from typing import Any, Dict, List
from google.genai import types
import time
from collections import defaultdict


def make_dummy_response():
    """Returns a dummy response dict for failed API calls so the next call can proceed."""
    return defaultdict(
        lambda: None,
        {
            "response": "Placeholder response; API call failed.",
            "usage": {
                "output_tokens": 0,
                "input_tokens": 0,
                "cached_tokens": 0,
            },
            "cost": 0.0,
        },
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)  # first {...} block


def _validate(obj: Any, schema: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Minimal validator: if a JSON Schema is provided, check required keys
    and that values are strings. (Keep it tiny; extend as needed or plug in jsonschema.)
    """
    if not schema:
        return None
    if schema.get("type") == "object":
        required: List[str] = schema.get("required", [])
        props: Dict[str, Any] = schema.get("properties", {})
        if not isinstance(obj, dict):
            return "Top-level value is not an object."
        missing = [k for k in required if k not in obj]
        if missing:
            return f"Missing required keys: {missing}"
        # very light type checks
        for k, v in obj.items():
            expected = props.get(k, {"type": "string"}).get("type", "string")
            if expected == "string" and not isinstance(v, str):
                return f"Key '{k}' should be string."
    return None


def gemma_json_call(
    client,
    model: str,
    contents: list,
    schema: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    **cfg_kwargs,
):
    """
    Calls a Gemma model and coerces the output to valid JSON.
    - `contents`: full prompt contents including system.
    - `schema`: optional (JSON-Schema-like) shape; we do light checks.
      For strict validation, swap _validate() with the jsonschema library.
    """
    # Strong JSON-only prompt (Gemma doesn't accept system_instruction)
    json_rules = [
        "**Output Format**:\n\nReturn ONLY valid JSON. ",
        "No explanations, no markdown, no code fences.",
        "Do not include comments or trailing commas.",
    ]
    if schema:
        json_rules.append("Conform to this schema (keys and types):")
        json_rules.append(json.dumps(schema, indent=2, ensure_ascii=False))

    output_instructions = "\n".join(f"- {r}" for r in json_rules)

    # Build contents: Gemma has no system_instruction; put everything in prompt
    contents.insert(1, output_instructions)  # after system prompt but before user

    attempt, wait = -1, 0
    while attempt < max_retries:
        attempt += 1
        time.sleep(wait)  # exponential backoff
        wait += 10
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            wait = 0  # reset wait on success, wait is only for resource exhaustion
            text = resp.text or ""
            obj = json_loads_safe(text)
            err = _validate(obj, schema)
            if obj is None or err:
                cfg_kwargs["temperature"] = min(
                    max(0.3, cfg_kwargs["temperature"] + 0.1), 1.0
                )
                if cfg_kwargs["temperature"] > 0.5:
                    text_check = text.rstrip("\\n") + '"}'
                    if json_loads_safe(text_check) and not _validate(
                        json_loads_safe(text_check), schema
                    ):
                        logger.warning(
                            f"Invalid JSON ('{err}'), but fixed by adding missing '}}'.\n"
                            f"Fixed output:\n{text_check}"
                        )
                        return json_loads_safe(text_check), resp

                raise ValueError(
                    f"Invalid JSON ('{err}'), raising temperature to {cfg_kwargs['temperature']}:\nInvalid output:\n{text}"
                )
            return obj, resp

        except Exception as e:
            logger.warning(f"Gemma API call failed: {e}")

    # Need a dummy response
    return None, None


def _extract_json(s: str) -> str:
    """
    Return the first top-level JSON object found in s.
    Falls back to s if no {...} block is detected (so json.loads can still try).
    """
    m = _JSON_BLOCK_RE.search(s)
    return m.group(0) if m else s.strip()


def json_loads_safe(s: str) -> dict | None:
    try:
        res = _extract_json(s)
        try:
            return json.loads(res)
        except json.JSONDecodeError as e:
            # Repair common mistake: \'
            if "Invalid \\escape" in str(e):
                fixed = re.sub(r"\\(?=\')", "", res)
                return json.loads(fixed)
            logger.warning(f"JSON decode failed: {e}")
            return None
    except Exception as e:
        logger.warning(f"Failed to decode JSON: {e}")
        return None


# %%
# %%
# Gemini token counting utilities, since the Gemini client one is broken for images
# (always returns 258 tokens regardless of input)
load_dotenv()
client = genai.Client()
tokens_per_tile = 258


def gemini_token_counter(
    model: str, urls: list[str] = [], prompts: list[str] = []
) -> int:
    images = [PIL.Image.open(url) for url in urls]
    contents = images + prompts
    return client.models.count_tokens(model=model, contents=contents).total_tokens


def approx_gemini_image_token_counter(url: str) -> int:

    # 1. get image dimensions in pixels
    with PIL.Image.open(url) as im:
        im.load()
        w, h = im.width, im.height
    # "With Gemini 2.0, image inputs with both dimensions <=384 pixels are counted as 258 tokens. Images larger in one or both dimensions are cropped and scaled as needed into tiles of 768x768 pixels, each counted as 258 tokens. Prior to Gemini 2.0, images used a fixed 258 tokens"

    if w <= 384 and h <= 384:
        return 1 * tokens_per_tile

    crop_unit_size = min(w, h) // 1.5
    num_tiles = (w / crop_unit_size) * (h / crop_unit_size)

    return int(num_tiles * tokens_per_tile)


def approximate_gemini_token_counter(
    model: str,
    urls: list[str] | dict[int, str] = [],
    prompts: list[str] = [],
    verbose: bool = False,
) -> int:

    if isinstance(urls, dict):
        urls = list(urls.values())
    image_tokens = sum(approx_gemini_image_token_counter(url) for url in urls)
    prompt_tokens = gemini_token_counter(model=model, prompts=prompts)
    total_tokens = image_tokens + prompt_tokens
    if verbose:
        api_token_count = gemini_token_counter(model=model, urls=urls, prompts=prompts)
        logger.info(
            "Difference between approximate and 'actual' token count: "
            f"{total_tokens - api_token_count} token, \n"
            f"~{(total_tokens - api_token_count) / tokens_per_tile:.1f} image tiles"
        )

    return total_tokens


def mllm_output_postprocessing(s: str | None) -> str | None:
    """
    General purpose MLLM output string postprocessing.
    - Convert escaped newlines to their actual values
    - Replace multiple newlines with single newline, and multiple spaces with single space
    - Remove whitespace before newlines, e.g. "\n[NEW_PAGE]  \n" -> "\n[NEW_PAGE]\n"
    """
    if s is None:
        return None

    s = re.sub(r"\\+n", "\n", s)
    while s != (s := s.replace("\n\n", "\n")):
        pass
    while s != (s := s.replace("  ", " ")):
        pass
    s = re.sub(r"\s+\n", "\n", s)
    return s
