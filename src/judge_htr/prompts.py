from typing import List, Dict, Any
from judge_htr import configs
import json


def make_paged_transcription_schema(page_ids: List[int]) -> Dict[str, Any]:
    """
    Build a Structured Outputs-compatible response_format for the OpenAI API
    that forces the model to return indexed page transcriptions for exactly
    the provided page IDs.

    Top-level object keys are the page IDs (as strings). Each key is required.
    Values are the page's transcribed text (type:string).

    Args:
        page_ids: List of integer page identifiers (need not be contiguous or start at 0).

    Returns:
        A dict you can pass as the `response_format` argument.
    """
    if not page_ids:
        raise ValueError("page_ids must be a non-empty list of integers.")
    if any(not isinstance(pid, int) for pid in page_ids):
        raise TypeError("All page_ids must be integers.")

    # de-duplicate while preserving order
    ordered_unique_ids = list(dict.fromkeys(page_ids))

    properties: Dict[str, Any] = {
        str(pid): {"type": "string", "description": f"Transcribed text for page {pid}"}
        for pid in ordered_unique_ids
    }

    schema = {
        "format": {
            "type": "json_schema",
            "name": "paged_transcription",
            "strict": True,  # require exact adherence to the schema
            "schema": {
                "type": "object",
                "description": "Map of fixed page IDs to their transcribed text.",
                "properties": properties,
                "required": [str(pid) for pid in ordered_unique_ids],
                "additionalProperties": False,  # disallow pages not in page_ids
            },
        }
    }

    return schema


def make_paged_transcription_json(
    page_ids: list[int],
    pages: list[str],
    *,
    pretty: bool = True,
    sort_pages: bool = True,
) -> str:
    """
    Convert a dict of {page_id:int -> text:str} into a JSON string suitable for
    'structured text' usage (e.g., feeding to a model or saving as output).

    - Keys become strings (e.g., 3 -> "3")
    - Values are the raw text for each page
    - Optional pretty-printing and numeric sorting

    Raises:
        ValueError / TypeError on invalid input.
    """
    pages = dict(zip(page_ids, pages))

    out = {}
    for pid, text in pages.items():
        if not isinstance(pid, int):
            raise TypeError(f"Page id {pid!r} is not an int.")
        if not isinstance(text, str):
            raise TypeError(f"Text for page {pid} must be a str.")
        out[str(pid)] = text

    if sort_pages:
        # sort numerically by page id
        out = {k: out[k] for k in sorted(out.keys(), key=lambda s: int(s))}

    return json.dumps(
        out,
        ensure_ascii=False,  # keep Unicode intact
        indent=2 if pretty else None,  # pretty-print if requested
        separators=None if pretty else (",", ":"),
    )


def make_page_selection_schema(page_ids: list[int]) -> dict:
    """
    Build a Structured Outputs-compatible response_format for the OpenAI API
    that forces the model to return a single chosen page_id (restricted to the
    given page_ids) and a brief reasoning string—nothing else.

    Output shape:
    {
      "page_id": <one of page_ids>,
      "reasoning": "<brief explanation>"
    }
    """
    if not page_ids:
        raise ValueError("page_ids must be a non-empty list of integers.")
    if any(not isinstance(pid, int) for pid in page_ids):
        raise TypeError("All page_ids must be integers.")

    # de-duplicate while preserving order
    ordered_unique_ids = list(dict.fromkeys(page_ids))

    schema = {
        "format": {
            "type": "json_schema",
            "name": "page_selection",
            "strict": True,  # require exact adherence to the schema
            "schema": {
                "type": "object",
                "description": "Chosen page id (must be one of the provided page_ids) and brief reasoning.",
                "properties": {
                    "page_id": {
                        "type": "integer",
                        "enum": ordered_unique_ids,
                        "description": "The selected page ID.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "A brief explanation for choosing that page.",
                    },
                },
                "required": ["page_id", "reasoning"],
                "additionalProperties": False,  # disallow anything else
            },
        }
    }

    return schema


with open(configs / "prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
