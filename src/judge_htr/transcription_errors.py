#!/usr/bin/env python3
"""
markdown_diff.py — produce a single Markdown string that highlights differences
between two input strings.

- Deletions are wrapped with ~~strike~~
- Insertions are wrapped with **bold**
- Uses word-level diff by default; pass --char for character-level
- Safe to embed in Markdown (escapes special characters inside wrappers)
"""

import re
import difflib
from judge_htr.mllm.caller import MLLM, api_call
import dotenv
from judge_htr.postprocessing import json_loads_safe

dotenv.load_dotenv()


# ---------- core helpers ----------

_WORDISH_RE = re.compile(r"\w+|\s+|[^\w\s]", flags=re.UNICODE)


def _tokenize(text: str, mode: str = "word"):
    """Split text into tokens for diffing."""
    if mode == "char":
        return list(text)
    return _WORDISH_RE.findall(text)


def _escape_md(s: str) -> str:
    """Escape Markdown/HTML specials so they don't break formatting."""
    specials = "\\`*_{}[]()#+-.!>|~<>&"
    out = []
    for ch in s:
        out.append("\\" + ch if ch in specials else ch)
    return "".join(out)


def markdown_diff(
    before: str,
    after: str,
    mode: str = "word",
    add_wrapper: tuple[str, str] = ("<ins>", "</ins>"),
    del_wrapper: tuple[str, str] = ("<del>", "</del>"),
) -> str:
    """
    Return a single Markdown string showing differences between before and after.

    - mode: "word" (default) or "char"
    - add_wrapper / del_wrapper: (open, close) wrapper strings, e.g.
        add_wrapper=("<ins>","</ins>"), del_wrapper=("<del>","</del>")
      if you prefer HTML tags.
    """
    before_tokens = _tokenize(before, mode)
    after_tokens = _tokenize(after, mode)
    sm = difflib.SequenceMatcher(a=before_tokens, b=after_tokens, autojunk=False)
    pieces: list[str] = []

    def _wrap(segment: str, wrapper: tuple[str, str]) -> str:
        # Don't wrap pure whitespace; otherwise wrap and escape internals
        if segment.strip() == "":
            return segment
        return f"{wrapper[0]}{_escape_md(segment)}{wrapper[1]}"

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pieces.append("".join(before_tokens[i1:i2]))
        elif tag == "delete":
            pieces.append(_wrap("".join(before_tokens[i1:i2]), del_wrapper))
        elif tag == "insert":
            pieces.append(_wrap("".join(after_tokens[j1:j2]), add_wrapper))
        elif tag == "replace":
            pieces.append(_wrap("".join(before_tokens[i1:i2]), del_wrapper))
            pieces.append(_wrap("".join(after_tokens[j1:j2]), add_wrapper))
    return "".join(pieces)


ERRORS_ONLY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "errors",
    "strict": True,  # enable Structured Outputs (schema adherence)
    "schema": {
        "type": "object",
        "properties": {
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        # required-but-nullable (key must exist; value may be null)
                        "gt": {"type": ["string", "null"]},
                        "pred": {"type": "string"},
                        "error_type": {
                            "type": "string",
                            "enum": [
                                "missing_content",
                                "hallucination",
                                "mistake_proper_noun",
                                "mistake_numerical",
                                "mistake_other",
                                "semantic",
                                "formatting",
                            ],
                        },
                    },
                    "required": ["id", "gt", "pred", "error_type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["errors"],
        "additionalProperties": False,
    },
}


get_errors_prompt = """You will receive a single Markdown diff string where deleted text is ground truth and inserted text is a prediction.
Extract **all transcription errors** and return **JSON only**.

Response format (API-enforced):
- Return a single JSON object with a top-level key `errors`.
- Each item in `errors` must be an object with keys: `id` (int ≥ 1), `gt` (string or null), `pred` (string), `error_type` (string).
- `error_type` must be exactly one of:
  - "missing_content"
  - "hallucination"
  - "mistake_proper_noun"
  - "mistake_numerical"
  - "mistake_other"
  - "semantic"
  - "formatting"
- If no errors are found, return `{"errors": []}` only.
- Do not include any other keys or any prose.

Diff markup:
- Deletions: `<del>gt text</del>` are ground-truth tokens removed or replaced.
- Insertions: `<ins>pred text</ins>` are predicted tokens added or replacing gt.
- A replacement is a delete immediately followed by an insert in the same position. Prefer pairing into one error when they clearly correspond.

Field rules:
- `gt`: deleted text for this error or `null` when there is no corresponding deletion (e.g., pure insertion/hallucination).
- `pred`: inserted text for this error. If there is no inserted counterpart (pure deletion), set `pred` to the empty string `""` (not null).
- Preserve text exactly as it appears inside the diff markers. Do not trim or normalize whitespace, punctuation, or case.
- Do not include diff marker text (`<ins>`, `<del>`) inside the returnd gt or pred

Error types (choose exactly one per error):
1) missing_content
  - A deletion with no suitable inserted counterpart. Example: `<del>word</del>` with no matching `<ins>word</ins>`.
  - Minor missing content, such as single punctuation marks, should be `formatting`.
2) hallucination
  - An insertion with no corresponding gt (including numbers that appear only in pred).
  - Minor hallucinations, such as single punctuation marks, should be `formatting`.
3) mistake_proper_noun
  - gt is a proper noun (possibly multi-word: full or partial names, places, institutions, titles) and pred misrenders it (letters/wording/order). 
  - Pure capitalization changes that don’t create a different name are semantic.
  - Extended 'proper names', such as full names, should be treated in their entirety, e.g. 'A. B. Smith'
4) mistake_numerical
  - Use only when gt contains a number/date/roman numeral/measure and pred changes its value or structure or fails to include it. 
  - Combine obviously linked components (e.g., an entire date like 18 March 1884) into one numerical error.
5) mistake_other
  - Non–proper-noun, non-numeric word/phrase error that affects normal reading (e.g., genuine misspelling or wrong lemma/word).
  - Use sparingly, only if none of the other error types are appropriate, not as a catch-all
6) semantic
  - Differences that are clearly meaning-preserving or trivial:
    - minor letter/spelling differences where the word couldn’t reasonably be mistaken for another,
    - alternative/modernized spellings or contractions with the same meaning
    - capitalization-only changes,
7) formatting
  - Minor erroneous punctuation, misplaced or missing newlines, misplaced structural content including text that has been moved from one place in the transcription to another
  - If a moved block contains internal real mistakes (e.g., a wrong year), add separate errors for those internal pieces with the appropriate types.

Granularity & pairing:
- Prefer a single error for a simple replacement (one `<del>gt</del>` + one `<ins>pred</ins>` that clearly correspond).
- Split into multiple errors when doing so isolates materially different phenomena (e.g., a numeric change embedded inside an otherwise proper-noun phrase).

Normalization:
- Return sensible and meaningful `gt`/`pred`; e.g. if there is a change in a name, include the whole name, not just the changed letter
- Punctuation/whitespace-only diffs are formatting unless they are part of a numeric expression from gt (then consider mistake_numerical).
- When unsure between mistake_other and semantic, choose semantic if meaning is clearly unchanged.
- If the gt is misspelled or an archaic/contracted spelling, and the pred is a corrected/modern/complete spelling, opt for `semantic` rather than a `mistake` if the meaning is preserved and/or comprehension is enhanced.

You should consider the local, semantic context of an error to accurately classify it, not just the deleted/inserted text.

What to return:
- Only the JSON object described above, with the `errors` array of items that conform to the schema. No extra text, headings, or code fences."""


def get_transcription_errors(
    gts: dict[int, str],
    preds: dict[int, str],
    mllm: MLLM,
    only_load_from_cache: bool = False,
) -> list[dict]:

    out = {}
    for pid in gts.keys():
        gt, pred = gts[pid], preds[pid]
        diff = markdown_diff(gt, pred, mode="word")
        if diff == gt:
            # no diff
            out[pid] = {"errors": [], "diff": diff}
            continue
        res = mllm.call(
            user_prompt=diff,
            system_prompt=get_errors_prompt,
            only_load_from_cache=only_load_from_cache,
            **{
                "text": {"format": ERRORS_ONLY_RESPONSE_FORMAT},
                "temperature": 0,
                "reasoning": {"effort": "minimal"},
            },
        )
        if res is None or (errors := json_loads_safe(res["response"])) is None:
            return None
        out[pid] = errors | {"diff": diff}
    return out
