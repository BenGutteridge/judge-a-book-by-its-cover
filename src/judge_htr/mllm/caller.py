"""
Generic MLLM API caller class for aggregation experiments/notebooks.
Supports text and vision MLLM OpenAI API calls.
Includes response processing, caching, and token limit handling.
"""

from time import sleep
from google.genai import types
from google import genai
from urllib import response
from judge_htr import data
from loguru import logger
from typing import Optional, NamedTuple
import openai
from openai import OpenAI
from pathlib import Path
from typing import Callable, Union
import json
import tiktoken
import base64
from PIL import Image
import shutil
import numpy as np
from judge_htr.mllm.cost_estimator import count_image_tokens, CostTracker
import sqlite3
from collections import defaultdict
import hashlib
import re
from judge_htr.postprocessing import (
    _extract_json,
    gemma_json_call,
    json_loads_safe,
    make_dummy_response,
)

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[^;,]+)(?:;base64)?,(?P<data>.*)$", re.DOTALL
)


MAX_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-3.5-turbo": 16_385,
}

default_cache_path = data / "mllm_cache.db"


def openai_api_call(client, model, messages, **kwargs):
    """Wrapper for OpenAI API call."""
    try:
        response = client.responses.create(
            model=model,
            input=messages,
            background=True,
            max_output_tokens=5_000,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")
        return
    while response.status in {"queued", "in_progress"}:
        sleep(3)
        response = client.responses.retrieve(response.id)

    if (
        response.status == "incomplete"
        or response.status == "failed"
        or (
            "text" in kwargs
            and kwargs["text"]["format"]["type"] == "json_schema"
            and json_loads_safe(response.output_text) is None
        )
    ):
        kwargs["temperature"] = max(0.3, kwargs["temperature"] + 0.1)
        if kwargs["temperature"] > 1:
            logger.error("Response incomplete after multiple retries, giving up.")
            return None
        logger.warning(
            f"Call failed, retrying with temperature {kwargs['temperature']}..."
        )
        return openai_api_call(client, model, messages, **kwargs)

    return openai_response_to_dict(response)


def google_api_call(client, model, messages, **kwargs):
    """Wrapper for Google API call."""

    contents = openai_messages_2_gemini_contents(messages)

    cfg = {
        "system_instruction": contents.pop(0),
        "thinking_config": types.ThinkingConfig(
            thinking_budget=(128 if "pro" in model else 0)
        ),
        "maxOutputTokens": 2000,
    }
    if "temperature" in kwargs:
        cfg["temperature"] = kwargs["temperature"]
        cfg["seed"] = kwargs.get("seed", 0)
    if "text" in kwargs:
        JSON_OUTPUT = True
        cfg["response_mime_type"] = "application/json"
        cfg["response_json_schema"] = kwargs["text"]["format"]["schema"]
    else:
        JSON_OUTPUT = False

    if "gemma" in model:
        cfg.pop("thinking_config", None)  # not supported for Gemma
        cfg.pop("response_mime_type", None)
        json_schema = cfg.pop("response_json_schema", None)
        contents = [cfg.pop("system_instruction")] + contents

        if json_schema is not None:
            # Gemma doesn't support JSON schema natively; use our wrapper
            _, response = gemma_json_call(
                client=client,
                model=model,
                contents=contents,
                schema=json_schema,
                **cfg,
            )
            res = gemini_response_to_dict(response)
            res["response"] = _extract_json(res["response"])

            return res

    attempts = 0
    while cfg["temperature"] <= 0.7 and attempts < 3:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg),
            )
            if (
                JSON_OUTPUT
                and json_loads_safe(response.candidates[0].content.parts[0].text)
                is None
            ):
                raise ValueError(f"Failed to parse JSON response: {response}")

            return gemini_response_to_dict(response)
        except Exception as e:
            logger.warning(f"Google API call failed: {e}")
            if "json" in str(e).lower():
                cfg["temperature"] = max(0.3, cfg.get("temperature", 0.0) + 0.1)
                logger.warning(f"Retrying with temperature {cfg['temperature']}...")
            attempts += 1
            sleep(2)
    logger.error("Google API call failed after retries.")
    return make_dummy_response()


def gemini_response_to_dict(response):
    """Converts Gemini response to output dict."""

    if response is None:
        return make_dummy_response()

    res = {
        "response": response.candidates[0].content.parts[0].text,
        "probs": None,
        "usage": {
            "output_tokens": response.usage_metadata.candidates_token_count,
            "input_tokens": response.usage_metadata.prompt_token_count,
            "cached_tokens": 0,
        },
        "model": response.model_version,
    }

    # Gemma responses don't include output tokens
    if "gemma" in res["model"]:
        client = genai.Client()
        res["usage"]["output_tokens"] = client.models.count_tokens(
            model=res["model"], contents=[res["response"]]
        ).total_tokens

    return res


def openai_messages_2_gemini_contents(messages):
    """
    Convert an OpenAI-style `messages` array into a Gemini `contents` list.

    Handles:
      - user.content as a pure string
      - user.content as a list of parts:
          {"type": "input_text", "text": "..."}
          {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}
        (also tolerates OpenAI's {"type":"image_url","image_url":{"url":"data:..."}})

    Returns a list mixing plain strings (text parts) and types.Part (image parts).
    """
    contents = []

    for message in messages:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "input_text":
                    contents.append(part["text"])
                if part["type"] == "input_image":
                    m = _DATA_URL_RE.match(part["image_url"])
                    if not m:
                        # Bare functionality: ignore non-data URLs here
                        continue

                    mime = m.group("mime") or "image/jpeg"
                    raw = base64.b64decode(m.group("data"))
                    contents.append(types.Part.from_bytes(data=raw, mime_type=mime))
        else:
            # Must be string
            contents.append(message["content"])

    return contents


def openai_response_to_dict(response):
    """Converts OpenAI GPT response to output dict."""

    return {
        "response": response.output_text,
        "probs": response_to_tokprobs(response.to_dict()),
        "usage": {
            "output_tokens": response.usage.output_tokens,
            "input_tokens": response.usage.input_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
        },
        "model": response.model,
    }


def api_call(client, model, messages, gemini_client=None, **kwargs):
    """Wrapper for api call."""
    if gemini_client is not None:
        response: dict | None = google_api_call(
            gemini_client, model, messages, **kwargs
        )
    else:
        response: dict | None = openai_api_call(client, model, messages, **kwargs)
    if response is None:
        return None

    return response


class MLLM:
    def __init__(
        self,
        model: str,
        api_call: Callable,
        verbose: bool = False,
        cache_path: Path = default_cache_path,
        update_cache_every_n_calls: int = 10,
        backup_cache_every_n_calls: int = 50,
        max_output_chars: int = 20000,
        min_output_chars: int = 0,
    ):
        self.cost_tracker = CostTracker(model=model)
        self.model = model
        self.api_call = self.cost_tracker(api_call)
        self.cache_path = cache_path
        self.cache_count = 0
        self.verbose = verbose
        self.update_cache_every_n_calls = update_cache_every_n_calls
        self.backup_cache_every_n_calls = backup_cache_every_n_calls
        self.max_output_chars = max_output_chars
        self.min_output_chars = min_output_chars

        # Initialize database for caching
        self.conn = sqlite3.connect(self.cache_path)
        self.cursor = self.conn.cursor()
        self._initialize_database()

        self.client = OpenAI()
        self.gemini_client = genai.Client()

        self.default_system_prompt = (
            "You are ChatGPT, a large language model trained by OpenAI. "
            "Follow the user's instructions carefully. Respond using markdown."
        )  # default system prompt for chat interface

    def _initialize_database(self):
        """Create cache table if it doesn't exist."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mllm_cache (
                input_hash TEXT PRIMARY KEY,
                response_data TEXT
            )
        """
        )
        self.conn.commit()

    def call(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        image_paths: list[Path] | dict[int, Path] | None = None,
        load_from_cache: bool = True,
        only_load_from_cache: bool = True,
        batch_api_filepath: Optional[Path] = None,
        batch_api_custom_id: Optional[str] = None,
        auto_truncate: bool = False,
        **api_call_kwargs,
    ) -> Optional[str]:
        """Generic MLLM call wrapper function.
        Includes the option to include images if image_paths is not None
        Includes the option to write API calls to a batch file instead of making the call directly
        (if batch_api_filepath and batch_api_custom_id are not None).
        NOTE that this only writes to the file, it does not submit the job.

        Args:
            user_prompt (str): The user prompt for the MLLM call.
            system_prompt (str, optional): The system prompt for the MLLM call. None defaults to the standard prompt.
            image_paths (list[Path], optional): List of image paths to include in the MLLM call, if using GPT-vision.
            load_from_cache (bool, optional): Whether to load the MLLM output from cache if available. Defaults to True.
            batch_api_filepath (Path, optional): Filepath to the batch API call file. Defaults to None.
            batch_api_custom_id (str, optional): Custom ID for the batch API call. Defaults to None.
            auto_truncate (bool, optional): Whether to automatically truncate the user prompt if it exceeds the token limit. Defaults to False.
            **api_call_kwargs: Additional keyword arguments for the API call.

        Returns:
            Optional[str]: The MLLM response as a string, or None if writing calls to a batch file.
        """
        original_user_prompt = user_prompt
        system_prompt = (
            system_prompt if system_prompt is not None else self.default_system_prompt
        )
        if user_prompt:
            user_prompt = [{"type": "input_text", "text": user_prompt}]
        else:
            user_prompt = []
        # Option to include images
        if image_paths:
            user_prompt = self.build_image_messages(image_paths) + user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Load from cache if available
        input_hash: str = self.api_input_to_hashable(messages, **api_call_kwargs)
        res = self.load_from_cache(input_hash)

        if (
            res is not None
            and "text" in api_call_kwargs
            and api_call_kwargs["text"]["format"]["type"] == "json_schema"
            and json_loads_safe(res["response"]) is None
        ):
            logger.warning("Cached response JSON is invalid, making new API call.")
            load_from_cache = False

        if load_from_cache and res is not None:
            if res.get("probs", None) is not None:
                res["probs"] = [TokProbs(tokens=t[0], probs=t[1]) for t in res["probs"]]
            if self.verbose:
                logger.info("Loaded MLLM output from cache.")

        # Write to batch API call file if specified
        elif batch_api_filepath and batch_api_custom_id:
            logger.info(
                f"Appending call to batch API call file at {batch_api_filepath}"
            )
            self.save_to_batch_file(
                messages=messages,
                input_hash=input_hash,
                custom_id=batch_api_custom_id,
                batch_filepath=batch_api_filepath,
                **api_call_kwargs,
            )
            return defaultdict(
                lambda: None,  # will return None for all undefined keys
                {
                    "response": "Placeholder response; this call was written to a batch file.",
                    "cost": 0.0,
                    "batch_call": True,
                },
            )  # return a placeholder response that won't be cached or crash scripts

        # Otherwise, make API call
        elif only_load_from_cache:
            logger.info("No cached response found, and only_load_from_cache is set.")
            return None
        else:
            response = self.api_call(
                client=self.client,
                model=self.model,
                messages=messages,
                gemini_client=(self.gemini_client if "gem" in self.model else None),
                **api_call_kwargs,
            )
            if response is None or response == make_dummy_response():
                logger.error("API call failed after retries.")
                res = make_dummy_response()
            else:
                res = self.add_cost_to_response(response)
                self.add_to_cache(input_hash, res)

        # Option to log prompt and response
        if self.verbose:
            prompt_to_log = (
                user_prompt
                if len(user_prompt) < 1000
                else user_prompt[:1000] + "\n...[truncated]"
            )
            response_to_log = (
                res["response"]
                if len(res["response"]) < 1000
                else res["response"][:1000] + "\n...[truncated]"
            )
            logger.info(prompt_to_log)
            logger.info(response_to_log)

        # Log cost, update cache if necessary, backup cache if necessary
        if (self.cache_count + 1) % self.update_cache_every_n_calls == 0:
            self.update_cache()

        # Output length checks and re-submissions, if necessary
        if len(res["response"]) > self.max_output_chars:
            logger.warning(
                f"Response length {len(res['response'])} exceeds max output chars {self.max_output_chars}."
            )
            # Run call again with a prompt addition to prevent catastrophic repeating
            additional_warning = "**NOTE:** be careful not to repeat yourself out of control and let your output become too long."
            api_call_kwargs["temperature"] = min(
                api_call_kwargs["temperature"] + 0.1, 1.0
            )
            logger.warning(f"New temperature: {api_call_kwargs['temperature']}")
            if additional_warning not in system_prompt:
                logger.warning(f"Adding additional warning to system prompt.")
                system_prompt += f"\n\n{additional_warning}"

            original_response = res["response"]
            res = self.call(
                user_prompt=original_user_prompt,
                system_prompt=system_prompt,
                image_paths=image_paths,
                load_from_cache=load_from_cache,
                batch_api_filepath=batch_api_filepath,
                batch_api_custom_id=batch_api_custom_id,
                auto_truncate=auto_truncate,
                **api_call_kwargs,
            )
            logger.info(f"New response length: {len(res['response'])} chars")
            if "API call failed" in res["response"]:
                res["response"] = trim_repeating_tail(original_response)
                if len(res["response"]) > self.max_output_chars:
                    logger.warning("No repeating tail found, cutting off response.")
                    res["response"] = res["response"][: self.max_output_chars]
                self.add_to_cache(input_hash, res)
            return res

        if len(res["response"]) < self.min_output_chars:
            logger.warning(
                f"Response length {len(res['response'])} is less than min output chars {self.min_output_chars}."
            )

        return res

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        """If model is changed, update the cost tracker model."""
        self._model = value
        # Fine-tuned models are formatted as ft:<model>:<organization>:<ft-job-info>
        # We remove the org/job info and just keep ft:<model>
        value = ":".join(value.split(":")[:2])
        self.cost_tracker.model = value

    def count_tokens(self, msg: str):
        model = self.cost_tracker.model.replace("ft:", "")
        if "gem" in model:
            return self.gemini_client.models.count_tokens(model=model, contents=msg)
        else:
            try:
                tokenizer = tiktoken.encoding_for_model(model)
                return len(tokenizer.encode(msg))
            except:
                raise ValueError(f"Model {self.model} not supported?")

    @property
    def max_context(self):
        try:
            self._max_context = self.cost_tracker.MAX_CONTEXT_WINDOWS[
                self.cost_tracker.model
            ]  # in tokens
        except:
            raise ValueError(f"Model {self.model} not supported")
        return self._max_context

    def build_image_messages(
        self, image_paths: list[Path] | dict[int, Path]
    ) -> list[dict]:
        """
        Builds image messages with index numbering for GPT Vision API call.
        image_paths can be a list, in which case they are numbered 1, 2, 3, ...
        or a dict mapping image index to Path, in which case the keys are used as the numbering.
        """
        if isinstance(image_paths, dict):
            index_image = True  # include index label if provided
            image_ids = sorted(image_paths.keys())
            image_paths = [image_paths[i] for i in sorted(image_paths.keys())]
        else:
            image_ids = list(range(1, len(image_paths) + 1))
            index_image = len(image_paths) > 1  # only include index if multiple images
        base64_images = [self.encode_image(image_path) for image_path in image_paths]

        messages = []
        for idx, base64_image in zip(image_ids, base64_images):
            # Add a textual label before each image
            if index_image:
                messages.append(
                    {"type": "input_text", "text": f"Page ID: {idx}. Image:"}
                )
            # Add the image message
            messages.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                }
            )
        return messages

    def encode_image(self, image_path: Path):
        """Image encode with back-off and retry."""
        for _ in range(10):  # Retry up to 10 times
            try:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to encode image {image_path}: {e}")
                sleep(10)
                # Is there anything else I can do to fix this?

    def get_token_count(
        self,
        messages: list[dict[str, str]],
        image_paths: list[Path] | None = None,
    ) -> int:
        """Get token count of messages."""
        token_count = 0
        for message in messages:
            if isinstance(message["content"], str):
                token_count += self.count_tokens(message["content"])
            elif isinstance(message["content"], list):
                for msg in message["content"]:
                    if isinstance(msg, str):
                        token_count += self.count_tokens(msg)
        if image_paths:
            for image_path in image_paths:
                for _ in range(5):  # Retry up to 5 times
                    try:
                        with Image.open(image_path) as img:
                            w, h = img.size
                            break
                    except Exception as e:
                        logger.warning(
                            f"Failed to open image {image_path}: {e}. "
                            "Defaulting to reasonable size, 1200x1600."
                        )
                        w, h = 1200, 1600  # reasonable default size if we can't open it
                token_count += count_image_tokens(width=w, height=h, model=self.model)
        return token_count

    def add_to_cache(self, input_hash: str, res: dict[str, str | list]):
        """Insert new API call results into cache."""
        self.cursor.execute(
            "INSERT OR REPLACE INTO gpt_cache (input_hash, response_data) VALUES (?, ?)",
            (input_hash, json.dumps(res)),
        )
        self.cache_count += 1

    def update_cache(self):
        """Commit changes to the database."""
        self.cost_tracker.log_cost()
        logger.info(f"Caching {self.cache_count} API calls, do not force-cancel run...")
        self.conn.commit()
        logger.info("Cache updated.")
        self.cache_count = 0

    def backup_cache(self):
        """Creates or overwrites a single backup of the database using SQLite's backup method."""
        backup_path = self.cache_path.with_suffix(".db.bak")
        # Create or overwrite the backup file
        logger.info(f"Backing up database to {backup_path}...")
        with sqlite3.connect(backup_path) as backup_conn:
            # Perform an efficient backup operation
            self.conn.backup(backup_conn)
        logger.info(f"Database backed up to {backup_path}")

    def load_from_cache(self, input_hash: str):
        """Retrieve response from database cache."""
        self.cursor.execute(
            "SELECT response_data FROM gpt_cache WHERE input_hash = ?", (input_hash,)
        )
        row = self.cursor.fetchone()
        return json.loads(row[0]) if row else None

    def api_input_to_hashable(
        self, messages: list[dict[str, str]], **kwargs
    ) -> tuple[tuple]:
        """Convert API input to hashable string for caching"""
        messages_tuples = [
            (message["role"], message["content"]) for message in messages
        ]
        payload = str(
            tuple([self.model] + list(sorted(kwargs.items())) + messages_tuples)
        )
        return "sha256:" + hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def add_cost_to_response(
        self,
        response: dict,
    ) -> dict[str, str | list]:
        """Adds cost information to the response."""
        response["cost"] = self.cost_tracker.calculate_cost(
            input_tokens=response["usage"]["input_tokens"],
            output_tokens=response["usage"]["output_tokens"],
            cached_tokens=response["usage"]["cached_tokens"],
            model=response["model"],  # defaults to self.model
        )
        return response

    def save_to_batch_file(
        self,
        messages: list[dict[str, str]],
        input_hash: str,
        custom_id: str,
        batch_filepath: Path,
        **kwargs,
    ) -> None:
        """Save messages to batch file for batch API call."""

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                **kwargs,
            },
        }

        custom_id_to_hash = {custom_id: input_hash}

        # File uploaded for batch API call
        with open(batch_filepath, "a") as f:
            f.write(json.dumps(request) + "\n")

        # File mapping custom_ids to hashes for saving to cache
        with open(batch_filepath.with_suffix(".hash"), "a") as f:
            f.write(json.dumps(custom_id_to_hash) + "\n")


class TokProbs(NamedTuple):
    tokens: list[str]
    probs: list[float]


def response_to_tokprobs(
    response: dict,
) -> Optional[list[TokProbs]]:
    """Converts GPT response to list of list of token-logprob tuples."""
    if (logprobs := response["output"][-1]["content"][0].get("logprobs")) is None:
        return None

    logprobs = [token_logprob["top_logprobs"] for token_logprob in logprobs]

    tokprobs = []
    for token_idx in logprobs:
        tokens, probs = [], []
        for token_logprobs in token_idx:
            try:
                tokens.append(token_logprobs.token)
                probs.append(np.exp(token_logprobs.logprob))
            except:
                tokens.append(token_logprobs["token"])
                probs.append(np.exp(token_logprobs["logprob"]))
        tokprobs.append(TokProbs(tokens=tokens, probs=probs))

    return tokprobs


def trim_repeating_tail(text, min_length=30):
    """
    Trims the text at the point where a repeated substring of at least `min_length` characters begins.

    Args:
        text (str): The input string to check.
        min_length (int): Minimum length of substring to consider for repetition.

    Returns:
        str: Trimmed string with repeated tail removed.
    """
    max_check_length = (
        len(text) // 2
    )  # no point checking substrings longer than half the text

    for length in range(min_length, max_check_length + 1):
        for start in range(len(text) - 2 * length + 1):
            substr = text[start : start + length]
            next_chunk = text[start + length : start + 2 * length]
            if substr == next_chunk:
                return text[: start + length]  # cut off after first occurrence

    return text  # return original if no repetition found
