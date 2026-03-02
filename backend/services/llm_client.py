"""
Async NVIDIA NIM client wrappers for triage-related model calls.
"""

import json
import os
from typing import Any

from openai import AsyncOpenAI

from models.llm_outputs import TriageSummaryResult
from services.prompts.capsule_prompt import CAPSULE_SYSTEM_PROMPT, build_capsule_user_prompt

_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_CHAT_MODEL = "meta/llama-3.1-70b-instruct"
_DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"


def _get_async_client() -> AsyncOpenAI:
    '''
    Purpose: Build an async OpenAI-compatible client configured for NVIDIA NIM.
    Parameters:
    None: Reads credentials and endpoint configuration from environment variables.
    Returns:
    AsyncOpenAI: Client bound to NVIDIA API key and base URL.
    Called by: backend/services/llm_client.py -> generate_triage_summary(), generate_embedding()
    Calls: openai.AsyncOpenAI()
    '''
    return AsyncOpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_BASE_URL", _DEFAULT_BASE_URL),
    )


async def generate_triage_summary(raw_text: str, metadata: dict[str, Any]) -> TriageSummaryResult:
    '''
    Purpose: Generate a structured triage summary JSON object from raw ticket evidence.
    Parameters:
    raw_text (str): Full ticket narrative and attached context to summarize.
    metadata (dict[str, Any]): Supplemental ticket fields used to ground generation.
    Returns:
    TriageSummaryResult: Parsed and validated structured capsule summary fields.
    Called by: backend/agents/capsule_generation.py -> generate_capsule_node()
    Calls: _get_async_client(), services.prompts.capsule_prompt.build_capsule_user_prompt()
    '''
    title = metadata.get("title", "Untitled Failure")
    user_prompt = build_capsule_user_prompt(title=title, raw_text=raw_text, metadata=metadata)
    model_id = os.getenv("NVIDIA_MODEL_ID", _DEFAULT_CHAT_MODEL)
    client = _get_async_client()

    response = await client.chat.completions.create(
        model=model_id,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CAPSULE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return TriageSummaryResult.model_validate(json.loads(content))


async def generate_embedding(text: str) -> list[float]:
    '''
    Purpose: Generate a semantic embedding vector for retrieval/indexing workflows.
    Parameters:
    text (str): Input text that represents the failure capsule semantics.
    Returns:
    list[float]: Dense embedding vector from NVIDIA embedding endpoint.
    Called by: backend/agents/capsule_generation.py -> generate_capsule_node()
    Calls: _get_async_client(), AsyncOpenAI.embeddings.create()
    '''
    model_id = os.getenv("NVIDIA_EMBEDDING_MODEL_ID", _DEFAULT_EMBEDDING_MODEL)
    client = _get_async_client()
    response = await client.embeddings.create(model=model_id, input=text)
    return list(response.data[0].embedding)
