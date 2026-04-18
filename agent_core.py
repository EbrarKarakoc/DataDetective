"""LLM agent core with tool-calling support for DataDetective.

This module connects to a local LM Studio-compatible OpenAI endpoint and
implements a robust tool-calling loop for:
- automated EDA
- hypothesis suggestion
- baseline model recommendation/training
"""

from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any

import pandas as pd
from openai import OpenAI

from tools_engine import recommend_and_train_model, run_automated_eda, suggest_hypothesis


def _build_tools_schema() -> list[dict[str, Any]]:
    """Return OpenAI tool schemas compatible with LM Studio function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "eda",
                "description": (
                    "Yuklenen veri seti icin otomatik EDA calistirir; veri tipleri, "
                    "eksik degerler ve sayisal korelasyon bilgisi dondurur."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hipotez",
                "description": (
                    "Verilen hedef degisken adina gore 2-3 mantiksal hipotez onerir."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_col": {
                            "type": "string",
                            "description": "Hedef degiskenin kolon adi.",
                        }
                    },
                    "required": ["target_col"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "model_egit",
                "description": (
                    "Hedef degiskene gore siniflandirma veya regresyon modeli secip "
                    "Random Forest ile egitir ve skor dondurur."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_col": {
                            "type": "string",
                            "description": "Egitim icin hedef degiskenin kolon adi.",
                        }
                    },
                    "required": ["target_col"],
                },
            },
        },
    ]


def _dispatch_tool(tool_name: str, arguments: dict[str, Any], df: pd.DataFrame) -> Any:
    """Execute mapped Python function for a tool call."""
    if tool_name == "eda":
        return run_automated_eda(df)
    if tool_name == "hipotez":
        return suggest_hypothesis(target_col=arguments["target_col"])
    if tool_name == "model_egit":
        return recommend_and_train_model(df=df, target_col=arguments["target_col"])
    raise ValueError(f"Unknown tool name: {tool_name}")


def run_agent(
    user_message: str,
    df: pd.DataFrame,
    model: str = "google/gemma-3-4b",
    api_key: str = "lm-studio",
    max_iterations: int = 6,
) -> str:
    """Run a full OpenAI-style tool-calling loop against LM Studio.

    The function:
    1) sends user prompt and tool schemas to the model,
    2) executes requested tools via dispatcher,
    3) feeds tool outputs back to the model,
    4) returns final assistant text when tool calls end.

    Args:
        user_message: Natural language user request.
        df: Active pandas DataFrame that tools will operate on.
        model: LM Studio model id (e.g. "qwen2.5-7b-instruct").
        api_key: API key placeholder for OpenAI SDK. LM Studio accepts any string.
        max_iterations: Safety limit for tool-calling loop.

    Returns:
        Final assistant response text.

    Raises:
        ValueError: If input message/data is invalid or no final response is produced.
        RuntimeError: If loop exceeds max_iterations.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be a non-empty string.")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")

    client = OpenAI(base_url="http://localhost:1234/v1", api_key=api_key)
    selected_model = model or os.getenv("LM_STUDIO_MODEL", "google/gemma-3-4b")
    tools = _build_tools_schema()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Sen DataDetective asistanisin. Gerekliyse araç kullan, degilse direkt "
                "yanit ver. Arac sonucunu analiz edip kisa ve acik bir yanit üret."
            ),
        },
        {"role": "user", "content": user_message.strip()},
    ]

    for _ in range(max_iterations):
        completion = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = completion.choices[0].message

        # Convert assistant message to plain dict for next turn context.
        messages.append(assistant_message.model_dump(exclude_none=True))

        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            return assistant_message.content or ""

        for tc in tool_calls:
            tool_name = tc.function.name

            # Local models may emit malformed JSON; guard with try-except.
            try:
                raw_args = tc.function.arguments or "{}"
                parsed_args = json.loads(raw_args)
                if not isinstance(parsed_args, dict):
                    raise ValueError("Tool arguments must decode to a JSON object.")
            except JSONDecodeError as exc:
                tool_result: Any = {
                    "error": "JSONDecodeError",
                    "message": "Tool argumanlari gecerli JSON degil.",
                    "raw_arguments": tc.function.arguments,
                    "details": str(exc),
                }
            except Exception as exc:  # noqa: BLE001
                tool_result = {
                    "error": "InvalidArguments",
                    "message": str(exc),
                    "raw_arguments": tc.function.arguments,
                }
            else:
                try:
                    tool_result = _dispatch_tool(tool_name, parsed_args, df)
                except Exception as exc:  # noqa: BLE001
                    tool_result = {
                        "error": "ToolExecutionError",
                        "tool_name": tool_name,
                        "message": str(exc),
                    }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                }
            )

    raise RuntimeError(
        "Tool calling loop max_iterations limitine ulasti. Yanit tamamlanamadi."
    )
