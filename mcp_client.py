from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_MODEL = "gpt-4o"


SYSTEM_PROMPT = """You are an autonomous game-playing agent controlling March7th Assistant through MCP tools.

Goal:
- Complete the user's task in the game.
- Work in a loop: observe the screen, reason briefly, call exactly the tools needed, then observe again.
- Prefer capture_screen first when visual state is unknown.
- Use OCR/find_text/click_text when text is important.
- Use screenshot coordinates unless a tool asks for another coordinate mode.
- Tool duration parameters are seconds, not milliseconds. For mouse_drag, use small values such as 0.25 to 1.2.
- Keep actions small and verify each important change with another observation.
- A drag action is not completion. After every drag, call capture_screen and inspect whether the item landed in the intended slot.
- If the user asks to complete the current game task, continue until the game UI clearly shows completion, reward, success, or no remaining required action.
- If a dragged item is in the wrong place, correct it with another small drag instead of stopping.
- If input does not seem to work, call get_input_diagnostics and focus_game before trying again.
- Never claim completion until the screen or tool output confirms it.
- Do not output hidden reasoning tags such as <|channel>thought. Use normal assistant text only.

When the task is complete, do not call more tools. Reply with:
DONE: <short completion summary>

If the task cannot be completed, reply with:
FAILED: <short reason and what was verified>
"""


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    timeout: float


@dataclass(frozen=True)
class ClientConfig:
    mcp_url: str
    max_steps: int
    max_tool_rounds: int
    keep_images: int
    final_check: bool
    show_messages: bool
    openai: OpenAIConfig


FINAL_MARKER_RE = re.compile(r"(DONE|FAILED)\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to call OpenAI-compatible API: {exc}") from exc
    return json.loads(body)


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON schema that is acceptable as an OpenAI-compatible tool schema."""
    if not schema:
        return {"type": "object", "properties": {}}
    compact = dict(schema)
    compact.setdefault("type", "object")
    compact.setdefault("properties", {})
    return compact


def _mcp_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    openai_tools = []
    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _compact_schema(tool.inputSchema),
                },
            }
        )
    return openai_tools


def _content_to_plain(content: Any) -> Any:
    if hasattr(content, "model_dump"):
        return content.model_dump(by_alias=True, exclude_none=True)
    if isinstance(content, dict):
        return content
    return {"type": type(content).__name__, "value": str(content)}


def _parse_text_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _extract_images_and_text(tool_name: str, result: Any) -> tuple[str, list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []
    text_parts: list[Any] = []

    for content in getattr(result, "content", []) or []:
        item = _content_to_plain(content)
        item_type = item.get("type")

        if item_type == "image" and item.get("data"):
            mime_type = item.get("mimeType") or "image/png"
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{item['data']}"},
                }
            )
            text_parts.append({"type": "image", "mimeType": mime_type, "note": "Image attached separately."})
            continue

        if item_type == "text" and isinstance(item.get("text"), str):
            parsed = _parse_text_json(item["text"])
            if isinstance(parsed, dict) and parsed.get("image_base64"):
                mime_type = f"image/{parsed.get('format', 'png')}"
                images.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{parsed['image_base64']}"},
                    }
                )
                parsed = dict(parsed)
                parsed["image_base64"] = "<attached separately>"
            text_parts.append(parsed)
            continue

        text_parts.append(item)

    payload = {
        "tool": tool_name,
        "is_error": bool(getattr(result, "isError", False)),
        "content": text_parts,
    }
    return json.dumps(payload, ensure_ascii=False), images


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(texts)
    return ""


def _clean_assistant_text(text: str) -> str:
    text = re.sub(r"<\|channel>?\s*thought", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|/?channel\|>", "", text)
    text = re.sub(r"<\|channel\|>\s*thought", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = text.replace("<channel|>", "")
    return text.strip()


def _extract_final_marker(text: str) -> str | None:
    cleaned = _clean_assistant_text(text)
    match = FINAL_MARKER_RE.search(cleaned)
    if not match:
        return None
    marker = match.group(1).upper()
    summary = " ".join(match.group(2).strip().split())
    return f"{marker}: {summary}"


def _trim_image_history(messages: list[dict[str, Any]], keep_images: int) -> None:
    if keep_images < 0:
        return

    image_message_indexes = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            image_message_indexes.append(index)

    for index in image_message_indexes[: max(0, len(image_message_indexes) - keep_images)]:
        content = messages[index].get("content")
        if not isinstance(content, list):
            continue
        text = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        )
        messages[index]["content"] = f"{text}\n[Older screenshot omitted to keep context small.]".strip()


def _normalize_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments are not valid JSON: {raw_arguments}") from exc
        if parsed is None:
            return {}
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Tool arguments must be a JSON object, got: {raw_arguments!r}")


def _coerce_duration_seconds(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if number > 10:
        # Models often emit UI durations in milliseconds. MCP tools expect seconds.
        number = number / 1000.0

    return max(minimum, min(maximum, number))


def _sanitize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(arguments)

    if name == "mouse_drag":
        sanitized["duration"] = _coerce_duration_seconds(
            sanitized.get("duration"),
            default=0.35,
            minimum=0.05,
            maximum=3.0,
        )
        sanitized["hold_before"] = _coerce_duration_seconds(
            sanitized.get("hold_before"),
            default=0.25,
            minimum=0.0,
            maximum=2.0,
        )
        sanitized["hold_after"] = _coerce_duration_seconds(
            sanitized.get("hold_after"),
            default=0.15,
            minimum=0.0,
            maximum=2.0,
        )
        if sanitized.get("button") not in (None, "left", "right", "middle"):
            sanitized["button"] = "left"

    elif name == "press_key":
        sanitized["duration"] = _coerce_duration_seconds(
            sanitized.get("duration"),
            default=0.2,
            minimum=0.02,
            maximum=3.0,
        )

    elif name == "wait":
        sanitized["seconds"] = _coerce_duration_seconds(
            sanitized.get("seconds"),
            default=1.0,
            minimum=0.0,
            maximum=15.0,
        )

    return sanitized


def _assistant_message_from_choice(choice: dict[str, Any]) -> dict[str, Any]:
    message = choice.get("message") or {}
    normalized = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        normalized["tool_calls"] = message["tool_calls"]
    return normalized


def _call_chat_completion(
    config: OpenAIConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "tools": tools,
        "tool_choice": "auto",
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    url = _join_url(config.base_url, "/chat/completions")
    response = _post_json(url, headers, payload, config.timeout)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible API returned no choices: {response}")
    return choices[0]


async def run_agent(task: str, config: ClientConfig) -> str:
    async with streamablehttp_client(config.mcp_url, timeout=timedelta(seconds=30)) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_result = await session.list_tools()
            tools = _mcp_tools_to_openai(tool_result.tools)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]
            final_check_pending = False
            final_check_tools_seen = 0
            idle_turns = 0

            for step in range(1, config.max_steps + 1):
                print(f"\n=== Step {step}/{config.max_steps} ===", flush=True)
                _trim_image_history(messages, config.keep_images)
                choice = _call_chat_completion(config.openai, messages, tools)
                assistant_message = _assistant_message_from_choice(choice)
                messages.append(assistant_message)

                assistant_text = _clean_assistant_text(_message_text(assistant_message))
                if assistant_text and config.show_messages:
                    print(assistant_text, flush=True)

                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls:
                    final_marker = _extract_final_marker(assistant_text)
                    if final_marker:
                        if not config.final_check:
                            return final_marker
                        if final_check_pending and final_check_tools_seen > 0:
                            return final_marker

                        final_check_pending = True
                        final_check_tools_seen = 0
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Before finishing, verify the claim with tools. "
                                    "Call capture_screen and ocr_screen now. If the screen does not clearly show "
                                    "completion/success/reward/no remaining action, continue the game task. "
                                    "Only after that verification may you reply DONE or FAILED."
                                ),
                            }
                        )
                        continue

                    idle_turns += 1
                    if idle_turns <= 2:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You did not call a tool or provide DONE/FAILED. Continue by calling exactly one "
                                    "useful MCP tool, usually capture_screen or ocr_screen."
                                ),
                            }
                        )
                        continue
                    return f"STOPPED: model returned no tool call and no DONE/FAILED marker.\n{assistant_text}"

                idle_turns = 0

                pending_image_messages: list[dict[str, Any]] = []
                skipped_tools = 0

                for round_index, tool_call in enumerate(tool_calls, start=1):
                    function = tool_call.get("function") or {}
                    name = function.get("name")
                    if not name:
                        raise RuntimeError(f"Tool call has no function name: {tool_call}")

                    if round_index > config.max_tool_rounds:
                        skipped_tools += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", f"call_{step}_{round_index}"),
                                "name": name,
                                "content": json.dumps(
                                    {
                                        "tool": name,
                                        "is_error": True,
                                        "content": (
                                            "Skipped because max_tool_rounds was reached. "
                                            "Call fewer tools in the next step."
                                        ),
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue

                    arguments = _sanitize_tool_arguments(name, _normalize_tool_arguments(function.get("arguments")))

                    print(f"Calling tool {round_index}: {name}({json.dumps(arguments, ensure_ascii=False)})", flush=True)
                    result = await session.call_tool(name, arguments)
                    if final_check_pending:
                        final_check_tools_seen += 1
                    tool_text, images = _extract_images_and_text(name, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", f"call_{step}_{round_index}"),
                            "name": name,
                            "content": tool_text,
                        }
                    )

                    if images:
                        pending_image_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Latest visual observation from MCP tool `{name}`. "
                                            "Use this image to decide the next action."
                                        ),
                                    },
                                    *images,
                                ],
                            }
                        )

                    if config.show_messages:
                        print(tool_text[:1000], flush=True)

                messages.extend(pending_image_messages)

                if skipped_tools:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{skipped_tools} tool calls were skipped after max_tool_rounds={config.max_tool_rounds}. "
                                "Continue with the next best action."
                            ),
                        }
                    )

            return f"STOPPED: reached max_steps={config.max_steps} before DONE/FAILED."


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-compatible MCP client for March7th Assistant.")
    parser.add_argument("task", nargs="*", help="Task to ask the AI agent to complete.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI-compatible API key.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL), help="API base URL.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="Model name.")
    parser.add_argument("--mcp-url", default=os.getenv("M7A_MCP_URL", DEFAULT_MCP_URL), help="March7th MCP URL.")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("OPENAI_TEMPERATURE", "0.2")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("OPENAI_TIMEOUT", "120")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("M7A_AGENT_MAX_STEPS", "80")))
    parser.add_argument("--max-tool-rounds", type=int, default=int(os.getenv("M7A_AGENT_MAX_TOOL_ROUNDS", "8")))
    parser.add_argument("--keep-images", type=int, default=int(os.getenv("M7A_AGENT_KEEP_IMAGES", "4")))
    parser.add_argument(
        "--no-final-check",
        action="store_true",
        help="Allow the model to finish immediately when it says DONE/FAILED.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide assistant/tool detail logs.")
    return parser


def _read_task_from_stdin() -> str:
    print("Enter task, then press Ctrl+Z and Enter on Windows, or Ctrl+D on Unix:", file=sys.stderr)
    return sys.stdin.read().strip()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    task = " ".join(args.task).strip() or _read_task_from_stdin()
    if not task:
        parser.error("task is required")
    if not args.api_key:
        parser.error("--api-key or OPENAI_API_KEY is required")

    config = ClientConfig(
        mcp_url=args.mcp_url,
        max_steps=max(1, args.max_steps),
        max_tool_rounds=max(1, args.max_tool_rounds),
        keep_images=max(0, args.keep_images),
        final_check=not args.no_final_check,
        show_messages=not args.quiet,
        openai=OpenAIConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
        ),
    )

    started_at = time.time()
    try:
        result = asyncio.run(run_agent(task, config))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n=== Result ===")
    print(result)
    print(f"Elapsed: {time.time() - started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
