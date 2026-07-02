from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import IFD, TAGS

from .schema import normalize_caption


_CONTENT_KEYS = {
    "text",
    "prompt",
    "value",
    "string",
    "positive",
    "text_g",
    "text_l",
    "wildcard_text",
    "populated_text",
}
_MIN_TEXT_LEN = 20
_EXIF_CANDIDATE_LIMIT = 5
_SKIP_NAME_HINTS = ("system", "negative")
_UTILITY_CLASSES = {
    "jsonextractstring",
    "jsonextract",
    "stringreplace",
    "previewany",
    "previewtext",
    "showtext",
    "string",
    "stringconcatenate",
    "note",
    "markdownnote",
    "comfymathexpression",
    "customcombo",
}
_CAPTION_KEYS = {"high_level_description", "style_description", "compositional_deconstruction"}
_PROMPT_BUILDER_CLASSES = {"ideogram4promptbuilderkj", "ideogram4promptbuilder"}
_A1111_SETTING_KEYS = (
    "Steps",
    "Sampler",
    "Schedule type",
    "CFG scale",
    "Seed",
    "Size",
    "Model hash",
    "Model",
    "Version",
)


def _decode_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
    else:
        return None
    text = text.strip()
    return text or None


def _decode_user_comment(raw: Any) -> str | None:
    if isinstance(raw, str):
        return raw.strip() or None
    if not isinstance(raw, bytes) or not raw:
        return None

    if raw[:8] == b"UNICODE\x00":
        body = raw[8:]
        for encoding in ("utf-16-be", "utf-16", "utf-16-le"):
            try:
                return body.decode(encoding).rstrip("\x00").strip() or None
            except UnicodeDecodeError:
                continue
    if raw[:8] == b"ASCII\x00\x00\x00":
        return raw[8:].decode("ascii", "replace").rstrip("\x00").strip() or None
    return raw.decode("utf-8", "replace").rstrip("\x00").strip() or None


def _collect_raw_fields(path: str | Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    with Image.open(path) as image:
        for key, value in image.info.items():
            if key.lower() in {"exif", "icc_profile"}:
                continue
            text = _decode_text(value)
            if text:
                fields[f"info:{key}"] = text

        exif = image.getexif()
        for tag, value in exif.items():
            name = TAGS.get(tag, str(tag))
            text = _decode_user_comment(value) if name == "UserComment" else _decode_text(value)
            if text:
                fields[f"exif:{name}"] = text

        try:
            exif_ifd = exif.get_ifd(IFD.Exif)
        except Exception:
            exif_ifd = {}
        for tag, value in exif_ifd.items():
            name = TAGS.get(tag, str(tag))
            text = _decode_user_comment(value) if name == "UserComment" else _decode_text(value)
            if text:
                fields[f"exif:{name}"] = text

    return fields


def _load_comfy_prompt(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    for prefix in ("Prompt:", "Workflow:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    has_api_nodes = any(
        isinstance(value, dict) and "class_type" in value and "inputs" in value for value in data.values()
    )
    has_ui_nodes = isinstance(data.get("nodes"), list)
    if has_api_nodes or has_ui_nodes:
        return data
    return None


def _node_texts_from_api_prompt(prompt: dict[str, Any]) -> list[tuple[str, str]]:
    nodes: list[tuple[str, str]] = []
    seen: set[str] = set()
    for node in prompt.values():
        if not isinstance(node, dict):
            continue

        class_type = node.get("class_type", "")
        if str(class_type).lower() in _UTILITY_CLASSES:
            continue

        title = (node.get("_meta") or {}).get("title", "")
        name = f"{class_type} {title}".lower()
        if "text" not in name and "prompt" not in name:
            continue
        if any(hint in name for hint in _SKIP_NAME_HINTS):
            continue

        display_name = title or class_type or "node"
        for key, value in (node.get("inputs") or {}).items():
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text or text in seen:
                continue
            is_content = key.lower() in _CONTENT_KEYS or "\n" in text or len(text) >= _MIN_TEXT_LEN
            if is_content:
                seen.add(text)
                nodes.append((str(display_name), text))
    return nodes


def _node_texts_from_ui_workflow(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    nodes: list[tuple[str, str]] = []
    seen: set[str] = set()
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue

        node_type = node.get("type", "")
        title = node.get("title", "")
        name = f"{node_type} {title}".lower()
        if "text" not in name and "prompt" not in name:
            continue
        if any(hint in name for hint in _SKIP_NAME_HINTS):
            continue

        display_name = title or node_type or "node"
        widgets = node.get("widgets_values", [])
        if not isinstance(widgets, list):
            continue
        for value in widgets:
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text or text in seen:
                continue
            if "\n" in text or len(text) >= _MIN_TEXT_LEN:
                seen.add(text)
                nodes.append((str(display_name), text))
    return nodes


def _extract_comfy_nodes(prompt: dict[str, Any]) -> list[tuple[str, str]]:
    return [*_node_texts_from_api_prompt(prompt), *_node_texts_from_ui_workflow(prompt)]


def _caption_from_prompt_builder(node: dict[str, Any]) -> dict[str, Any] | None:
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None

    elements: list[dict[str, Any]] = []
    elements_raw = inputs.get("elements_data")
    if isinstance(elements_raw, str) and elements_raw.strip():
        try:
            parsed_elements = json.loads(elements_raw)
        except json.JSONDecodeError:
            parsed_elements = []
        if isinstance(parsed_elements, list):
            for item in parsed_elements:
                if not isinstance(item, dict):
                    continue
                element: dict[str, Any] = {"type": "text" if item.get("type") == "text" else "obj"}
                try:
                    x = float(item.get("x", 0))
                    y = float(item.get("y", 0))
                    w = float(item.get("w", 0))
                    h = float(item.get("h", 0))
                except (TypeError, ValueError):
                    pass
                else:
                    element["bbox"] = [
                        round(y * 1000),
                        round(x * 1000),
                        round((y + h) * 1000),
                        round((x + w) * 1000),
                    ]
                if element["type"] == "text":
                    element["text"] = str(item.get("text", "") or "")
                element["desc"] = str(item.get("desc", "") or "")
                palette = item.get("palette")
                if isinstance(palette, list):
                    element["color_palette"] = palette
                elements.append(element)

    mode = str(inputs.get("style", "photo") or "photo").strip()
    style: dict[str, Any] = {
        "aesthetics": str(inputs.get("aesthetics", "") or ""),
        "lighting": str(inputs.get("lighting", "") or ""),
        "medium": str(inputs.get("medium", "") or ("photograph" if mode == "photo" else "illustration")),
    }
    if mode == "photo":
        style["photo"] = str(inputs.get("style.photo", inputs.get("photo", "")) or "")
    else:
        style["art_style"] = str(inputs.get("style.art_style", inputs.get("art_style", "")) or "")

    palette_raw = inputs.get("style_palette_data")
    if isinstance(palette_raw, str) and palette_raw.strip():
        try:
            style["color_palette"] = json.loads(palette_raw)
        except json.JSONDecodeError:
            pass

    caption = {
        "high_level_description": str(inputs.get("high_level_description", "") or ""),
        "style_description": style,
        "compositional_deconstruction": {
            "background": str(inputs.get("background", "") or ""),
            "elements": elements,
        },
    }
    normalized = normalize_caption(caption)
    if (
        normalized["high_level_description"].strip()
        or normalized["compositional_deconstruction"]["background"].strip()
        or normalized["compositional_deconstruction"]["elements"]
    ):
        return normalized
    return None


def _caption_from_comfy_prompt(prompt: dict[str, Any]) -> dict[str, Any] | None:
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", "") or "")
        title = str((node.get("_meta") or {}).get("title", "") or "")
        class_key = class_type.lower().replace(" ", "")
        title_key = title.lower().replace(" ", "")
        if class_key in _PROMPT_BUILDER_CLASSES or title_key in _PROMPT_BUILDER_CLASSES:
            caption = _caption_from_prompt_builder(node)
            if caption is not None:
                return caption
    return None


def extract_workflow_text_nodes(path: str | Path) -> list[tuple[str, str]]:
    """Return (node_name, text) pairs discovered in embedded ComfyUI metadata."""
    try:
        fields = _collect_raw_fields(path)
    except Exception:
        return []

    nodes: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in fields.values():
        prompt = _load_comfy_prompt(value)
        if not prompt:
            continue
        for name, text in _extract_comfy_nodes(prompt):
            if text not in seen:
                seen.add(text)
                nodes.append((name, text))
    return nodes


def workflow_text_candidates(path: str | Path, *, limit: int = _EXIF_CANDIDATE_LIMIT) -> list[str]:
    """Return the longest unique workflow text strings, up to ``limit``."""
    nodes = extract_workflow_text_nodes(path)
    ordered = sorted(nodes, key=lambda item: len(item[1]), reverse=True)
    return [text for _name, text in ordered[:limit]]


def _metadata_text_candidates(path: str | Path, *, limit: int = _EXIF_CANDIDATE_LIMIT) -> list[str]:
    try:
        values = list(_collect_raw_fields(path).values())
    except Exception:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    ordered = sorted(candidates, key=len, reverse=True)
    return ordered[:limit]


def _try_parse_caption_json(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not any(key in parsed for key in _CAPTION_KEYS):
        return None
    return normalize_caption(parsed)


def try_import_caption_from_exif(path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    """Try to import structured caption data from image metadata."""
    try:
        fields = _collect_raw_fields(path)
    except Exception:
        fields = {}

    for value in fields.values():
        prompt = _load_comfy_prompt(value)
        if not prompt:
            continue
        caption = _caption_from_comfy_prompt(prompt)
        if caption is not None:
            return caption, "Imported Ideogram caption fields from image metadata; click Save to persist."

    candidates: list[str] = []
    seen: set[str] = set()
    for text in [*workflow_text_candidates(path), *_metadata_text_candidates(path)]:
        if text not in seen:
            seen.add(text)
            candidates.append(text)

    for text in candidates:
        caption = _try_parse_caption_json(text)
        if caption is not None:
            return caption, "Imported caption JSON from image metadata; click Save to persist."
    return None, None


def _looks_like_json_text(text: str) -> bool:
    text = text.strip()
    return bool(text) and (text[0] in "[{" or text.startswith("```"))


def _looks_like_a1111_parameters(text: str) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return False
    settings_line = lines[-1]
    return sum(1 for key in _A1111_SETTING_KEYS if f"{key}:" in settings_line) >= 2


def _strip_a1111_generation_parameters(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    lines = text.splitlines()
    prompt_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Negative prompt:"):
            break
        if prompt_lines and sum(1 for key in _A1111_SETTING_KEYS if f"{key}:" in stripped) >= 2:
            break
        prompt_lines.append(line)

    prompt = "\n".join(prompt_lines).strip()
    if prompt.startswith("Prompt:"):
        prompt = prompt[len("Prompt:") :].strip()
    return prompt or None


def _stable_diffusion_prompt_candidates(path: str | Path) -> list[str]:
    try:
        fields = _collect_raw_fields(path)
    except Exception:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for key, value in fields.items():
        key_name = key.split(":", 1)[-1].lower()
        text = value.strip()
        prompt: str | None = None
        if key_name in {"parameters", "usercomment", "imagedescription", "comment"} or _looks_like_a1111_parameters(text):
            prompt = _strip_a1111_generation_parameters(text)
        elif "prompt" in key_name and not _looks_like_json_text(text):
            prompt = text

        if prompt and len(prompt) >= _MIN_TEXT_LEN and prompt not in seen and not _looks_like_json_text(prompt):
            seen.add(prompt)
            candidates.append(prompt)
    return candidates


def try_import_prompt_text_from_exif(path: str | Path) -> tuple[str | None, str | None]:
    """Try to import plain generation prompt text from image metadata."""
    for text in [*_stable_diffusion_prompt_candidates(path), *workflow_text_candidates(path)]:
        text = text.strip()
        if len(text) >= _MIN_TEXT_LEN and not _looks_like_json_text(text):
            return text, "Imported prompt text from image metadata; click Save to persist as text caption."
    return None, None
