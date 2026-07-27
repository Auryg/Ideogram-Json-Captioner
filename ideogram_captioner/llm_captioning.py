from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import ExifTags, Image

from .schema import normalize_bbox, normalize_caption


ProgressCallback = Callable[[str], None]

BBOX_BACKEND_VLM = "vlm"
BBOX_BACKEND_YOLOE26 = "yoloe26"
BBOX_BACKENDS = {BBOX_BACKEND_VLM, BBOX_BACKEND_YOLOE26}
PLAIN_CAPTION_MODE_STANDARD = "standard"
PLAIN_CAPTION_MODE_PERSON = "person"
PLAIN_CAPTION_MODES = {PLAIN_CAPTION_MODE_STANDARD, PLAIN_CAPTION_MODE_PERSON}
DEFAULT_YOLOE26_BBOX_MODEL = "yoloe-26l-seg.pt"
DEFAULT_ULTRALYTICS_BBOX_CONFIDENCE = 0.25
DEFAULT_YOLO_BBOX_IMGSZ = 1024


class AutoCaptionError(RuntimeError):
    """Raised for user-actionable captioning setup or model failures."""


class ModelJsonError(AutoCaptionError):
    """Raised when a model response cannot be parsed as the expected JSON object."""

    def __init__(
        self,
        message: str,
        raw_output: str = "",
        candidate: str = "",
        repair_output: str = "",
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.candidate = candidate
        self.repair_output = repair_output


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    api_model: str
    kind: str = "hf"
    hf_repo: str = ""
    mmproj_repo: str = ""
    model_filename: str = ""
    mmproj_filename: str = ""
    local_model_path: str = ""
    local_mmproj_path: str = ""


@dataclass(frozen=True)
class ModelRuntimeConfig:
    label: str
    api_model: str
    kind: str = "hf"
    hf_repo: str = ""
    mmproj_repo: str = ""
    model_filename: str = ""
    mmproj_filename: str = ""
    local_model_path: str = ""
    local_mmproj_path: str = ""


@dataclass(frozen=True)
class ModelAssets:
    model_path: Path | None = None
    mmproj_path: Path | None = None


@dataclass(frozen=True)
class YoloDetection:
    index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]


DEFAULT_JSON_REFINE_INSTRUCTIONS = """
Improve the existing structured JSON caption while preserving the image's actual content.
Add useful detail where the current JSON is vague, but do not invent subjects, text, brands,
or identities that are not visible in the image.
""".strip()


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    return app_base_dir() / "models"


def default_settings_path() -> Path:
    return app_base_dir() / "captioner_settings.json"


def default_profiles_path() -> Path:
    return app_base_dir() / "captioner_model_profiles.json"


def default_profiles_example_path() -> Path:
    return app_base_dir() / "captioner_model_profiles.example.json"


def default_prompts_path() -> Path:
    return app_base_dir() / "captioner_prompts"


def find_llama_server() -> Path | None:
    candidates: list[Path] = []
    base = app_base_dir()
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    candidates.extend(
        [
            base / executable,
            base / "tools" / executable,
            base / "llama.cpp" / executable,
            base / "llama.cpp" / "build" / "bin" / executable,
            base / "llama.cpp-cuda" / executable,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


DEFAULT_PROFILE_DATA: dict[str, Any] = {
    "profiles": [
        {
            "id": "unsloth-qwen25vl-7b-q4",
            "label": "Download: Qwen2.5-VL 7B Q4 (recommended)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "qwen25vl",
            "hf_repo": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "model_filename": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-BF16.gguf",
        },
        {
            "id": "unsloth-qwen3vl-30b-q4",
            "label": "Download: Unsloth Qwen3-VL 30B Q4",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "unsloth-qwen3vl-30b",
            "hf_repo": "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF",
            "model_filename": "Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-BF16.gguf",
        },
        {
            "id": "hauhaucs-qwen35-9b-aggressive-q6k",
            "label": "Download: Qwen3.5-9B Uncensored HauhauCS Aggressive Q6_K (7GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "hauhaucs-qwen35-9b",
            "hf_repo": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
            "model_filename": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf",
            "mmproj_filename": "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        },
        {
            "id": "hauhaucs-gemma4-26b-balanced-q4km",
            "label": "Download: Gemma4-26B A4B Uncensored HauhauCS-Balanced Q4_K_M (17GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "gemma4-26b-balanced",
            "hf_repo": "HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced",
            "model_filename": "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
            "mmproj_filename": "mmproj-Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-f16.gguf",
        },
        {
            "id": "huihui-qwen3vl-30b-abliterated-i1-q4ks",
            "label": "Download: Huihui Qwen3-VL 30B abliterated i1 Q4_K_S (17GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "huihui-qwen3vl-30b",
            "hf_repo": "mradermacher/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-i1-GGUF",
            "mmproj_repo": "mradermacher/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-GGUF",
            "model_filename": "Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated.i1-Q4_K_S.gguf",
            "mmproj_filename": "Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated.mmproj-f16.gguf",
        },
        {
            "id": "davidau-qwen36-27b-heretic-q6k",
            "label": "Download: DavidAU Qwen3.6 27B Heretic Q6_K (22GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "davidau-qwen36-27b-heretic",
            "hf_repo": "DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF",
            "model_filename": "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q6_K.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
        },
        {
            "id": "local-model",
            "label": "Existing server: discover or enter model ID",
            "tasks": ["caption", "bbox"],
            "kind": "server",
            "api_model": "",
        },
        {
            "id": "server-qwen3vl",
            "label": "Existing server model ID: qwen3vl",
            "tasks": ["caption", "bbox"],
            "kind": "server",
            "api_model": "qwen3vl",
        },
        {
            "id": "server-gemma-vl",
            "label": "Existing server model ID: gemma-vl",
            "tasks": ["caption"],
            "kind": "server",
            "api_model": "gemma-vl",
        },
    ]
}

CUSTOM_HF_PROFILE = ModelProfile("custom-hf", "Custom Hugging Face GGUF", "", kind="custom_hf")
CUSTOM_LOCAL_PROFILE = ModelProfile("custom-local", "Custom local GGUF files", "local-model", kind="custom_local")


def _profile_from_dict(raw: dict[str, Any]) -> ModelProfile | None:
    profile_id = str(raw.get("id", "")).strip()
    label = str(raw.get("label", "")).strip()
    if not profile_id or not label:
        return None

    kind = str(raw.get("kind", "")).strip().lower()
    if not kind:
        kind = "hf" if raw.get("hf_repo") else "server"
    if kind not in {"hf", "server", "local"}:
        return None

    api_model = str(raw.get("api_model", "")).strip()
    if kind == "server" and label.startswith("Existing server alias: "):
        if profile_id == "local-model":
            label = "Existing server: discover or enter model ID"
        else:
            label = label.replace("Existing server alias: ", "Existing server model ID: ", 1)
    if profile_id == "local-model" and kind == "server":
        if api_model == "local-model":
            api_model = ""

    return ModelProfile(
        id=profile_id,
        label=label,
        api_model=api_model,
        kind=kind,
        hf_repo=str(raw.get("hf_repo", "")).strip(),
        mmproj_repo=str(raw.get("mmproj_repo", "")).strip(),
        model_filename=str(raw.get("model_filename", "")).strip(),
        mmproj_filename=str(raw.get("mmproj_filename", "")).strip(),
        local_model_path=str(raw.get("local_model_path", "")).strip(),
        local_mmproj_path=str(raw.get("local_mmproj_path", "")).strip(),
    )


def _profile_tasks(raw: dict[str, Any]) -> set[str]:
    tasks = raw.get("tasks", ["caption", "bbox"])
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list):
        return {"caption", "bbox"}
    out = {str(task).strip().lower() for task in tasks}
    if "all" in out:
        return {"caption", "bbox"}
    return {task for task in out if task in {"caption", "bbox"}}


def _read_profile_data(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict) and isinstance(loaded.get("profiles"), list):
        return loaded
    return None


def profile_seed_data() -> dict[str, Any]:
    return _read_profile_data(default_profiles_example_path()) or DEFAULT_PROFILE_DATA


def load_model_profiles(path: Path | None = None) -> dict[str, tuple[ModelProfile, ...]]:
    if path is not None:
        data = _read_profile_data(path) or DEFAULT_PROFILE_DATA
    else:
        data = _read_profile_data(default_profiles_path()) or profile_seed_data()

    profiles_by_task: dict[str, list[ModelProfile]] = {"caption": [], "bbox": []}
    seen: dict[str, set[str]] = {"caption": set(), "bbox": set()}
    for raw in data.get("profiles", []):
        if not isinstance(raw, dict):
            continue
        profile = _profile_from_dict(raw)
        if profile is None:
            continue
        for task in _profile_tasks(raw):
            if profile.id in seen[task]:
                continue
            profiles_by_task[task].append(profile)
            seen[task].add(profile.id)

    for task in ("caption", "bbox"):
        if not profiles_by_task[task]:
            for raw in DEFAULT_PROFILE_DATA["profiles"]:
                if task in _profile_tasks(raw):
                    profile = _profile_from_dict(raw)
                    if profile is not None:
                        profiles_by_task[task].append(profile)
        profiles_by_task[task].extend([CUSTOM_HF_PROFILE, CUSTOM_LOCAL_PROFILE])

    return {task: tuple(profiles) for task, profiles in profiles_by_task.items()}


@dataclass
class CaptioningSettings:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "dummy"
    hf_token: str = ""
    models_dir: str = ""

    caption_profile_id: str = "unsloth-qwen25vl-7b-q4"
    caption_model: str = "qwen25vl"
    caption_hf_repo: str = ""
    caption_model_filename: str = ""
    caption_mmproj_filename: str = ""
    caption_local_model_path: str = ""
    caption_local_mmproj_path: str = ""

    bbox_profile_id: str = "unsloth-qwen25vl-7b-q4"
    bbox_model: str = "qwen25vl"
    bbox_hf_repo: str = ""
    bbox_model_filename: str = ""
    bbox_mmproj_filename: str = ""
    bbox_local_model_path: str = ""
    bbox_local_mmproj_path: str = ""

    add_bboxes_after_json: bool = True
    overwrite_bboxes: bool = True
    filter_bbox_targets: bool = False
    bbox_backend: str = BBOX_BACKEND_VLM
    yolo_bbox_model: str = DEFAULT_YOLOE26_BBOX_MODEL
    yolo_bbox_confidence: float = DEFAULT_ULTRALYTICS_BBOX_CONFIDENCE
    yolo_bbox_imgsz: int = DEFAULT_YOLO_BBOX_IMGSZ
    use_caption_model_for_bboxes: bool = False
    creative_json: bool = True
    disable_thinking: bool = True
    debug_llm_output: bool = False
    vision_image_format: str = "auto"
    plain_caption_mode: str = PLAIN_CAPTION_MODE_STANDARD
    person_caption_name: str = ""
    json_refine_instructions: str = DEFAULT_JSON_REFINE_INSTRUCTIONS

    max_tokens_caption: int = 2000
    max_tokens_json: int = 12000
    max_tokens_bboxes: int = 3000
    context_chars: int = 1200
    max_targets_per_call: int = 0

    server_start_mode: str = "local"
    auto_start_server: bool = True
    llama_server_path: str = ""
    llama_context: int = 32768
    llama_gpu_layers: int = 999
    llama_batch: int = 2048
    llama_ubatch: int = 2048
    llama_threads: int = 0
    llama_extra_args: str = "-fa on"
    llama_reasoning_budget: int = 2048
    caption_server_command: str = ""
    bbox_server_command: str = ""
    server_startup_timeout: float = 120.0
    stop_server_after_job: bool = False

    def __post_init__(self) -> None:
        if not self.models_dir:
            self.models_dir = str(default_models_dir())
        if self.caption_profile_id == "custom":
            self.caption_profile_id = "custom-hf"
        if self.bbox_profile_id == "custom":
            self.bbox_profile_id = "custom-hf"
        # Legacy setting kept only so older settings files still load.
        self.use_caption_model_for_bboxes = False
        backend = str(self.bbox_backend).strip().lower()
        if backend in {"yolo", "yolo26", "ultralytics", "yoloe", "yoloe26", "yoloe-26", "open-vocab", "open_vocab"}:
            self.bbox_backend = BBOX_BACKEND_YOLOE26
        else:
            self.bbox_backend = BBOX_BACKEND_VLM
        self.plain_caption_mode = str(self.plain_caption_mode or "").strip().lower()
        if self.plain_caption_mode not in PLAIN_CAPTION_MODES:
            self.plain_caption_mode = PLAIN_CAPTION_MODE_STANDARD
        self.person_caption_name = str(self.person_caption_name or "").strip()
        yolo_bbox_model = str(self.yolo_bbox_model or "").strip()
        if self.bbox_backend == BBOX_BACKEND_YOLOE26 and (
            not yolo_bbox_model or re.fullmatch(r"yolo26[nsmlx]?\.pt", yolo_bbox_model)
        ):
            yolo_bbox_model = DEFAULT_YOLOE26_BBOX_MODEL
        self.yolo_bbox_model = yolo_bbox_model or DEFAULT_YOLOE26_BBOX_MODEL
        try:
            confidence = float(self.yolo_bbox_confidence)
        except (TypeError, ValueError):
            confidence = DEFAULT_ULTRALYTICS_BBOX_CONFIDENCE
        self.yolo_bbox_confidence = max(0.0, min(1.0, confidence))
        try:
            imgsz = int(self.yolo_bbox_imgsz)
        except (TypeError, ValueError):
            imgsz = DEFAULT_YOLO_BBOX_IMGSZ
        self.yolo_bbox_imgsz = max(0, imgsz)


def bbox_backend_uses_server(settings: CaptioningSettings) -> bool:
    return settings.bbox_backend != BBOX_BACKEND_YOLOE26


def profile_labels(task: str) -> list[str]:
    return [profile.label for profile in profiles_for_task(task)]


def profiles_for_task(task: str) -> tuple[ModelProfile, ...]:
    profiles = load_model_profiles()
    return profiles["bbox"] if task == "bbox" else profiles["caption"]


def profile_id_from_label(task: str, label: str) -> str:
    for profile in profiles_for_task(task):
        if profile.label == label:
            return profile.id
    return profiles_for_task(task)[0].id


def profile_label_from_id(task: str, profile_id: str) -> str:
    if profile_id == "custom":
        profile_id = "custom-hf"
    for profile in profiles_for_task(task):
        if profile.id == profile_id:
            return profile.label
    return profiles_for_task(task)[0].label


def _profile_by_id(task: str, profile_id: str) -> ModelProfile:
    if profile_id == "custom":
        profile_id = "custom-hf"
    for profile in profiles_for_task(task):
        if profile.id == profile_id:
            return profile
    return profiles_for_task(task)[0]


def _custom_runtime_config(settings: CaptioningSettings, task: str, profile: ModelProfile) -> ModelRuntimeConfig:
    if task == "bbox":
        api_model = settings.bbox_model.strip()
        if profile.kind == "custom_local":
            return ModelRuntimeConfig(
                label=profile.label,
                api_model=api_model or profile.api_model,
                kind="local",
                local_model_path=settings.bbox_local_model_path.strip(),
                local_mmproj_path=settings.bbox_local_mmproj_path.strip(),
            )
        return ModelRuntimeConfig(
            label=profile.label,
            api_model=api_model,
            kind="hf",
            hf_repo=settings.bbox_hf_repo.strip(),
            model_filename=settings.bbox_model_filename.strip(),
            mmproj_filename=settings.bbox_mmproj_filename.strip(),
        )

    api_model = settings.caption_model.strip()
    if profile.kind == "custom_local":
        return ModelRuntimeConfig(
            label=profile.label,
            api_model=api_model or profile.api_model,
            kind="local",
            local_model_path=settings.caption_local_model_path.strip(),
            local_mmproj_path=settings.caption_local_mmproj_path.strip(),
        )
    return ModelRuntimeConfig(
        label=profile.label,
        api_model=api_model,
        kind="hf",
        hf_repo=settings.caption_hf_repo.strip(),
        model_filename=settings.caption_model_filename.strip(),
        mmproj_filename=settings.caption_mmproj_filename.strip(),
    )


def runtime_config_for_task(settings: CaptioningSettings, task: str) -> ModelRuntimeConfig:
    profile_id = settings.bbox_profile_id if task == "bbox" else settings.caption_profile_id
    profile = _profile_by_id(task, profile_id)
    if profile.kind in {"custom_hf", "custom_local"}:
        return _custom_runtime_config(settings, task, profile)

    api_model = settings.bbox_model.strip() if task == "bbox" else settings.caption_model.strip()
    return ModelRuntimeConfig(
        label=profile.label,
        api_model=api_model or profile.api_model,
        kind=profile.kind,
        hf_repo=profile.hf_repo,
        mmproj_repo=profile.mmproj_repo,
        model_filename=profile.model_filename,
        mmproj_filename=profile.mmproj_filename,
        local_model_path=profile.local_model_path,
        local_mmproj_path=profile.local_mmproj_path,
    )


def load_settings(path: Path | None = None) -> CaptioningSettings:
    path = path or default_settings_path()
    defaults = CaptioningSettings()
    if not path.exists():
        return defaults

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    allowed = {field.name for field in fields(CaptioningSettings)}
    values = asdict(defaults)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in allowed:
                values[key] = value
    return CaptioningSettings(**values)


def save_settings(settings: CaptioningSettings, path: Path | None = None) -> Path:
    path = path or default_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def safe_repo_dir(repo_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id.strip())
    return cleaned.strip("._") or "custom_model"


def server_host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 8000
    return host, port


def _split_filenames(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def ensure_model_assets(
    settings: CaptioningSettings,
    task: str,
    progress: ProgressCallback | None = None,
) -> ModelAssets:
    config = runtime_config_for_task(settings, task)
    if config.local_model_path:
        model_path = Path(config.local_model_path).expanduser()
        if not model_path.exists():
            raise AutoCaptionError(f"Local model file does not exist: {model_path}")
        mmproj_path: Path | None = None
        if config.local_mmproj_path:
            mmproj_path = Path(config.local_mmproj_path).expanduser()
            if not mmproj_path.exists():
                raise AutoCaptionError(f"Local mmproj file does not exist: {mmproj_path}")
        if progress:
            progress(f"Using local model file: {model_path.name}")
        return ModelAssets(model_path=model_path, mmproj_path=mmproj_path)

    filenames = _split_filenames(config.model_filename)
    mmproj_filenames = _split_filenames(config.mmproj_filename)

    if not config.hf_repo or not filenames:
        return ModelAssets()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise AutoCaptionError("Install huggingface_hub to download Hugging Face model files.") from exc

    models_root = Path(settings.models_dir).expanduser().resolve()
    model_repo = config.hf_repo
    mmproj_repo = config.mmproj_repo or config.hf_repo
    model_dir = models_root / safe_repo_dir(model_repo)
    mmproj_dir = models_root / safe_repo_dir(mmproj_repo)
    model_dir.mkdir(parents=True, exist_ok=True)
    mmproj_dir.mkdir(parents=True, exist_ok=True)
    token = settings.hf_token.strip() or None

    def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
        target = local_dir / filename
        if target.exists():
            if progress:
                progress(f"Using cached model file: {target.name}")
            return target
        if progress:
            progress(f"Downloading {filename} from {repo_id}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                token=token,
            )
        except Exception as exc:  # pragma: no cover - depends on network/HF
            raise AutoCaptionError(f"Could not download {filename} from {repo_id}: {exc}") from exc
        return Path(path)

    downloaded_models = [download_file(model_repo, filename, model_dir) for filename in filenames]
    downloaded_mmproj = [download_file(mmproj_repo, filename, mmproj_dir) for filename in mmproj_filenames]

    model_path = downloaded_models[0] if downloaded_models else None
    mmproj_path = downloaded_mmproj[0] if downloaded_mmproj else None
    return ModelAssets(model_path=model_path, mmproj_path=mmproj_path)


def format_server_command(
    template: str,
    settings: CaptioningSettings,
    task: str,
    assets: ModelAssets,
) -> str:
    config = runtime_config_for_task(settings, task)
    values = {
        "base_url": settings.base_url,
        "api_model": config.api_model,
        "models_dir": str(Path(settings.models_dir).expanduser().resolve()),
        "model_path": str(assets.model_path or ""),
        "mmproj_path": str(assets.mmproj_path or ""),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise AutoCaptionError(f"Unknown server command placeholder: {exc}") from exc


def _split_extra_args(value: str) -> list[str]:
    if not value.strip():
        return []
    if os.name == "nt":
        # Keep this field simple: users can put switches and values separated by spaces.
        return [part for part in value.split() if part]

    import shlex

    return shlex.split(value)


def build_llama_server_command(settings: CaptioningSettings, task: str, assets: ModelAssets) -> str:
    server_path = Path(settings.llama_server_path).expanduser() if settings.llama_server_path.strip() else find_llama_server()
    if server_path is None or not server_path.exists():
        raise AutoCaptionError("Choose llama-server.exe in Preferences before using local captioning.")

    if assets.model_path is None:
        raise AutoCaptionError("Local llama.cpp mode needs a downloadable or local GGUF model profile.")

    host, port = server_host_port(settings.base_url)
    config = runtime_config_for_task(settings, task)
    args = [
        str(server_path),
        "-m",
        str(assets.model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--alias",
        config.api_model or "captioner-model",
        "-c",
        str(max(512, int(settings.llama_context))),
        "-ngl",
        str(max(0, int(settings.llama_gpu_layers))),
        "-b",
        str(max(1, int(settings.llama_batch))),
        "-ub",
        str(max(1, int(settings.llama_ubatch))),
    ]
    if assets.mmproj_path is not None:
        args.extend(["--mmproj", str(assets.mmproj_path)])
    if settings.llama_threads > 0:
        args.extend(["-t", str(settings.llama_threads)])
    if settings.disable_thinking:
        args.extend(["--reasoning", "off"])
    elif settings.llama_reasoning_budget >= 0:
        args.extend(["--reasoning-budget", str(max(0, int(settings.llama_reasoning_budget)))])
    args.extend(_split_extra_args(settings.llama_extra_args))
    if os.name == "nt":
        return subprocess.list2cmdline(args)

    import shlex

    return shlex.join(args)


def api_models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def server_model_ids(base_url: str, api_key: str = "", timeout: float = 3.0) -> set[str]:
    request = urllib.request.Request(api_models_url(base_url))
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= getattr(response, "status", 200) < 300:
            raise AutoCaptionError(f"Server /models returned HTTP {getattr(response, 'status', 'unknown')}.")
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        return set()
    models = payload.get("data", [])
    if not isinstance(models, list):
        return set()

    ids: set[str] = set()
    for model in models:
        if isinstance(model, dict):
            model_id = model.get("id")
            if isinstance(model_id, str) and model_id.strip():
                ids.add(model_id.strip())
    return ids


def resolve_server_model_id(
    configured_model: str,
    model_ids: Iterable[str],
    *,
    auto_select_single: bool = False,
) -> str:
    configured = configured_model.strip()
    available = sorted({model_id.strip() for model_id in model_ids if model_id.strip()})
    if configured and (not available or configured in available):
        return configured
    if auto_select_single and len(available) == 1:
        return available[0]

    available_text = ", ".join(available) or "(none reported)"
    if configured:
        raise AutoCaptionError(
            f"The server does not expose the configured API model ID '{configured}'. "
            f"Available model IDs: {available_text}. In Preferences, use Test / Discover Models "
            "and select the exact ID reported by the server."
        )
    raise AutoCaptionError(
        f"No API model ID is selected. Available model IDs: {available_text}. "
        "In Preferences, use Test / Discover Models and select a model."
    )


def is_server_ready(base_url: str, api_key: str = "", timeout: float = 3.0) -> bool:
    try:
        server_model_ids(base_url, api_key, timeout)
        return True
    except Exception:
        return False


def start_server_process(
    command: str,
    base_url: str,
    api_key: str,
    log_dir: Path,
    name: str,
    startup_timeout: float,
    progress: ProgressCallback | None = None,
) -> subprocess.Popen:
    if not command.strip():
        raise AutoCaptionError("Server auto-start is enabled, but no server command is configured.")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("a", encoding="utf-8", errors="replace")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0

    if progress:
        progress(f"Starting {name} server...")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    process._captioner_log_file = log_file  # type: ignore[attr-defined]

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            close_process_log(process)
            raise AutoCaptionError(f"{name} server exited during startup. See {log_path}.")
        if is_server_ready(base_url, api_key=api_key, timeout=3.0):
            if progress:
                progress(f"{name} server is ready.")
            return process
        time.sleep(1.0)

    stop_server_process(process)
    raise AutoCaptionError(f"{name} server did not become ready within {startup_timeout:.0f} seconds.")


def close_process_log(process: subprocess.Popen) -> None:
    try:
        log_file = getattr(process, "_captioner_log_file", None)
        if log_file is not None:
            log_file.close()
    except Exception:
        pass


def stop_server_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        try:
            process.wait(timeout=10)
        except Exception:
            pass
    finally:
        close_process_log(process)


def image_to_data_url(path: Path, vision_image_format: str = "auto") -> str:
    fmt = vision_image_format.lower().strip()
    if fmt not in {"auto", "original", "png", "jpeg", "jpg"}:
        raise ValueError(f"Invalid vision image format: {vision_image_format}")

    suffix = path.suffix.lower()
    convert_to: str | None = None
    if fmt == "png":
        convert_to = "PNG"
    elif fmt in {"jpeg", "jpg"}:
        convert_to = "JPEG"
    elif fmt == "auto" and suffix == ".webp":
        convert_to = "PNG"

    if convert_to is None:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    with Image.open(path) as image:
        if convert_to == "JPEG":
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            mime = "image/jpeg"
        else:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            mime = "image/png"

        buffer = io.BytesIO()
        image.save(buffer, format=convert_to)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


def _exif_tag_maps(exif: Image.Exif) -> list[dict[int, Any]]:
    maps: list[dict[int, Any]] = [dict(exif)]
    try:
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        exif_ifd = {}
    if exif_ifd:
        maps.append(dict(exif_ifd))
    return maps


def _first_exif_value(tag_maps: list[dict[int, Any]], *tags: int) -> Any:
    for tag in tags:
        for tag_map in tag_maps:
            if tag in tag_map:
                value = tag_map[tag]
                if value is not None and value != "":
                    return value
    return None


def _rational_float(value: Any) -> float | None:
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            denominator = float(value[1])
            if denominator == 0:
                return None
            return float(value[0]) / denominator
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _format_exif_decimal(value: float, digits: int = 1) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_exif_string(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text or None


def _format_f_stop(value: Any) -> str | None:
    number = _rational_float(value)
    if number is None or number <= 0:
        return None
    return f"f/{_format_exif_decimal(number, 1)}"


def _format_focal_length(value: Any) -> str | None:
    number = _rational_float(value)
    if number is None or number <= 0:
        return None
    return f"{_format_exif_decimal(number, 1)}mm"


def _format_exposure_time(value: Any) -> str | None:
    seconds = _rational_float(value)
    if seconds is None or seconds <= 0:
        return None
    if seconds < 1:
        denominator = round(1 / seconds)
        if denominator > 0 and abs(seconds - (1 / denominator)) < 0.0005:
            return f"1/{denominator}s"
    return f"{_format_exif_decimal(seconds, 3)}s"


def _format_iso(value: Any) -> str | None:
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    try:
        iso = int(value)
    except (TypeError, ValueError):
        return _format_exif_string(value)
    return str(iso) if iso > 0 else None


def _format_exposure_bias(value: Any) -> str | None:
    number = _rational_float(value)
    if number is None:
        return None
    sign = "+" if number > 0 else ""
    return f"{sign}{_format_exif_decimal(number, 2)} EV"


def _mapped_exif_value(value: Any, names: dict[int, str]) -> str | None:
    try:
        return names.get(int(value)) or _format_exif_string(value)
    except (TypeError, ValueError):
        return _format_exif_string(value)


def _format_flash(value: Any) -> str | None:
    try:
        flags = int(value)
    except (TypeError, ValueError):
        return _format_exif_string(value)
    return "fired" if flags & 1 else "did not fire"


def image_exif_context(image_path: Path) -> dict[str, str]:
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except Exception:
        return {}
    if not exif:
        return {}

    tag_maps = _exif_tag_maps(exif)
    exposure_programs = {
        0: "not defined",
        1: "manual",
        2: "normal program",
        3: "aperture priority",
        4: "shutter priority",
        5: "creative program",
        6: "action program",
        7: "portrait mode",
        8: "landscape mode",
    }
    metering_modes = {
        0: "unknown",
        1: "average",
        2: "center-weighted average",
        3: "spot",
        4: "multi-spot",
        5: "pattern",
        6: "partial",
        255: "other",
    }
    white_balance_modes = {0: "auto", 1: "manual"}
    field_specs: tuple[tuple[str, Callable[[Any], str | None], tuple[int, ...]], ...] = (
        ("camera_make", _format_exif_string, (int(ExifTags.Base.Make),)),
        ("camera_model", _format_exif_string, (int(ExifTags.Base.Model),)),
        ("lens_make", _format_exif_string, (int(ExifTags.Base.LensMake),)),
        ("lens_model", _format_exif_string, (int(ExifTags.Base.LensModel),)),
        ("f_stop", _format_f_stop, (int(ExifTags.Base.FNumber),)),
        ("exposure_time", _format_exposure_time, (int(ExifTags.Base.ExposureTime),)),
        ("iso", _format_iso, (int(ExifTags.Base.ISOSpeedRatings), 34855)),
        ("focal_length", _format_focal_length, (int(ExifTags.Base.FocalLength),)),
        ("exposure_bias", _format_exposure_bias, (int(ExifTags.Base.ExposureBiasValue),)),
        (
            "exposure_program",
            lambda value: _mapped_exif_value(value, exposure_programs),
            (int(ExifTags.Base.ExposureProgram),),
        ),
        ("metering_mode", lambda value: _mapped_exif_value(value, metering_modes), (int(ExifTags.Base.MeteringMode),)),
        ("flash", _format_flash, (int(ExifTags.Base.Flash),)),
        ("white_balance", lambda value: _mapped_exif_value(value, white_balance_modes), (int(ExifTags.Base.WhiteBalance),)),
    )

    context: dict[str, str] = {}
    for key, formatter, tags in field_specs:
        value = _first_exif_value(tag_maps, *tags)
        if value is None:
            continue
        formatted = formatter(value)
        if formatted:
            context[key] = formatted
    return context


def append_image_exif_context(user_prompt: str, image_path: Path) -> str:
    context = image_exif_context(image_path)
    if not context:
        return user_prompt
    metadata_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join(
        (
            user_prompt.rstrip(),
            "Image EXIF metadata, if useful for photographic style fields. Treat the image as authoritative:",
            metadata_json,
        )
    )


def extract_json(text: str) -> dict[str, Any]:
    raw_output = text
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ModelJsonError(f"No JSON object found in model output: {text[:500]!r}", raw_output=raw_output)

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ModelJsonError(f"Could not parse model JSON: {exc}", raw_output=raw_output, candidate=candidate) from exc
    if not isinstance(parsed, dict):
        raise ModelJsonError("Model output JSON root was not an object.", raw_output=raw_output, candidate=candidate)
    return parsed


def _make_openai_client(settings: CaptioningSettings):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AutoCaptionError("Install openai to use auto captioning.") from exc
    return OpenAI(base_url=settings.base_url.rstrip("/") + "/", api_key=settings.api_key or "dummy")


def request_user_prompt(settings: CaptioningSettings, user: str) -> str:
    if not settings.disable_thinking:
        return user
    stripped = user.lstrip()
    if stripped.startswith("/no_think"):
        return user
    return "/no_think\n\n" + user


THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def _plain_response_value(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain_response_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_response_value(item, depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _plain_response_value(value.model_dump(mode="json"), depth + 1)
        except TypeError:
            return _plain_response_value(value.model_dump(), depth + 1)
    if hasattr(value, "dict"):
        try:
            return _plain_response_value(value.dict(), depth + 1)
        except TypeError:
            pass
    if hasattr(value, "__dict__"):
        return {
            key: _plain_response_value(item, depth + 1)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def _message_field(message: Any, name: str) -> Any:
    if message is None:
        return None
    value = getattr(message, name, None)
    if value is not None:
        return value
    for extra_name in ("model_extra", "additional_kwargs"):
        extra = getattr(message, extra_name, None)
        if isinstance(extra, dict) and name in extra:
            return extra[name]
    if isinstance(message, dict):
        return message.get(name)
    return None


def _debug_log_path(settings: CaptioningSettings, kind: str, model: str) -> Path:
    folder = Path(settings.models_dir).expanduser().resolve() / "llm_debug"
    folder.mkdir(parents=True, exist_ok=True)
    safe_kind = re.sub(r"[^A-Za-z0-9_.-]+", "_", kind).strip("_") or "request"
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return folder / f"{stamp}-{time.time_ns()}-{safe_kind}-{safe_model}.json"


def write_llm_debug_log(settings: CaptioningSettings, kind: str, model: str, response: Any, choice: Any) -> Path | None:
    if not settings.debug_llm_output:
        return None
    try:
        message = getattr(choice, "message", None) if choice is not None else None
        content = _message_field(message, "content")
        content_text = content if isinstance(content, str) else ""
        thinking_blocks = [match.group(1).strip() for match in THINK_BLOCK_RE.finditer(content_text)]
        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": kind,
            "model": model,
            "response_model": getattr(response, "model", ""),
            "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
            "usage": _plain_response_value(getattr(response, "usage", None)),
            "reasoning_content": _plain_response_value(_message_field(message, "reasoning_content")),
            "thinking_blocks": thinking_blocks,
            "visible_content": strip_thinking_output(content_text) if content_text else "",
            "raw_content": _plain_response_value(content),
            "response": _plain_response_value(response),
        }
        path = _debug_log_path(settings, kind, model)
        path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except (OSError, TypeError, ValueError):
        return None


def strip_thinking_output(content: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", content)
    return cleaned.strip()


def empty_response_detail(response: Any, choice: Any, content: str | None = None) -> str:
    finish_reason = getattr(choice, "finish_reason", None)
    response_model = getattr(response, "model", "")
    detail = f"finish_reason={finish_reason}, response_model={response_model or 'unknown'}"
    if content and not strip_thinking_output(content):
        detail += ", content contained only thinking output"
    return detail


def empty_response_hint(settings: CaptioningSettings) -> str:
    if settings.disable_thinking:
        return (
            "The server returned a completion object but no assistant text. Check the llama-server log for "
            "template/mmproj errors."
        )
    return (
        "Thinking/reasoning is enabled and the server returned no visible assistant text. If finish_reason=length, "
        "the model likely used the response budget before producing the final answer. Increase the task max-token "
        "budget and Context size, lower the Thinking token budget, or turn Disable thinking/reasoning back on."
    )


def request_failure_message(kind: str, exc: Exception) -> str:
    message = str(exc)
    hint = ""
    if "connection error" in message.lower():
        hint = (
            " The local server may have crashed or closed the connection during generation. "
            "Check models/server_logs for llama-server assertions or out-of-memory errors."
        )
    return f"{kind} model request failed: {exc}.{hint}"


def chat_text(
    settings: CaptioningSettings,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    if not model:
        raise AutoCaptionError("No caption model name is configured.")
    client = _make_openai_client(settings)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": request_user_prompt(settings, user)},
            ],
        )
    except Exception as exc:
        raise AutoCaptionError(request_failure_message("Text", exc)) from exc
    choice = response.choices[0] if response.choices else None
    write_llm_debug_log(settings, "text", model, response, choice)
    content = choice.message.content if choice is not None else None
    visible_content = strip_thinking_output(content) if content else ""
    if visible_content:
        return visible_content
    detail = empty_response_detail(response, choice, content)
    raise AutoCaptionError(f"Text model '{model}' returned no visible response. {detail}. {empty_response_hint(settings)}")


def chat_vision(
    settings: CaptioningSettings,
    model: str,
    image_path: Path,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    if not model:
        raise AutoCaptionError("No vision model name is configured.")
    client = _make_openai_client(settings)
    image_url = image_to_data_url(image_path, settings.vision_image_format)

    def request(image_first: bool):
        content_parts = [
            {"type": "text", "text": request_user_prompt(settings, user)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        if image_first:
            content_parts.reverse()
        return client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content_parts},
            ],
        )

    errors: list[str] = []
    for image_first in (True, False):
        try:
            response = request(image_first=image_first)
        except Exception as exc:
            raise AutoCaptionError(request_failure_message("Vision", exc)) from exc
        choice = response.choices[0] if response.choices else None
        write_llm_debug_log(settings, f"vision_image_first_{image_first}", model, response, choice)
        content = choice.message.content if choice is not None else None
        visible_content = strip_thinking_output(content) if content else ""
        if visible_content:
            return visible_content
        errors.append(f"image_first={image_first}, {empty_response_detail(response, choice, content)}")

    raise AutoCaptionError(
        f"Vision model '{model}' returned no visible response after image-order retry. {'; '.join(errors)}. "
        f"{empty_response_hint(settings)}"
    )


JSON_REPAIR_SYSTEM = """
You repair malformed JSON emitted by another model.
Return exactly one compact valid JSON object. No markdown. No commentary.
Preserve the original content and field names whenever possible.
If the response contains extra prose, remove the prose.
If the response is truncated or impossible to fully repair, return the most complete valid object that preserves the usable content.
""".strip()

JSON_REPAIR_USER = """
The model response below was supposed to be {expected}.

Parser error:
{error}

Repair the response into one valid JSON object.

Model response:
{raw_output}
""".strip()


def _repair_prompt_text(raw_output: str, limit: int = 24000) -> str:
    raw_output = raw_output.strip()
    if len(raw_output) <= limit:
        return raw_output
    half = limit // 2
    return raw_output[:half].rstrip() + "\n\n... [middle omitted for JSON repair] ...\n\n" + raw_output[-half:].lstrip()


def parse_json_with_repair(
    settings: CaptioningSettings,
    task: str,
    raw_output: str,
    expected: str,
    max_tokens: int,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    first_error: ModelJsonError
    try:
        return extract_json(raw_output)
    except ModelJsonError as exc:
        first_error = exc
        if progress is not None:
            progress("Model returned invalid JSON; retrying with a JSON repair prompt.")

    config = runtime_config_for_task(settings, task)
    repair_raw = ""
    try:
        repair_raw = chat_text(
            settings=settings,
            model=config.api_model,
            system=JSON_REPAIR_SYSTEM,
            user=format_prompt(
                JSON_REPAIR_USER,
                expected=expected,
                error=str(first_error),
                raw_output=_repair_prompt_text(first_error.raw_output or raw_output),
            ),
            max_tokens=max(2000, max_tokens),
            temperature=0.0,
        )
        parsed = extract_json(repair_raw)
    except Exception as repair_error:
        message = f"{first_error}; repair retry failed: {repair_error}"
        raise ModelJsonError(
            message,
            raw_output=first_error.raw_output or raw_output,
            candidate=first_error.candidate,
            repair_output=repair_raw,
        ) from repair_error

    if progress is not None:
        progress("JSON repair retry succeeded.")
    return parsed


PLAIN_CAPTION_SYSTEM = """
You write factual image captions for image-generation datasets.
Return one polished plain-text caption only. No markdown, no JSON, no bullet points.
Preserve any visible text exactly. Describe the main subjects, setting, style,
lighting, camera/viewpoint, and notable objects without guessing identities.
""".strip()

PLAIN_CAPTION_USER = """
Write a detailed but clean text-to-image caption for this image. Keep it useful
for recreating the image, but avoid unsupported proper names or speculation.
""".strip()

PERSON_CAPTION_SYSTEM = """
You write factual plain-text captions for image-generation training datasets of known people.
Return one polished single-line caption only. No markdown, no JSON, no bullet points, and no newlines.
Use the provided person's name at least once.
Do not describe the named person's physical features, face, body, age, ethnicity, skin, hair, or other biometric traits.
You may describe clothing, pose, action, objects, setting, lighting, viewpoint, composition, mood, and photo quality.
If there are multiple people, identify them by position such as left, right, center, foreground, or background.
Use bracketed generic labels like [MAN], [WOMAN], [BOY], or [GIRL] for people whose names are not provided.
If more than two people are visible, assume the provided name belongs to the most prominent or in-focus person; if equal, assume it belongs to the leftmost person.
""".strip()

PERSON_CAPTION_USER = """
Caption this image for a text-to-image LoRA dataset.
Provided person name: {person_name}

Describe the image naturally and descriptively while following the system rules.
For images with two or more featured people, explicitly describe each featured person by position and use the provided name for the named person.
""".strip()

CREATIVE_DIRECTIVE = """
Expansion policy:
- Preserve the source caption's idea.
- Add useful visual detail when it helps the caption.
- Add only supportive background and scene details that do not replace or contradict the source caption.
- Never introduce a different main subject.
- Preserve trigger tokens/names/styles exactly.
- Do not invent appearance details for named people or trigger identities.
""".strip()

FAITHFUL_DIRECTIVE = """
Fidelity policy:
- Fill in only what the structured schema needs.
- Do not add new subjects, props, setting, style details, colors, brands, text, or atmosphere not present in the source caption.
- If the source caption is sparse, the JSON stays sparse.
""".strip()

JSON_SCHEMA_INSTRUCTIONS = """
Return exactly one compact valid JSON object. No markdown. No commentary.

Schema:
{
  "high_level_description": "...",
  "style_description": {
    "aesthetics": "...",
    "lighting": "...",
    "photo": "...",
    "medium": "photograph"
  },
  "compositional_deconstruction": {
    "background": "...",
    "elements": [
      {"type": "obj", "desc": "..."},
      {"type": "text", "text": "...", "desc": "..."}
    ]
  }
}

Field guidance:
- high_level_description: one or two sentences summarizing the whole image.
- aesthetics: concise visual style keywords, e.g. "moody, cinematic, desaturated" or "warm, playful, vibrant".
- lighting: concrete light quality, source, and shadow behavior, e.g. "golden hour, rim light, dramatic shadows" or "bright afternoon sunlight, long soft shadows".
- photo: camera, lens, viewpoint, focus, and photographic traits for photos, e.g. "35mm, f/1.4, bokeh", "shallow depth of field, eye-level, 85mm lens", or "wide angle, f/8, long exposure".
- medium: use a compact medium label such as "photograph", "illustration", "3d_render", "painting", or "graphic_design".
- art_style: style and medium traits for non-photo captions, e.g. "flat vector illustration, bold outlines" or "flat vector design, generous whitespace, sans-serif typography".
- background: describe the environment, setting, distant scenery, surfaces, and atmosphere.
- elements desc: describe each subject/object with its visible appearance, clothing/materials, pose/action, and important props.

Rules:
- Include high_level_description, style_description, and compositional_deconstruction.
- compositional_deconstruction must contain background first, then elements.
- Use "photo" for photographic images, or replace it with "art_style" for non-photo artwork.
- Use exactly one of "photo" or "art_style".
- Do not include bbox values. Bboxes are added in a separate pass.
- Do not include color_palette fields.
- type is "obj" for normal subjects/objects and "text" only for literal visible text.
- Text elements must preserve the literal visible text exactly.
- A coherent subject is one element; do not split people, vehicles, plants, buildings, or products into parts.
- Put ground, sky, walls, distant scenery, and ambient environment into background.
- Put people, animals, vehicles, products, furniture, props, signs, and visible text into elements.
- Keep trigger tokens, names, identifiers, and stylized spelling exactly.
""".strip()

TEXT_TO_JSON_SYSTEM = """
You convert an existing vetted text caption into an Ideogram 4 structured JSON caption.
The source caption is authoritative; organize it into the schema without recaptioning the image.
""".strip()

TEXT_TO_JSON_USER = """
Existing vetted caption:
{caption}

Convert this caption into Ideogram 4 structured JSON.
""".strip()

IMAGE_TO_JSON_SYSTEM = """
You inspect an image and produce an Ideogram 4 structured JSON caption.
The image is authoritative. Describe only what is visible.
""".strip()

IMAGE_TO_JSON_USER = """
Create an Ideogram 4 structured JSON caption for this image.
Do not reference any existing text caption.
""".strip()

JSON_REFINE_SYSTEM = """
You revise an existing Ideogram 4 structured JSON caption for an image dataset.
Use the image as the visual authority, the current JSON as the structure to improve,
the text caption as supporting context, and the user's edit instructions as the task.

Return exactly one compact valid JSON object. No markdown. No commentary.

Schema:
{
  "high_level_description": "...",
  "style_description": {
    "aesthetics": "...",
    "lighting": "...",
    "photo": "...",
    "medium": "photograph"
  },
  "compositional_deconstruction": {
    "background": "...",
    "elements": [
      {"type": "obj", "bbox": [y1,x1,y2,x2], "desc": "..."},
      {"type": "text", "bbox": [y1,x1,y2,x2], "text": "...", "desc": "..."}
    ]
  }
}

Field guidance:
- high_level_description: one or two sentences summarizing the whole image.
- aesthetics: concise visual style keywords, e.g. "moody, cinematic, desaturated" or "warm, playful, vibrant".
- lighting: concrete light quality, source, and shadow behavior, e.g. "golden hour, rim light, dramatic shadows" or "bright afternoon sunlight, long soft shadows".
- photo: camera, lens, viewpoint, focus, and photographic traits for photos, e.g. "35mm, f/1.4, bokeh", "shallow depth of field, eye-level, 85mm lens", or "wide angle, f/8, long exposure".
- medium: use a compact medium label such as "photograph", "illustration", "3d_render", "painting", or "graphic_design".
- art_style: style and medium traits for non-photo captions, e.g. "flat vector illustration, bold outlines" or "flat vector design, generous whitespace, sans-serif typography".
- background: describe the environment, setting, distant scenery, surfaces, and atmosphere.
- elements desc: describe each subject/object with its visible appearance, clothing/materials, pose/action, and important props.
- bbox: when present, use [y_min,x_min,y_max,x_max] normalized 0..1000 with origin at top-left.

Rules:
- Preserve trigger tokens, names, identifiers, and stylized spelling exactly.
- Preserve literal visible text exactly.
- Preserve existing bbox values for unchanged elements. Do not invent bboxes for new elements.
- Do not remove real visible elements unless the user's instructions explicitly say to.
- A coherent subject is one element; do not split people, vehicles, plants, buildings, or products into parts.
- Put people, animals, vehicles, products, furniture, props, signs, and visible text into elements.
- Put ground, sky, walls, distant scenery, and ambient environment into background.
- For photographic images use "photo"; for non-photo artwork use "art_style". Use exactly one of those keys.
""".strip()

JSON_REFINE_USER = """
User edit instructions:
{instructions}

Existing text caption:
{source_caption}

Current structured JSON:
{caption_json}

Revise the structured JSON according to the instructions.
""".strip()

BATCH_GROUND_SYSTEM = """
Locate multiple existing target elements in the image.
The targets already exist in a structured JSON caption. Your job is only to supply coordinates.
Do not invent new elements. Do not split or merge elements. Do not reinterpret targets.

Return only valid compact JSON in exactly this shape:
{"bboxes":{"0":[x1,y1,x2,y2],"1":null}}

Rules:
- Include every requested target id exactly once.
- Use null if the target is not visible or you are not confident.
- bbox values are normalized 0..1000.
- bbox origin is top-left.
- bbox format is [x1,y1,x2,y2].
- bbox should tightly cover the visible extent of that target only.
""".strip()

BATCH_GROUND_USER = """
Supporting structured JSON context:
{context_json}

Targets to locate:
{targets_json}

Return only:
{{"bboxes":{{"0":[x1,y1,x2,y2],"1":null}}}}
""".strip()


DEFAULT_PROMPT_TEXTS: dict[str, str] = {
    "plain_caption_system": PLAIN_CAPTION_SYSTEM,
    "plain_caption_user": PLAIN_CAPTION_USER,
    "person_caption_system": PERSON_CAPTION_SYSTEM,
    "person_caption_user": PERSON_CAPTION_USER,
    "creative_directive": CREATIVE_DIRECTIVE,
    "faithful_directive": FAITHFUL_DIRECTIVE,
    "json_schema_instructions": JSON_SCHEMA_INSTRUCTIONS,
    "text_to_json_system": TEXT_TO_JSON_SYSTEM,
    "text_to_json_user": TEXT_TO_JSON_USER,
    "image_to_json_system": IMAGE_TO_JSON_SYSTEM,
    "image_to_json_user": IMAGE_TO_JSON_USER,
    "json_refine_system": JSON_REFINE_SYSTEM,
    "json_refine_user": JSON_REFINE_USER,
    "bbox_system": BATCH_GROUND_SYSTEM,
    "bbox_user": BATCH_GROUND_USER,
}


def load_prompts(path: Path | None = None) -> dict[str, Any]:
    folder = path or default_prompts_path()
    prompts = dict(DEFAULT_PROMPT_TEXTS)
    if not folder.exists() or not folder.is_dir():
        return prompts
    for name in DEFAULT_PROMPT_TEXTS:
        prompt_path = folder / f"{name}.txt"
        if not prompt_path.exists():
            continue
        try:
            prompts[name] = prompt_path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            continue
    return prompts


def write_default_prompts(path: Path | None = None) -> Path:
    folder = path or default_prompts_path()
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in DEFAULT_PROMPT_TEXTS.items():
        prompt_path = folder / f"{name}.txt"
        if not prompt_path.exists():
            prompt_path.write_text(text, encoding="utf-8")
    return folder


def format_prompt(template: str, **values: Any) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise AutoCaptionError(f"Prompt is missing required placeholder {{{exc.args[0]}}}.") from exc


def single_line_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip('"').strip())


def generate_plain_caption(settings: CaptioningSettings, image_path: Path, person_name: str | None = None) -> str:
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    if settings.plain_caption_mode == PLAIN_CAPTION_MODE_PERSON:
        resolved_name = (person_name if person_name is not None else settings.person_caption_name).strip()
        if not resolved_name:
            raise AutoCaptionError("Enter a person name before using person text caption mode.")
        system = prompts["person_caption_system"]
        user = format_prompt(prompts["person_caption_user"], person_name=resolved_name)
    else:
        system = prompts["plain_caption_system"]
        user = prompts["plain_caption_user"]
    raw = chat_vision(
        settings=settings,
        model=config.api_model,
        image_path=image_path,
        system=system,
        user=user,
        max_tokens=settings.max_tokens_caption,
        temperature=0.2,
    )
    return single_line_caption(raw)


def _directive(settings: CaptioningSettings) -> str:
    prompts = load_prompts()
    key = "creative_directive" if settings.creative_json else "faithful_directive"
    return str(prompts[key])


def json_system_prompt(prompts: dict[str, Any], task_system_key: str, settings: CaptioningSettings) -> str:
    return "\n\n".join(
        part
        for part in (
            str(prompts[task_system_key]).strip(),
            str(prompts["json_schema_instructions"]).strip(),
            _directive(settings).strip(),
        )
        if part
    )


def generate_json_from_text(
    settings: CaptioningSettings,
    caption_text: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    caption_text = caption_text.strip()
    if not caption_text:
        raise AutoCaptionError("No source text caption was found.")
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    raw = chat_text(
        settings=settings,
        model=config.api_model,
        system=json_system_prompt(prompts, "text_to_json_system", settings),
        user=format_prompt(
            prompts["text_to_json_user"],
            caption=caption_text,
            directive="",
        ),
        max_tokens=settings.max_tokens_json,
        temperature=0.0,
    )
    parsed = parse_json_with_repair(
        settings=settings,
        task="caption",
        raw_output=raw,
        expected="an Ideogram 4 structured caption JSON object",
        max_tokens=settings.max_tokens_json,
        progress=progress,
    )
    return normalize_caption(parsed)


def generate_json_from_image(
    settings: CaptioningSettings,
    image_path: Path,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    raw = chat_vision(
        settings=settings,
        model=config.api_model,
        image_path=image_path,
        system=json_system_prompt(prompts, "image_to_json_system", settings),
        user=append_image_exif_context(
            format_prompt(prompts["image_to_json_user"], directive=""),
            image_path,
        ),
        max_tokens=settings.max_tokens_json,
        temperature=0.0,
    )
    parsed = parse_json_with_repair(
        settings=settings,
        task="caption",
        raw_output=raw,
        expected="an Ideogram 4 structured caption JSON object",
        max_tokens=settings.max_tokens_json,
        progress=progress,
    )
    return normalize_caption(parsed)


def _preserve_missing_refined_bboxes(original: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    original = normalize_caption(original)
    refined = normalize_caption(refined)
    original_elements = original.get("compositional_deconstruction", {}).get("elements", [])
    refined_elements = refined.get("compositional_deconstruction", {}).get("elements", [])
    for index, refined_element in enumerate(refined_elements):
        if normalize_bbox(refined_element.get("bbox")) is not None or index >= len(original_elements):
            continue
        original_element = original_elements[index]
        if refined_element.get("type") != original_element.get("type"):
            continue
        bbox = normalize_bbox(original_element.get("bbox"))
        if bbox is not None:
            refined_element["bbox"] = bbox
    return normalize_caption(refined)


def generate_json_refinement(
    settings: CaptioningSettings,
    image_path: Path,
    caption: dict[str, Any],
    source_caption: str,
    instructions: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    instructions = instructions.strip()
    if not instructions:
        raise AutoCaptionError("No JSON refinement instructions were provided.")
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    current_caption = normalize_caption(caption)
    raw = chat_vision(
        settings=settings,
        model=config.api_model,
        image_path=image_path,
        system=prompts["json_refine_system"],
        user=append_image_exif_context(
            format_prompt(
                prompts["json_refine_user"],
                instructions=instructions,
                source_caption=source_caption.strip() or "(none)",
                caption_json=json.dumps(current_caption, ensure_ascii=False, indent=2),
            ),
            image_path,
        ),
        max_tokens=settings.max_tokens_json,
        temperature=0.0,
    )
    parsed = parse_json_with_repair(
        settings=settings,
        task="caption",
        raw_output=raw,
        expected="a refined Ideogram 4 structured caption JSON object",
        max_tokens=settings.max_tokens_json,
        progress=progress,
    )
    return _preserve_missing_refined_bboxes(current_caption, parsed)


def bbox_xyxy_to_yxyx(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return normalize_bbox([y1, x1, y2, x2])


def bbox_xyxy_pixels_to_yxyx(bbox: Any, width: int, height: int) -> list[int] | None:
    if width <= 0 or height <= 0 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return bbox_xyxy_to_yxyx(
        [
            x1 * 1000.0 / width,
            y1 * 1000.0 / height,
            x2 * 1000.0 / width,
            y2 * 1000.0 / height,
        ]
    )


_YOLO_MODEL_CACHE: dict[str, Any] = {}


def _load_yolo_model(model_name: str, progress: ProgressCallback | None = None, label: str = "Ultralytics") -> Any:
    name = model_name.strip() or DEFAULT_YOLOE26_BBOX_MODEL
    cached = _YOLO_MODEL_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise AutoCaptionError(f"Install ultralytics>=8.4.0 to use the {label} bbox backend.") from exc
    if progress:
        progress(f"Loading {label} model {name}...")
    try:
        model = YOLO(name)
    except Exception as exc:  # pragma: no cover - depends on local model download/runtime
        raise AutoCaptionError(f"Could not load {label} model '{name}': {exc}") from exc
    _YOLO_MODEL_CACHE[name] = model
    return model


def _plain_list(value: Any) -> Any:
    if value is None:
        return []
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _rows_list(value: Any) -> list[list[float]]:
    rows = _plain_list(value)
    if not isinstance(rows, list):
        return []
    out: list[list[float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        try:
            out.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        except (TypeError, ValueError):
            continue
    return out


def _flat_list(value: Any) -> list[Any]:
    items = _plain_list(value)
    if items is None:
        return []
    if isinstance(items, (int, float, str)):
        return [items]
    if isinstance(items, tuple):
        return list(items)
    if isinstance(items, list):
        return items
    return []


def _class_names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for key, value in names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            label = str(value).strip()
            if label:
                out[class_id] = label
        return out
    if isinstance(names, (list, tuple)):
        return {index: str(value).strip() for index, value in enumerate(names) if str(value).strip()}
    return {}


def _yolo_class_names(model: Any, result: Any) -> dict[int, str]:
    result_names = _class_names_map(getattr(result, "names", None))
    if result_names:
        return result_names
    return _class_names_map(getattr(model, "names", None))


def _image_size_from_result(image_path: Path, result: Any) -> tuple[int, int]:
    shape = getattr(result, "orig_shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            height = int(shape[0])
            width = int(shape[1])
            if width > 0 and height > 0:
                return width, height
        except (TypeError, ValueError):
            pass
    with Image.open(image_path) as image:
        return image.size


def yolo_detections_for_image(
    settings: CaptioningSettings,
    image_path: Path,
    progress: ProgressCallback | None = None,
    class_prompts: list[str] | None = None,
    model_name: str | None = None,
    label: str = "YOLOE-26",
) -> tuple[list[YoloDetection], tuple[str, ...]]:
    model_name = model_name or settings.yolo_bbox_model.strip() or DEFAULT_YOLOE26_BBOX_MODEL
    model = _load_yolo_model(model_name, progress, label=label)
    if class_prompts is not None:
        try:
            model.set_classes(class_prompts)
        except Exception as exc:
            raise AutoCaptionError(f"Could not set {label} text prompts: {exc}") from exc
    if progress:
        progress(f"Running {label} bbox detector ({model_name})...")
    predict_kwargs: dict[str, Any] = {
        "source": str(image_path),
        "conf": settings.yolo_bbox_confidence,
        "verbose": False,
    }
    if settings.yolo_bbox_imgsz > 0:
        predict_kwargs["imgsz"] = settings.yolo_bbox_imgsz
    try:
        results = model.predict(**predict_kwargs)
    except TypeError:
        call_kwargs = {key: value for key, value in predict_kwargs.items() if key != "source"}
        results = model(str(image_path), **call_kwargs)
    except Exception as exc:  # pragma: no cover - depends on local model runtime
        raise AutoCaptionError(f"{label} bbox detection failed for {image_path.name}: {exc}") from exc
    if not results:
        return [], tuple()

    result = results[0]
    boxes = getattr(result, "boxes", None)
    names = _yolo_class_names(model, result)
    class_names = tuple(name for _class_id, name in sorted(names.items()))
    if boxes is None:
        return [], class_names

    width, height = _image_size_from_result(image_path, result)
    xyxy_rows = _rows_list(getattr(boxes, "xyxy", None))
    confidences = _flat_list(getattr(boxes, "conf", None))
    classes = _flat_list(getattr(boxes, "cls", None))

    detections: list[YoloDetection] = []
    for index, xyxy in enumerate(xyxy_rows):
        try:
            class_id = int(float(classes[index])) if index < len(classes) else -1
        except (TypeError, ValueError):
            class_id = -1
        try:
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        bbox = bbox_xyxy_pixels_to_yxyx(xyxy, width=width, height=height)
        if bbox is None:
            continue
        detections.append(
            YoloDetection(
                index=index,
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=confidence,
                bbox=bbox,
            )
        )
    return detections, class_names


def _normalized_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for word in list(words):
        if len(word) > 4 and word.endswith("ies"):
            words.add(word[:-3] + "y")
        elif len(word) > 4 and word.endswith("es"):
            words.add(word[:-2])
        elif len(word) > 3 and word.endswith("s"):
            words.add(word[:-1])
    return words


def _normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _target_text(element: dict[str, Any]) -> str:
    parts = [str(element.get("desc", ""))]
    if element.get("type") == "text":
        parts.append(str(element.get("text", "")))
    return " ".join(part for part in parts if part.strip())


def _detection_class_key(detection: YoloDetection) -> str:
    return _normalized_phrase(detection.class_name)


def _detection_area(detection: YoloDetection) -> int:
    y1, x1, y2, x2 = detection.bbox
    return max(0, y2 - y1) * max(0, x2 - x1)


def _detection_position_score(element: dict[str, Any], detection: YoloDetection) -> float:
    words = _normalized_words(_target_text(element))
    y1, x1, y2, x2 = detection.bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = _detection_area(detection)
    score = detection.confidence * 1000.0
    if words.intersection({"left", "leftmost"}):
        score += 1000.0 - cx
    if words.intersection({"right", "rightmost"}):
        score += cx
    if words.intersection({"top", "upper", "above"}):
        score += 1000.0 - cy
    if words.intersection({"bottom", "lower", "below"}):
        score += cy
    if words.intersection({"center", "centered", "central", "middle"}):
        score += 1000.0 - abs(cx - 500.0) - abs(cy - 500.0)
    if words.intersection({"large", "largest", "big", "biggest"}):
        score += area / 1000.0
    if words.intersection({"small", "smallest", "tiny"}):
        score -= area / 1000.0
    return score


def yoloe_prompt_for_element(element: dict[str, Any], max_chars: int = 120) -> str:
    if element.get("type") == "text":
        visible_text = str(element.get("text", "")).strip()
        if visible_text:
            text = f'visible text "{visible_text}"'
        else:
            text = str(element.get("desc", "")).strip() or "visible text"
    else:
        text = str(element.get("desc", "")).strip() or "object"

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or text[:max_chars].strip()


def yoloe_prompts_for_targets(
    elements: list[dict[str, Any]],
    indices: list[int],
) -> tuple[list[str], dict[int, str]]:
    prompts: list[str] = []
    prompt_by_index: dict[int, str] = {}
    seen: set[str] = set()
    for index in indices:
        prompt = yoloe_prompt_for_element(elements[index])
        prompt_by_index[index] = prompt
        key = _normalized_phrase(prompt)
        if not key or key in seen:
            continue
        prompts.append(prompt)
        seen.add(key)
    return prompts, prompt_by_index


def match_yoloe_detections_to_targets(
    elements: list[dict[str, Any]],
    indices: list[int],
    detections: list[YoloDetection],
    prompt_by_index: dict[int, str],
) -> tuple[dict[int, list[int] | None], dict[int, str]]:
    located: dict[int, list[int] | None] = {}
    reasons: dict[int, str] = {}
    used_detection_indices: set[int] = set()
    detections_by_prompt: dict[str, list[YoloDetection]] = {}
    for detection in detections:
        detections_by_prompt.setdefault(_detection_class_key(detection), []).append(detection)

    for index in indices:
        prompt = prompt_by_index.get(index, "")
        prompt_key = _normalized_phrase(prompt)
        if not prompt_key:
            located[index] = None
            reasons[index] = "YOLOE-26 prompt unavailable"
            continue

        candidates = [
            detection
            for detection in detections_by_prompt.get(prompt_key, [])
            if detection.index not in used_detection_indices
        ]
        if not candidates:
            located[index] = None
            reasons[index] = "YOLOE-26 detected no matching prompt"
            continue

        chosen = max(candidates, key=lambda detection: _detection_position_score(elements[index], detection))
        located[index] = chosen.bbox
        used_detection_indices.add(chosen.index)
    return located, reasons


def parse_batch_bboxes_with_reasons(
    raw: str,
    settings: CaptioningSettings | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, list[int] | None], dict[str, str]]:
    if settings is None:
        parsed = extract_json(raw)
    else:
        parsed = parse_json_with_repair(
            settings=settings,
            task="bbox",
            raw_output=raw,
            expected='a compact JSON object shaped like {"bboxes":{"0":[x1,y1,x2,y2],"1":null}}',
            max_tokens=settings.max_tokens_bboxes,
            progress=progress,
        )
    if "bboxes" in parsed and isinstance(parsed["bboxes"], dict):
        raw_map = parsed["bboxes"]
    elif "targets" in parsed and isinstance(parsed["targets"], list):
        raw_map = {}
        for item in parsed["targets"]:
            if not isinstance(item, dict):
                continue
            target_id = item.get("id")
            if target_id is not None:
                raw_map[str(target_id)] = item.get("bbox") if item.get("found", True) else None
    else:
        raise AutoCaptionError(f"Response must contain a bboxes object. Got keys: {list(parsed.keys())}")

    out: dict[str, list[int] | None] = {}
    reasons: dict[str, str] = {}
    for key, value in raw_map.items():
        key_text = str(key)
        if value is None:
            out[key_text] = None
            reasons[key_text] = "model returned null"
            continue
        bbox = bbox_xyxy_to_yxyx(value)
        out[key_text] = bbox
        if bbox is None:
            reasons[key_text] = "model returned invalid bbox"
    return out, reasons


def parse_batch_bboxes(raw: str) -> dict[str, list[int] | None]:
    return parse_batch_bboxes_with_reasons(raw)[0]


def should_try_bbox(element: dict[str, Any]) -> bool:
    element_type = element.get("type")
    if element_type not in {"obj", "text"}:
        return False

    desc = str(element.get("desc", "")).lower()
    words = re.findall(r"[a-z0-9]+", desc)
    dense_terms = {
        "crowd",
        "crowds",
        "starfield",
        "stars",
        "particles",
        "confetti",
        "field of",
        "background",
        "sky",
        "clouds",
        "grass field",
        "water surface",
    }
    if any(term in desc for term in dense_terms):
        return False

    vague_dense_terms = {"pattern", "patterns", "texture", "textures"}
    if any(term in words for term in vague_dense_terms):
        concrete_terms = {
            "animal",
            "arm",
            "body",
            "boy",
            "car",
            "cat",
            "chair",
            "child",
            "dog",
            "door",
            "face",
            "frame",
            "girl",
            "hand",
            "head",
            "headband",
            "holder",
            "leg",
            "man",
            "pants",
            "panties",
            "person",
            "roll",
            "shirt",
            "shelf",
            "sign",
            "sink",
            "table",
            "tattoo",
            "toilet",
            "top",
            "vehicle",
            "woman",
        }
        return any(term in words for term in concrete_terms)

    return True


def bbox_target_indices_with_reasons(
    elements: list[dict[str, Any]],
    settings: CaptioningSettings,
) -> tuple[list[int], dict[int, str]]:
    to_locate: list[int] = []
    skipped: dict[int, str] = {}
    for index, element in enumerate(elements):
        if element.get("type") not in {"obj", "text"}:
            skipped[index] = "not an obj/text element"
            continue
        has_bbox = normalize_bbox(element.get("bbox")) is not None
        if has_bbox and not settings.overwrite_bboxes:
            skipped[index] = "existing bbox kept"
            continue
        if settings.filter_bbox_targets and not should_try_bbox(element):
            skipped[index] = "filtered as vague/ambient"
            continue
        to_locate.append(index)
    return to_locate, skipped


def bbox_target_indices(elements: list[dict[str, Any]], settings: CaptioningSettings) -> list[int]:
    to_locate, _skipped = bbox_target_indices_with_reasons(elements, settings)
    return to_locate


def make_localization_context(data: dict[str, Any], max_chars: int) -> str:
    context: dict[str, Any] = {}
    high = data.get("high_level_description")
    if isinstance(high, str) and high.strip():
        context["high_level_description"] = high.strip()
    comp = data.get("compositional_deconstruction")
    if isinstance(comp, dict):
        background = comp.get("background")
        if isinstance(background, str) and background.strip():
            context["background"] = background.strip()

    text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def build_targets_for_chunk(elements: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for index in indices:
        element = elements[index]
        element_type = element.get("type", "obj")
        target: dict[str, Any] = {
            "id": str(index),
            "type": element_type if element_type in {"obj", "text"} else "obj",
            "desc": str(element.get("desc", "")).strip(),
        }
        if element_type == "text":
            text = str(element.get("text", "")).strip()
            if text:
                target["text"] = text
        targets.append(target)
    return targets


def ordered_element_with_bbox(
    element: dict[str, Any],
    bbox: list[int] | None,
    keep_existing_if_no_new: bool,
) -> dict[str, Any]:
    element_type = element.get("type", "obj")
    if element_type not in {"obj", "text"}:
        element_type = "obj"

    existing_bbox = normalize_bbox(element.get("bbox")) if "bbox" in element else None
    final_bbox = bbox if bbox is not None else (existing_bbox if keep_existing_if_no_new else None)
    desc = str(element.get("desc", "")).strip()

    if element_type == "text":
        out: dict[str, Any] = {"type": "text"}
        if final_bbox is not None:
            out["bbox"] = final_bbox
        out["text"] = str(element.get("text", "")).strip()
        out["desc"] = desc
        return out

    out = {"type": "obj"}
    if final_bbox is not None:
        out["bbox"] = final_bbox
    out["desc"] = desc
    return out


def add_yoloe_bboxes_to_caption(
    settings: CaptioningSettings,
    image_path: Path,
    caption: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], int, int, dict[str, int]]:
    data = normalize_caption(caption)
    elements = data.get("compositional_deconstruction", {}).get("elements", [])
    elements = [element for element in elements if isinstance(element, dict)]

    to_locate, skipped_before = bbox_target_indices_with_reasons(elements, settings)
    located: dict[int, list[int] | None] = {}
    skipped_reasons: dict[int, str] = dict(skipped_before)
    attempted = len(to_locate)

    if to_locate:
        prompts, prompt_by_index = yoloe_prompts_for_targets(elements, to_locate)
        if not prompts:
            for index in to_locate:
                skipped_reasons[index] = "YOLOE-26 prompt unavailable"
        else:
            if progress:
                progress(f"YOLOE-26 prompts: {len(prompts)} class(es).")
            model_name = settings.yolo_bbox_model.strip() or DEFAULT_YOLOE26_BBOX_MODEL
            detections, _class_names = yolo_detections_for_image(
                settings,
                image_path,
                progress=progress,
                class_prompts=prompts,
                model_name=model_name,
                label="YOLOE-26",
            )
            if progress:
                progress(f"YOLOE-26 detections: {len(detections)} object(s).")
            located, response_reasons = match_yoloe_detections_to_targets(
                elements,
                to_locate,
                detections,
                prompt_by_index,
            )
            skipped_reasons.update(response_reasons)

    added = sum(1 for bbox in located.values() if bbox is not None)
    new_elements = [
        ordered_element_with_bbox(
            element,
            bbox=located.get(index),
            keep_existing_if_no_new=not settings.overwrite_bboxes,
        )
        for index, element in enumerate(elements)
    ]
    data["compositional_deconstruction"]["elements"] = new_elements
    reason_counts: dict[str, int] = {}
    for reason in skipped_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return normalize_caption(data), attempted, added, reason_counts


def add_bboxes_to_caption(
    settings: CaptioningSettings,
    image_path: Path,
    caption: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], int, int, dict[str, int]]:
    if settings.bbox_backend == BBOX_BACKEND_YOLOE26:
        return add_yoloe_bboxes_to_caption(settings, image_path, caption, progress=progress)

    data = normalize_caption(caption)
    elements = data.get("compositional_deconstruction", {}).get("elements", [])
    elements = [element for element in elements if isinstance(element, dict)]

    to_locate, skipped_before = bbox_target_indices_with_reasons(elements, settings)

    config = runtime_config_for_task(settings, "bbox")
    prompts = load_prompts()
    context_json = make_localization_context(data, settings.context_chars)
    located: dict[int, list[int] | None] = {}
    skipped_reasons: dict[int, str] = dict(skipped_before)
    attempted = 0
    added = 0

    for chunk in chunk_list(to_locate, settings.max_targets_per_call):
        if not chunk:
            continue
        prompt = format_prompt(
            prompts["bbox_user"],
            context_json=context_json,
            targets_json=json.dumps(build_targets_for_chunk(elements, chunk), ensure_ascii=False, separators=(",", ":")),
        )
        attempted += len(chunk)
        raw = chat_vision(
            settings=settings,
            model=config.api_model,
            image_path=image_path,
            system=prompts["bbox_system"],
            user=prompt,
            max_tokens=settings.max_tokens_bboxes,
            temperature=0.0,
        )
        bbox_map, response_reasons = parse_batch_bboxes_with_reasons(raw, settings=settings, progress=progress)
        for index in chunk:
            key = str(index)
            if key not in bbox_map:
                located[index] = None
                skipped_reasons[index] = "model omitted target id"
                continue
            bbox = bbox_map.get(key)
            located[index] = bbox
            if bbox is not None:
                added += 1
            else:
                skipped_reasons[index] = response_reasons.get(key, "model returned no bbox")

    new_elements = [
        ordered_element_with_bbox(
            element,
            bbox=located.get(index),
            keep_existing_if_no_new=not settings.overwrite_bboxes,
        )
        for index, element in enumerate(elements)
    ]
    data["compositional_deconstruction"]["elements"] = new_elements
    reason_counts: dict[str, int] = {}
    for reason in skipped_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return normalize_caption(data), attempted, added, reason_counts
