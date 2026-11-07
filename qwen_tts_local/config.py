# SPDX-License-Identifier: Apache-2.0
"""Portable local-demo configuration and explicit process-local resource settings."""

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path


def user_directory(variable, fallback):
    return Path(os.environ.get(variable) or Path.home() / fallback) / "qwen3-tts"


def default_config_path():
    return user_directory("XDG_CONFIG_HOME", ".config") / "local.json"


@dataclass(frozen=True)
class Settings:
    checkpoint: str = ""
    state_dir: str = ""
    projects_dir: str = ""
    ui_language: str = "en"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    port: int = 7867
    threads: int = 2
    interop_threads: int = 1
    gpu_memory_fraction: float = 0.7
    max_new_tokens: int = 256
    section_max_new_tokens: int = 256
    offline: bool = True
    open_browser: bool = True
    startup_wait_seconds: int = 120
    stop_wait_seconds: int = 10

    def validate(self, require_checkpoint=True):
        for name in (
            "checkpoint",
            "state_dir",
            "projects_dir",
            "device",
            "dtype",
            "attention",
            "ui_language",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name}: expected text (需要字符串).")
        if require_checkpoint and not self.checkpoint.strip():
            raise ValueError(
                "Set a checkpoint path or cached model ID (请设置模型路径或已缓存的模型 ID)."
            )
        for name in (
            "threads",
            "interop_threads",
            "max_new_tokens",
            "section_max_new_tokens",
            "startup_wait_seconds",
            "stop_wait_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65536:
                raise ValueError(
                    f"{name}: expected an integer from 1 to 65536 (需要 1 至 65536 的整数)."
                )
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError("Port must be 1024–65535 (端口必须为 1024 至 65535).")
        if self.ui_language not in ("en", "bilingual"):
            raise ValueError(
                "UI language must be en or bilingual (界面语言必须为 en 或 bilingual)."
            )
        if self.dtype not in (
            "bfloat16",
            "float16",
            "float32",
        ) or self.attention not in ("sdpa", "eager", "flash_attention_2"):
            raise ValueError(
                "Unsupported dtype or attention setting (不支持的数据类型或注意力设置)."
            )
        if self.device != "cpu" and not (
            self.device == "cuda"
            or self.device.startswith("cuda:")
            and self.device[5:].isdigit()
        ):
            raise ValueError(
                "Use cpu or one cuda device (请使用 cpu 或单个 cuda 设备)."
            )
        if (
            isinstance(self.gpu_memory_fraction, bool)
            or not isinstance(self.gpu_memory_fraction, (float, int))
            or not math.isfinite(self.gpu_memory_fraction)
            or not 0 < self.gpu_memory_fraction <= 1
        ):
            raise ValueError(
                "GPU memory fraction must be in (0, 1] (GPU 内存比例必须大于 0 且不超过 1)."
            )
        for name in ("offline", "open_browser"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name}: expected a boolean (需要布尔值).")
        return self

    @property
    def state_path(self):
        return (
            Path(self.state_dir).expanduser().resolve()
            if self.state_dir
            else user_directory("XDG_STATE_HOME", ".local/state").resolve()
        )

    @property
    def projects_path(self):
        return (
            Path(self.projects_dir).expanduser().resolve()
            if self.projects_dir
            else user_directory("XDG_DATA_HOME", ".local/share").resolve() / "projects"
        )

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def demo_args(self):
        self.validate()
        return [
            "--checkpoint",
            self.checkpoint,
            "--ip",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--no-share",
            "--concurrency",
            "1",
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--attention",
            self.attention,
            "--ui-language",
            self.ui_language,
            "--threads",
            str(self.threads),
            "--interop-threads",
            str(self.interop_threads),
            "--gpu-memory-fraction",
            str(self.gpu_memory_fraction),
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--long-text-max-new-tokens",
            str(self.section_max_new_tokens),
            "--long-text-projects-dir",
            str(self.projects_path),
        ]

    def environment(self):
        env = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            env[name] = str(self.threads)
        env.update(
            GRADIO_ANALYTICS_ENABLED="False",
            HF_HUB_DISABLE_TELEMETRY="1",
            TOKENIZERS_PARALLELISM="false",
        )
        # Do not alter any user's global environment or settings.
        env["HF_HUB_OFFLINE"] = "1" if self.offline else "0"
        env["TRANSFORMERS_OFFLINE"] = "1" if self.offline else "0"
        return env


def load_settings(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(Settings)}:
        raise ValueError(
            "Unknown configuration keys or invalid JSON object (配置键未知或 JSON 对象无效)."
        )
    Settings(**data).validate(require_checkpoint=False)
    for name in ("state_dir", "projects_dir"):
        if data.get(name):
            value = Path(data[name]).expanduser()
            data[name] = str(value if value.is_absolute() else path.parent / value)
    # Model IDs stay intact; explicit relative model paths resolve beside the config.
    if isinstance(data.get("checkpoint"), str) and data["checkpoint"].startswith(
        ("./", "../", "~")
    ):
        value = Path(data["checkpoint"]).expanduser()
        data["checkpoint"] = str(
            (value if value.is_absolute() else path.parent / value).resolve()
        )
    return Settings(**data).validate(require_checkpoint=False)


def write_template(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(asdict(Settings()), out, indent=2)
        out.write("\n")


def apply_runtime(
    torch, device, threads=None, interop_threads=None, gpu_memory_fraction=None
):
    """Explicit settings in the model process only; no memory-pressure watchdog."""
    for name, value in (("threads", threads), ("interop_threads", interop_threads)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be positive ({name} 必须为正整数).")
    if gpu_memory_fraction is not None and (
        not math.isfinite(gpu_memory_fraction) or not 0 < gpu_memory_fraction <= 1
    ):
        raise ValueError(
            "GPU memory fraction must be in (0, 1] (GPU 内存比例必须大于 0 且不超过 1)."
        )
    if threads is not None:
        torch.set_num_threads(threads)
    if interop_threads is not None:
        torch.set_num_interop_threads(interop_threads)
    if gpu_memory_fraction is not None and str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            gpu_memory_fraction, torch.device(device)
        )
