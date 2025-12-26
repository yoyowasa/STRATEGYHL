from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_MODES = {"research", "mm_sim", "mm_replay", "mm_shadow", "mm_live"}


class ConfigError(ValueError):
    """設定の検証・読み込みで失敗した際の例外。"""


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    output_dir: str = "outputs"
    log_dir: str = "logs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "baseline"
    risk_limit: float = 0.1
    leverage: float = 1.0
    max_positions: int = 10
    extra_params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(frozen=True)
class HLMMConfig:
    mode: str
    paths: PathsConfig
    strategy: StrategyConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "paths": self.paths.to_dict(),
            "strategy": self.strategy.to_dict(),
        }


def _ensure_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} は空でない文字列である必要があります")
    return value.strip()


def _normalize_paths(raw: Any) -> PathsConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("paths はマッピングである必要があります")
    if "data_dir" not in raw:
        raise ConfigError("paths.data_dir は必須です")
    data_dir = _ensure_str(raw["data_dir"], "paths.data_dir")
    output_dir = _ensure_str(raw.get("output_dir", "outputs"), "paths.output_dir")
    log_dir = _ensure_str(raw.get("log_dir", "logs"), "paths.log_dir")
    return PathsConfig(data_dir=data_dir, output_dir=output_dir, log_dir=log_dir)


def _normalize_strategy(raw: Any) -> StrategyConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("strategy はマッピングである必要があります")
    name = _ensure_str(raw.get("name", "baseline"), "strategy.name")

    risk_limit = raw.get("risk_limit", 0.1)
    if not isinstance(risk_limit, (int, float)):
        raise ConfigError("strategy.risk_limit は数値である必要があります")

    leverage = raw.get("leverage", 1.0)
    if not isinstance(leverage, (int, float)):
        raise ConfigError("strategy.leverage は数値である必要があります")

    max_positions = raw.get("max_positions", 10)
    if not isinstance(max_positions, int):
        raise ConfigError("strategy.max_positions は整数である必要があります")

    extra_params = raw.get("extra_params", {})
    if not isinstance(extra_params, Mapping):
        raise ConfigError("strategy.extra_params はマッピングである必要があります")

    return StrategyConfig(
        name=name,
        risk_limit=float(risk_limit),
        leverage=float(leverage),
        max_positions=max_positions,
        extra_params=dict(extra_params),
    )


def normalize_config(raw: Any) -> HLMMConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("設定はマッピングである必要があります")
    if "mode" not in raw:
        raise ConfigError("mode は必須です")

    mode = _ensure_str(raw["mode"], "mode")
    if mode not in ALLOWED_MODES:
        raise ConfigError(f"mode は {sorted(ALLOWED_MODES)} のいずれかである必要があります")

    if "paths" not in raw:
        raise ConfigError("paths は必須です")
    paths = _normalize_paths(raw["paths"])

    strategy = _normalize_strategy(raw.get("strategy"))
    return HLMMConfig(mode=mode, paths=paths, strategy=strategy)


def load_config(path: str | Path) -> HLMMConfig:
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {file_path}")
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"設定ファイルの読み込みに失敗しました: {exc}") from exc

    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAMLのパースに失敗しました: {exc}") from exc

    if raw is None:
        raise ConfigError("設定ファイルが空です")

    return normalize_config(raw)
