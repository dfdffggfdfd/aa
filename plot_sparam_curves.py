#!/usr/bin/env python3
"""按配置绘制 Touchstone（.s16p）的幅度和相位。

规则：
  - plot = reflection / transmission / all，决定画反射、透射，或两者都画。
  - 反射端口为 all：全部反射叠在同一张图（上幅度、下相位）。16 端口就是 16 条曲线。
  - 其他情况（指定若干反射端口，或任意透射）：每个 S 参数单独一张图。

透射端口对按 [接收, 发射] 记，即 S接收,发射。例如 [2, 1] 是 S2,1。

运行：
  python plot_sparam_curves.py
  python plot_sparam_curves.py --config config.yaml
"""

from __future__ import annotations

import argparse
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from matplotlib import font_manager

# 装配图：丝印号 = Touchstone 端口号。图像右 = 患者左。
LEFT_BRAIN_PORTS = (2, 1, 16, 15, 14, 13, 12, 11)
RIGHT_BRAIN_PORTS = (3, 4, 5, 6, 7, 8, 9, 10)
RING_CW_PORTS = (2, 1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3)

ANT_FROM_TS_PRESETS: dict[str, np.ndarray] = {
    # 丝印 = 端口
    "cad": np.arange(1, 17),
    "identity": np.arange(1, 17),
    # 下标沿环向。图上的编号仍是丝印号。
    "cad_ring": np.array(RING_CW_PORTS, dtype=int),
    "real_sym": np.array([4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]),
    "sim_sym": np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10]),
}

_PORT_TO_HEMI = {
    **{p: "L" for p in LEFT_BRAIN_PORTS},
    **{p: "R" for p in RIGHT_BRAIN_PORTS},
}

_COLORS = [
    "#c0392b", "#e74c3c", "#d35400", "#e67e22",
    "#f39c12", "#27ae60", "#1abc9c", "#16a085",
    "#2980b9", "#3498db", "#8e44ad", "#9b59b6",
    "#2c3e50", "#7f8c8d", "#1e8449", "#2471a3",
]

_PLOT_ALIASES = {
    "reflection": "reflection",
    "refl": "reflection",
    "反射": "reflection",
    "transmission": "transmission",
    "trans": "transmission",
    "透射": "transmission",
    "all": "all",
    "both": "all",
    "全部": "all",
}

_ALL_WORDS = {"all", "全部", "所有", "*"}
_METRICS = ("mag", "phase", "re", "im")
_PAIR_RE = re.compile(r"^S?\s*(\d+)\s*,\s*S?\s*(\d+)\s*$", re.IGNORECASE)
_CONFIG_KEYS = {
    "root",
    "out_dir",
    "freq_min_ghz",
    "freq_max_ghz",
    "apply_antenna_map",
    "map_preset",
    "plot",
    "reflection_ports",
    "transmission_pairs",
    "metrics",
    "dpi",
    "skip_existing",
    "suffixes",
    "label_prefix",
}


class ConfigError(Exception):
    """配置无法解释，或和当前文件的端口数对不上。"""


@dataclass(frozen=True)
class SideSelect:
    """某一侧端口。mode: absent / all / list。"""

    mode: str
    ports: tuple[int, ...] = ()


@dataclass(frozen=True)
class TransmissionSpec:
    """kind=all 画全部非对角；kind=select 只画 pairs，并在写了 tx/rx 时做笛卡尔展开。"""

    kind: str
    pairs: tuple[tuple[int, int], ...] = ()
    tx: SideSelect = SideSelect("absent")
    rx: SideSelect = SideSelect("absent")


@dataclass(frozen=True)
class Config:
    root: Path
    out_dir: Path
    freq_min_ghz: float | None
    freq_max_ghz: float | None
    apply_antenna_map: bool
    map_preset: str
    plot: str
    reflection_ports: str | tuple[int, ...]
    transmission: TransmissionSpec
    metrics: tuple[str, ...]
    dpi: int
    skip_existing: bool
    suffixes: tuple[str, ...]
    label_prefix: str | None = None


@dataclass(frozen=True)
class PlotTask:
    """kind: reflection_overlay / reflection / transmission。pairs 为丝印端口 (rx, tx)。"""

    kind: str
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RunStats:
    files: int
    ok: int
    skip: int
    fail: int


def _check_geometry() -> None:
    if sorted(LEFT_BRAIN_PORTS + RIGHT_BRAIN_PORTS) != list(range(1, 17)):
        raise RuntimeError("左右脑端口分区没有覆盖 1..16")
    if sorted(RING_CW_PORTS) != list(range(1, 17)):
        raise RuntimeError("环向端口不是 1..16 的排列")
    for name, vec in ANT_FROM_TS_PRESETS.items():
        if sorted(int(v) for v in vec) != list(range(1, 17)):
            raise RuntimeError(f"映射 {name} 不是 1..16 的排列")


_check_geometry()

_FONT_READY = False
_FONT_NAME: str | None = None


def setup_chinese_font() -> str | None:
    global _FONT_READY, _FONT_NAME
    if _FONT_READY:
        return _FONT_NAME
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Micro Hei Mono",
        "Droid Sans Fallback",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            _FONT_NAME = name
            _FONT_READY = True
            return name
    _FONT_READY = True
    return None


def hemi_tag(port_1based: int) -> str:
    return _PORT_TO_HEMI.get(int(port_1based), "?")


def silk_port_for_index(idx0: int, ant_from_ts: np.ndarray | None) -> int:
    """画图下标 -> 丝印 / Touchstone 端口号（1-based）。"""
    if ant_from_ts is None:
        return idx0 + 1
    return int(ant_from_ts[idx0] + 1)


def port_index(silk: int, ant_from_ts: np.ndarray | None, n_ports: int) -> int:
    if not 1 <= int(silk) <= n_ports:
        raise ConfigError(f"端口 {silk} 超出范围，当前文件只有 {n_ports} 个端口")
    if ant_from_ts is None:
        return int(silk) - 1
    hits = np.flatnonzero(ant_from_ts + 1 == int(silk))
    if len(hits) != 1:
        raise ConfigError(f"丝印端口 {silk} 在当前映射里不存在或不唯一")
    return int(hits[0])


def remap_s(s: np.ndarray, ant_from_ts: np.ndarray) -> np.ndarray:
    """Touchstone 端口序 -> 物理天线序。S_phys[rx, tx] = S_touch[map[rx], map[tx]]。"""
    return s[:, ant_from_ts, :][:, :, ant_from_ts]


def ant_from_ts_0based(preset: str, n_ports: int) -> np.ndarray:
    if preset not in ANT_FROM_TS_PRESETS:
        known = ", ".join(ANT_FROM_TS_PRESETS)
        raise ConfigError(f"未知 map_preset={preset}，可选 {known}")
    vec = ANT_FROM_TS_PRESETS[preset].astype(int)
    if len(vec) != n_ports:
        raise ConfigError(
            f"map_preset={preset} 固定 {len(vec)} 个端口，当前文件是 {n_ports} 端口"
        )
    return vec - 1


def _dedupe(items: list) -> tuple:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _as_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{key} 需要 true 或 false，收到 {value!r}")


def _as_optional_float(value, key: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} 需要数字或 null，收到 {value!r}") from exc


def _is_all_word(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in _ALL_WORDS


def _parse_port_atom(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigError(f"{key} 里的端口号无法识别：{value!r}")
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        raise ConfigError(f"{key} 里的端口号无法识别：{value!r}")
    port = int(text)
    if port < 1:
        raise ConfigError(f"{key} 的端口号从 1 开始，收到 {port}")
    return port


def parse_port_list(value, key: str) -> tuple[int, ...]:
    """解析端口列表。支持 [1, 2, 5]、\"1-8,11\"、以及列表里夹带 \"10-12\"。"""
    tokens: list[str] = []
    if isinstance(value, str):
        tokens = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and "," in item:
                tokens.extend(part.strip() for part in item.split(",") if part.strip())
            elif isinstance(item, str):
                tokens.append(item.strip())
            else:
                tokens.append(str(_parse_port_atom(item, key)))
    else:
        raise ConfigError(f"{key} 需要 all、端口列表或范围字符串，收到 {value!r}")

    if not tokens:
        raise ConfigError(f"{key} 是空的")

    ports: list[int] = []
    for token in tokens:
        if re.fullmatch(r"\d+\s*-\s*\d+", token):
            left, right = (part.strip() for part in token.split("-", 1))
            start, end = int(left), int(right)
            if end < start:
                raise ConfigError(f"{key} 的范围要从小到大，收到 {token}")
            ports.extend(range(start, end + 1))
        else:
            ports.append(_parse_port_atom(token, key))
    return _dedupe(ports)


def parse_reflection_ports(value) -> str | tuple[int, ...]:
    if value is None or _is_all_word(value):
        return "all"
    return parse_port_list(value, "reflection_ports")


def _parse_pair(value, key: str) -> tuple[int, int]:
    if isinstance(value, str):
        match = _PAIR_RE.fullmatch(value.strip())
        if not match:
            raise ConfigError(f"{key} 的端口对无法识别：{value!r}，应写成 [2, 1] 或 \"S2,1\"")
        rx, tx = int(match.group(1)), int(match.group(2))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        rx = _parse_port_atom(value[0], key)
        tx = _parse_port_atom(value[1], key)
    else:
        raise ConfigError(f"{key} 的端口对无法识别：{value!r}，应写成 [接收, 发射]")
    if rx == tx:
        raise ConfigError(f"{key} 里 [{rx}, {tx}] 是反射。反射请写 reflection_ports，透射的接收和发射不能相同")
    if rx < 1 or tx < 1:
        raise ConfigError(f"{key} 的端口号从 1 开始，收到 [{rx}, {tx}]")
    return rx, tx


def _parse_side(value, key: str) -> SideSelect:
    if value is None or _is_all_word(value):
        return SideSelect("all")
    ports = parse_port_list(value, key)
    return SideSelect("list", ports)


def parse_transmission(value) -> TransmissionSpec:
    if value is None or _is_all_word(value):
        return TransmissionSpec(kind="all")
    if isinstance(value, dict):
        unknown = set(value) - {"pairs", "tx", "rx"}
        if unknown:
            raise ConfigError(
                "transmission_pairs 只接受 pairs / tx / rx，未知项："
                + ", ".join(sorted(unknown))
            )
        raw_pairs = value.get("pairs", []) or []
        if isinstance(raw_pairs, str) or (
            isinstance(raw_pairs, list) and len(raw_pairs) == 2 and not isinstance(raw_pairs[0], (list, tuple, str))
        ):
            pair_items = [raw_pairs]
        elif isinstance(raw_pairs, list):
            pair_items = raw_pairs
        else:
            raise ConfigError(f"transmission_pairs.pairs 无法识别：{raw_pairs!r}")
        pairs = _dedupe([_parse_pair(item, "transmission_pairs.pairs") for item in pair_items])
        tx = _parse_side(value["tx"], "transmission_pairs.tx") if "tx" in value else SideSelect("absent")
        rx = _parse_side(value["rx"], "transmission_pairs.rx") if "rx" in value else SideSelect("absent")
        if not pairs and tx.mode == "absent" and rx.mode == "absent":
            raise ConfigError("transmission_pairs 是空的，写 all、端口对，或 tx/rx")
        return TransmissionSpec(kind="select", pairs=pairs, tx=tx, rx=rx)
    if isinstance(value, str):
        return TransmissionSpec(kind="select", pairs=(_parse_pair(value, "transmission_pairs"),))
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, (int, str)) and not isinstance(item, bool) for item in value):
            if isinstance(value[0], str) and not re.fullmatch(r"\d+", value[0].strip()):
                pairs = [_parse_pair(item, "transmission_pairs") for item in value]
            else:
                try:
                    pairs = [_parse_pair(value, "transmission_pairs")]
                except ConfigError:
                    pairs = [_parse_pair(item, "transmission_pairs") for item in value]
        else:
            pairs = [_parse_pair(item, "transmission_pairs") for item in value]
        if not pairs:
            raise ConfigError("transmission_pairs 是空的")
        return TransmissionSpec(kind="select", pairs=_dedupe(pairs))
    raise ConfigError(f"transmission_pairs 无法识别：{value!r}")


def load_config(path: Path) -> Config:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("需要 PyYAML：pip install pyyaml") from exc

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"找不到配置文件：{path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ConfigError(f"配置文件不是合法的 YAML：{path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("配置文件的顶层必须是键值")

    unknown = set(data) - _CONFIG_KEYS
    if unknown:
        raise ConfigError("配置里有无法识别的项：" + ", ".join(sorted(unknown)))

    if "root" not in data or "out_dir" not in data:
        raise ConfigError("配置必须写 root 和 out_dir")

    freq_min = _as_optional_float(data.get("freq_min_ghz"), "freq_min_ghz")
    freq_max = _as_optional_float(data.get("freq_max_ghz"), "freq_max_ghz")
    if freq_min is not None and freq_max is not None and freq_min > freq_max:
        raise ConfigError(f"freq_min_ghz ({freq_min}) 大于 freq_max_ghz ({freq_max})")

    plot_raw = data.get("plot", "all")
    if not isinstance(plot_raw, str) or plot_raw.strip().lower() not in _PLOT_ALIASES:
        raise ConfigError("plot 只能是 reflection / transmission / all（或 反射 / 透射 / 全部）")
    plot = _PLOT_ALIASES[plot_raw.strip().lower()]

    metrics_raw = data.get("metrics", ["mag", "phase"])
    if isinstance(metrics_raw, str):
        metrics_raw = [metrics_raw]
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise ConfigError("metrics 至少要有一项，例如 [mag, phase]")
    metrics = []
    for name in metrics_raw:
        if not isinstance(name, str) or name.strip().lower() not in _METRICS:
            raise ConfigError(f"未知 metrics 项 {name!r}，可选 mag / phase / re / im")
        metrics.append(name.strip().lower())
    metrics = _dedupe(metrics)

    dpi = data.get("dpi", 120)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ConfigError(f"dpi 需要正整数，收到 {dpi!r}")

    suffixes_raw = data.get("suffixes", [".s16p"])
    if isinstance(suffixes_raw, str):
        suffixes_raw = [suffixes_raw]
    if not isinstance(suffixes_raw, list) or not suffixes_raw:
        raise ConfigError("suffixes 至少要有一个后缀，例如 [.s16p]")
    suffixes = []
    for suffix in suffixes_raw:
        if not isinstance(suffix, str) or not suffix.strip():
            raise ConfigError(f"suffixes 项无法识别：{suffix!r}")
        text = suffix.strip().lower()
        if not text.startswith("."):
            text = "." + text
        suffixes.append(text)

    preset = data.get("map_preset", "cad")
    if not isinstance(preset, str) or preset.strip() not in ANT_FROM_TS_PRESETS:
        known = ", ".join(ANT_FROM_TS_PRESETS)
        raise ConfigError(f"未知 map_preset={preset!r}，可选 {known}")

    label = data.get("label_prefix")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ConfigError(f"label_prefix 需要非空字符串或省略，收到 {label!r}")

    return Config(
        root=Path(str(data["root"])).expanduser(),
        out_dir=Path(str(data["out_dir"])).expanduser(),
        freq_min_ghz=freq_min,
        freq_max_ghz=freq_max,
        apply_antenna_map=_as_bool(data.get("apply_antenna_map", True), "apply_antenna_map"),
        map_preset=preset.strip(),
        plot=plot,
        reflection_ports=parse_reflection_ports(data.get("reflection_ports", "all")),
        transmission=parse_transmission(data.get("transmission_pairs", "all")),
        metrics=metrics,
        dpi=dpi,
        skip_existing=_as_bool(data.get("skip_existing", True), "skip_existing"),
        suffixes=tuple(suffixes),
        label_prefix=None if label is None else label.strip(),
    )


def freq_mask(freqs_ghz: np.ndarray, freq_min: float | None, freq_max: float | None) -> np.ndarray:
    # 端点按“包含”处理，并留出 1 kHz 量级的余量，避免 GHz/Hz 换算把正好落在端点上的频点挤出去。
    tol = 1e-6
    mask = np.ones(len(freqs_ghz), dtype=bool)
    if freq_min is not None:
        mask &= freqs_ghz >= freq_min - tol
    if freq_max is not None:
        mask &= freqs_ghz <= freq_max + tol
    return mask


def load_touchstone(
    path: Path,
    freq_min_ghz: float | None,
    freq_max_ghz: float | None,
    apply_antenna_map: bool,
    map_preset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    network = rf.Network(str(path))
    freqs_ghz = np.asarray(network.f, dtype=float) / 1e9
    mask = freq_mask(freqs_ghz, freq_min_ghz, freq_max_ghz)
    if not np.any(mask):
        raise ConfigError(f"频率范围内没有采样点：{path}")
    s = np.asarray(network.s, dtype=complex)[mask]
    if s.ndim != 3 or s.shape[1] != s.shape[2]:
        raise ConfigError(f"S 矩阵形状异常：{s.shape} ({path.name})")
    n_ports = int(s.shape[1])
    ant_from_ts = None
    if apply_antenna_map:
        ant_from_ts = ant_from_ts_0based(map_preset, n_ports)
        s = remap_s(s, ant_from_ts)
    return freqs_ghz[mask], s, ant_from_ts


def _silks_in_index_order(n_ports: int, ant_from_ts: np.ndarray | None) -> list[int]:
    return [silk_port_for_index(i, ant_from_ts) for i in range(n_ports)]


def _side_ports(side: SideSelect, key: str, n_ports: int) -> list[int]:
    """筛选展开用端口号顺序，不用环向下标顺序。环向顺序只留给 transmission_pairs: all。"""
    if side.mode in ("absent", "all"):
        return list(range(1, n_ports + 1))
    for port in side.ports:
        if not 1 <= port <= n_ports:
            raise ConfigError(f"{key} 端口 {port} 超出范围，当前文件只有 {n_ports} 个端口")
    return list(side.ports)


def resolve_transmission_pairs(
    spec: TransmissionSpec,
    n_ports: int,
    ant_from_ts: np.ndarray | None,
) -> list[tuple[int, int]]:
    ordered = _silks_in_index_order(n_ports, ant_from_ts)
    if spec.kind == "all":
        pairs = [(rx, tx) for tx in ordered for rx in ordered if rx != tx]
        return pairs

    pairs = list(spec.pairs)
    if spec.tx.mode != "absent" or spec.rx.mode != "absent":
        # 没写的那一侧，以及写了 all 的那一侧，都按端口号 1..N。写了列表就保持列表顺序。
        txs = _side_ports(spec.tx, "transmission_pairs.tx", n_ports)
        rxs = _side_ports(spec.rx, "transmission_pairs.rx", n_ports)
        for tx in txs:
            if not 1 <= tx <= n_ports:
                raise ConfigError(f"transmission_pairs.tx 端口 {tx} 超出范围，当前文件只有 {n_ports} 个端口")
            for rx in rxs:
                if not 1 <= rx <= n_ports:
                    raise ConfigError(
                        f"transmission_pairs.rx 端口 {rx} 超出范围，当前文件只有 {n_ports} 个端口"
                    )
                if rx != tx:
                    pairs.append((rx, tx))

    checked = []
    for rx, tx in pairs:
        if not 1 <= rx <= n_ports or not 1 <= tx <= n_ports:
            raise ConfigError(f"透射端口对 [{rx}, {tx}] 超出范围，当前文件只有 {n_ports} 个端口")
        if rx == tx:
            raise ConfigError(f"[{rx}, {tx}] 是反射，不能放在 transmission_pairs")
        checked.append((rx, tx))
    return list(_dedupe(checked))


def build_tasks(
    plot: str,
    reflection_ports: str | tuple[int, ...],
    transmission: TransmissionSpec,
    n_ports: int,
    ant_from_ts: np.ndarray | None,
) -> list[PlotTask]:
    tasks: list[PlotTask] = []
    if plot in ("reflection", "all"):
        if reflection_ports == "all":
            silks = _silks_in_index_order(n_ports, ant_from_ts)
            tasks.append(PlotTask("reflection_overlay", tuple((silk, silk) for silk in silks)))
        else:
            for port in reflection_ports:
                if not 1 <= port <= n_ports:
                    raise ConfigError(f"reflection_ports 端口 {port} 超出范围，当前文件只有 {n_ports} 个端口")
                tasks.append(PlotTask("reflection", ((port, port),)))
    if plot in ("transmission", "all"):
        for rx, tx in resolve_transmission_pairs(transmission, n_ports, ant_from_ts):
            tasks.append(PlotTask("transmission", ((rx, tx),)))
    return tasks


def metric_array(s_ij: np.ndarray, name: str) -> tuple[np.ndarray, str, str]:
    s_ij = np.asarray(s_ij, dtype=complex)
    if name == "mag":
        y = 20.0 * np.log10(np.maximum(np.abs(s_ij), 1e-15))
        return y, "幅度", "幅度 (dB)"
    if name == "phase":
        return np.angle(s_ij), "相位", "相位 (rad)"
    if name == "re":
        return np.real(s_ij), "实部", "Re(S)"
    if name == "im":
        return np.imag(s_ij), "虚部", "Im(S)"
    raise ConfigError(f"未知 metrics 项 {name}")


def s_element(
    s: np.ndarray,
    silk_rx: int,
    silk_tx: int,
    ant_from_ts: np.ndarray | None,
) -> np.ndarray:
    n_ports = s.shape[1]
    rx = port_index(silk_rx, ant_from_ts, n_ports)
    tx = port_index(silk_tx, ant_from_ts, n_ports)
    return s[:, rx, tx]


def _freq_note(freq_min: float | None, freq_max: float | None) -> str:
    if freq_min is None and freq_max is None:
        return ""
    if freq_min is None:
        return f"频率 ≤ {freq_max:g} GHz"
    if freq_max is None:
        return f"频率 ≥ {freq_min:g} GHz"
    return f"频率 {freq_min:g}-{freq_max:g} GHz"


def _wrap_title(text: str, width: int = 88) -> str:
    if len(text) <= width:
        return text
    lines: list[str] = []
    current = ""
    for part in text.split("/"):
        piece = part if not current else f"{current}/{part}"
        if current and len(piece) > width:
            lines.append(current)
            current = part
        else:
            current = piece
    if current:
        lines.append(current)
    return "\n".join(lines)


def _style_axis(ax, freq: np.ndarray, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("频率 (GHz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    if len(freq) >= 2:
        ax.set_xlim(float(freq[0]), float(freq[-1]))


def _plot_line(ax, freq: np.ndarray, y: np.ndarray, **style) -> None:
    if len(freq) < 2:
        style = dict(style)
        style.setdefault("marker", "o")
    ax.plot(freq, y, **style)


def draw_task(
    freq: np.ndarray,
    s: np.ndarray,
    task: PlotTask,
    ant_from_ts: np.ndarray | None,
    metrics: tuple[str, ...],
    title: str,
    label_prefix: str,
    freq_min: float | None,
    freq_max: float | None,
) -> plt.Figure:
    setup_chinese_font()
    note = _freq_note(freq_min, freq_max)
    note_suffix = f"  |  {note}" if note else ""
    file_title = _wrap_title(title)
    n_metrics = len(metrics)

    if task.kind == "reflection_overlay":
        fig, axes = plt.subplots(
            n_metrics,
            1,
            figsize=(15.2, 4.5 * n_metrics),
            squeeze=False,
            facecolor="white",
        )
        fig.suptitle(
            f"{file_title}\n反射  {len(task.pairs)} 条叠画  |  L=左脑  R=右脑{note_suffix}",
            fontsize=13,
            fontweight="bold",
        )
        for row, metric in enumerate(metrics):
            ax = axes[row, 0]
            ylabel = ""
            metric_name = metric
            for idx, (silk_rx, silk_tx) in enumerate(task.pairs):
                y, metric_name, ylabel = metric_array(
                    s_element(s, silk_rx, silk_tx, ant_from_ts), metric
                )
                _plot_line(
                    ax,
                    freq,
                    y,
                    color=_COLORS[idx % len(_COLORS)],
                    lw=1.2,
                    label=f"S{silk_rx},{silk_tx} ({label_prefix}{silk_rx}[{hemi_tag(silk_rx)}])",
                )
            _style_axis(ax, freq, f"反射{metric_name}", ylabel)
        fig.tight_layout(rect=(0, 0, 0.78, 0.93))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.79, 0.5),
            fontsize=8,
            framealpha=0.92,
            borderaxespad=0,
        )
        return fig

    silk_rx, silk_tx = task.pairs[0]
    kind_name = "反射" if task.kind == "reflection" else "透射"
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(11.5, 3.8 * n_metrics),
        squeeze=False,
        facecolor="white",
    )
    fig.suptitle(
        f"S{silk_rx},{silk_tx}  {kind_name}    "
        f"Rx={silk_rx}[{hemi_tag(silk_rx)}]  Tx={silk_tx}[{hemi_tag(silk_tx)}]"
        f"{note_suffix}\n{file_title}",
        fontsize=12,
        fontweight="bold",
    )
    series = s_element(s, silk_rx, silk_tx, ant_from_ts)
    for row, metric in enumerate(metrics):
        y, metric_name, ylabel = metric_array(series, metric)
        ax = axes[row, 0]
        _plot_line(
            ax,
            freq,
            y,
            color="#c0392b",
            lw=1.4,
            label=f"S{silk_rx},{silk_tx}",
        )
        _style_axis(ax, freq, f"{kind_name}{metric_name}", ylabel)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def figure_path(base: Path, task: PlotTask) -> Path:
    if task.kind == "reflection_overlay":
        return base / "reflection_all.png"
    rx, tx = task.pairs[0]
    if task.kind == "reflection":
        return base / "reflection" / f"S{rx:02d}_{tx:02d}_{hemi_tag(rx)}.png"
    return (
        base
        / "transmission"
        / f"Tx_Port_{tx:02d}_{hemi_tag(tx)}"
        / f"S{rx:02d}_{tx:02d}_{hemi_tag(rx)}{hemi_tag(tx)}.png"
    )


def safe_part(text: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", text).strip()
    return cleaned or "_"


def safe_rel_parts(rel: Path) -> Path:
    parts = [safe_part(part) for part in rel.parts if part not in ("", ".")]
    return Path(*parts) if parts else Path(".")


def collect_files(root: Path, suffixes: tuple[str, ...]) -> tuple[Path, list[Path]]:
    root = root.expanduser().resolve()
    allowed = {suffix.lower() for suffix in suffixes}
    if root.is_file():
        if root.suffix.lower() not in allowed:
            raise ConfigError(f"root 指向的文件后缀不在 suffixes 里：{root}")
        return root.parent, [root]
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise ConfigError(f"root 既不是目录也不是文件：{root}")
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    files = sorted(set(files), key=lambda path: str(path.relative_to(root)).lower())
    return root, files


def output_base(out_dir: Path, scan_root: Path, sfile: Path) -> Path:
    rel_parent = sfile.parent.relative_to(scan_root)
    return Path(out_dir) / safe_rel_parts(rel_parent) / safe_part(sfile.stem)


def file_title(config_root: Path, sfile: Path) -> str:
    if config_root.is_file() or config_root.resolve() == sfile.resolve():
        return f"{config_root.parent.name}/{config_root.name}"
    root = config_root.resolve()
    return str(Path(config_root.name) / sfile.relative_to(root))


def label_prefix_for(cfg: Config) -> str:
    if cfg.label_prefix:
        return cfg.label_prefix
    return "Ant" if cfg.apply_antenna_map else "Port"


def run(cfg: Config) -> RunStats:
    font = setup_chinese_font()
    if font is None:
        print("[!] 没找到中文字体，标题里的中文可能显示为方框", flush=True)
    else:
        print(f"[*] 字体 {font}", flush=True)

    scan_root, files = collect_files(cfg.root, cfg.suffixes)
    print(f"ROOT = {cfg.root}", flush=True)
    print(f"OUT  = {cfg.out_dir}", flush=True)
    print(f"plot = {cfg.plot}", flush=True)
    if cfg.plot in ("reflection", "all"):
        shown = cfg.reflection_ports if cfg.reflection_ports == "all" else list(cfg.reflection_ports)
        print(f"反射 = {shown}", flush=True)
    if cfg.plot in ("transmission", "all"):
        if cfg.transmission.kind == "all":
            print("透射 = all（每个 S 参数一张图）", flush=True)
        else:
            print(
                f"透射 = 指定端口对 {len(cfg.transmission.pairs)} 组"
                f"  tx={cfg.transmission.tx.mode}  rx={cfg.transmission.rx.mode}",
                flush=True,
            )
    print("规则：反射 all 叠成一张；指定反射、以及全部透射，都是每个 S 参数一张图", flush=True)
    print(f"找到数据文件 = {len(files)}", flush=True)
    if not files:
        print("未找到可画的 Touchstone 文件", flush=True)
        return RunStats(files=0, ok=0, skip=0, fail=0)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    prefix = label_prefix_for(cfg)

    for sfile in files:
        try:
            freq, s, ant_from_ts = load_touchstone(
                sfile,
                cfg.freq_min_ghz,
                cfg.freq_max_ghz,
                cfg.apply_antenna_map,
                cfg.map_preset,
            )
            n_ports = int(s.shape[1])
            if n_ports != 16:
                print(f"  [warn] 端口数={n_ports}，不是 16：{sfile.name}", flush=True)
            tasks = build_tasks(
                cfg.plot,
                cfg.reflection_ports,
                cfg.transmission,
                n_ports,
                ant_from_ts,
            )
            if not tasks:
                print(f"  [warn] 没有要画的曲线：{sfile.name}", flush=True)
                continue
            base = output_base(cfg.out_dir, scan_root, sfile)
            title = file_title(cfg.root, sfile)
            print(f"[*] {sfile.name}  端口={n_ports}  计划 {len(tasks)} 张", flush=True)
            for task in tasks:
                out_path = figure_path(base, task)
                if cfg.skip_existing and out_path.exists():
                    print(f"  [skip] {out_path.relative_to(cfg.out_dir)}", flush=True)
                    skip += 1
                    continue
                fig = draw_task(
                    freq,
                    s,
                    task,
                    ant_from_ts,
                    cfg.metrics,
                    title,
                    prefix,
                    cfg.freq_min_ghz,
                    cfg.freq_max_ghz,
                )
                save_figure(fig, out_path, cfg.dpi)
                print(f"  [ok] {out_path.relative_to(cfg.out_dir)}", flush=True)
                ok += 1
        except Exception as exc:
            fail += 1
            print(f"  [fail] {sfile}: {exc}", flush=True)
            if not isinstance(exc, (ConfigError, ValueError, OSError, FileNotFoundError)):
                traceback.print_exc()

    print(f"\n完成: ok={ok}, skip={skip}, fail={fail}", flush=True)
    print(f"结果目录: {cfg.out_dir}", flush=True)
    return RunStats(files=len(files), ok=ok, skip=skip, fail=fail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按配置绘制 .s16p 反射/透射的幅度和相位")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="配置文件，默认是脚本旁边的 config.yaml",
    )
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        stats = run(cfg)
    except ConfigError as exc:
        print(f"[fail] {exc}", flush=True)
        return 1
    except FileNotFoundError as exc:
        print(f"[fail] 找不到路径：{exc}", flush=True)
        return 1
    if stats.files == 0 or stats.fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
