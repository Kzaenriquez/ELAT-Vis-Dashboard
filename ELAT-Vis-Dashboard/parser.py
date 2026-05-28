#!/usr/bin/env python3
"""
parser.py

ELAT-Vis parser for Ezaki Energy Landscape Analysis Toolkit (ELAT) outputs.

Purpose
-------
Read one ELAT output directory and normalize the output files into clean,
dashboard-ready CSV/JSON tables. This script does NOT plot, rerun ELAT, rerun
fMRI preprocessing, or rerun statistics. It only parses, validates, and exports.

Expected ELAT directory contents
--------------------------------
Required / core:
  - BasinList*.txt or BasinList.txt
  - result*.txt, results*.txt, or Result*.txt
  - Dynamics*.csv or Dynamics.csv
  - *_SN.csv  activity/state-number time series per input file
  - *_BN.csv  basin-number time series per input file

Optional:
  - Figure_100.* to Figure_103.* or figure_100.* to figure_103.*
  - alignment_hrfcols/timelines_merged.csv from make_timeline_flex.py
  - ELA_features*/ outputs from extract_ela_features_v5.py
  - confirmatory*/ outputs from confirmatory_ela_analysis_v3_dynamic_features.py

Outputs
-------
By default, writes to <ELAT_DIR>/parsed/:
  - file_manifest.json
  - model_metadata.json
  - parse_warnings.json
  - available_filters.json
  - basins_table.csv
  - state_basin_membership.csv
  - local_minima_table.csv
  - h_parameters.csv
  - J_parameters_long.csv
  - J_parameters_matrix.csv
  - basin_graph.csv
  - states_table.csv
  - series_table.csv
  - dynamics_wide_clean.csv
  - dynamics_frequency_long.csv
  - dynamics_direct_transition_long.csv
  - dynamics_total_transition_long.csv
  - dynamics_indirect_transition_long.csv
  - timeline_table.csv                  if detected
  - feature_file_manifest.csv           if detected
  - confirmatory_file_manifest.csv      if detected

State convention
----------------
Default state label mapping follows the ELAT-style little-endian mapping used in
our notebooks and v5 feature extractor:
  state 1 = all off / all -1
  state 2 = ROI 1 on, others off
  state 2^N = all on / all +1

For dashboard node labels, this script stores both:
  - binary_01: 0/1 activity pattern
  - sigma_pattern: -1/+1 Ising pattern encoded as 0/1 where 1 means +1/on

Usage
-----
  python parser.py /path/to/ELAT_output

  python parser.py /path/to/ELAT_output \
      --out_dir /path/to/ELAT_output/parsed \
      --roi_names "Amygdala,Hippocampus,Insula,dACC,vmPFC"

  python parser.py /path/to/ELAT_output --roi_names_file roiname.dat
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Basic path / filename helpers
# =============================================================================


def latest_file(folder: Path, patterns: Sequence[str], recursive: bool = False) -> Optional[Path]:
    """Return the newest file matching any glob pattern, or None."""
    folder = Path(folder)
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(folder.rglob(pat) if recursive else folder.glob(pat))
    hits = sorted({p.resolve() for p in hits if p.is_file()}, key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def all_files(folder: Path, patterns: Sequence[str], recursive: bool = False) -> List[Path]:
    """Return sorted unique files matching any glob pattern."""
    folder = Path(folder)
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(folder.rglob(pat) if recursive else folder.glob(pat))
    return sorted({p.resolve() for p in hits if p.is_file()})


def rel_or_none(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def safe_read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def safe_name(text: str) -> str:
    """Make a stable, machine-friendly column/name token."""
    s = str(text).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# =============================================================================
# BIDS-ish metadata helpers
# =============================================================================


def clean_subject_id(subject: str) -> str:
    """sub-FC001 -> FC001; FC001 -> FC001."""
    s = str(subject)
    return re.sub(r"^sub-", "", s)


def infer_group_from_subject(subject: str) -> Optional[str]:
    """Best-effort group inference for Hennings-style FC IDs.

    FC001-FC099 -> HC, FC100+ -> PTSD. Returns None if not recognized.
    """
    subj = clean_subject_id(subject)
    m = re.search(r"FC(\d+)", subj, flags=re.I)
    if not m:
        return None
    num = int(m.group(1))
    if num < 100:
        return "HC"
    return "PTSD"


def infer_group_from_path(path: Path) -> Optional[str]:
    text = str(path)
    if re.search(r"(^|[/_\\-])HC($|[/_\\-])", text, flags=re.I):
        return "HC"
    if re.search(r"PTSD", text, flags=re.I):
        return "PTSD"
    return None


def parse_subject_from_text(text: str) -> Optional[str]:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(text))
    if m:
        return m.group(1)
    m = re.search(r"\b(FC\d{3})\b", str(text), flags=re.I)
    if m:
        return m.group(1).upper()
    return None


def parse_session_from_text(text: str) -> Optional[str]:
    m = re.search(r"ses-([^_\\/\.]+)", str(text))
    return str(m.group(1)) if m else None


def parse_task_from_text(text: str) -> Optional[str]:
    m = re.search(r"task-([^_\\/\.]+)", str(text))
    if m:
        return str(m.group(1))
    lowered = str(text).lower()
    for task in ["baseline", "acquisition", "extinction", "renewal", "localizer", "memory"]:
        if task in lowered:
            return task
    return None


def parse_series_key(path: Path) -> Tuple[str, str, str]:
    """Key used to pair SN and BN files: subject, session, task."""
    text = str(path)
    subject = parse_subject_from_text(text) or Path(path).stem
    session = parse_session_from_text(text) or "UNKNOWN"
    task = parse_task_from_text(text) or "UNKNOWN"
    return clean_subject_id(subject), str(session), str(task)


def parse_inputfile_metadata(input_file: str, folder_group: Optional[str] = None) -> Dict[str, str]:
    subject = parse_subject_from_text(input_file) or Path(str(input_file)).stem
    subject = clean_subject_id(subject)
    task = parse_task_from_text(input_file) or "UNKNOWN"
    session = parse_session_from_text(input_file) or "UNKNOWN"
    group = infer_group_from_subject(subject) or folder_group or "UNKNOWN"
    return {"subject": subject, "session": session, "task": task, "group": group}


# =============================================================================
# State coding helpers
# =============================================================================


def state_label_to_bits(state_label: int, n_rois: int, bit_order: str = "little") -> List[int]:
    """Convert ELAT state label 1..2^N into 0/1 bits.

    bit_order="little" means ROI 1 is the least significant bit:
      state 1 = 000...0
      state 2 = 100...0
      state 2^N = 111...1

    bit_order="big" reverses the displayed ROI order after decoding.
    """
    state = int(state_label)
    state_zero = state - 1
    if state_zero < 0:
        raise ValueError(f"State labels must be >= 1. Got {state_label!r}")
    max_state = 2 ** int(n_rois)
    if state > max_state:
        raise ValueError(f"State {state} exceeds 2^N={max_state} for n_rois={n_rois}")
    bits = [(state_zero >> i) & 1 for i in range(n_rois)]
    if bit_order == "big":
        bits = list(reversed(bits))
    elif bit_order != "little":
        raise ValueError("bit_order must be 'little' or 'big'")
    return [int(x) for x in bits]


def bits_to_state_label(bits: Sequence[int], bit_order: str = "little") -> int:
    """Inverse of state_label_to_bits."""
    raw = [int(x) for x in bits]
    if bit_order == "big":
        raw = list(reversed(raw))
    elif bit_order != "little":
        raise ValueError("bit_order must be 'little' or 'big'")
    return 1 + sum(bit << i for i, bit in enumerate(raw))


def state_label_to_binary_string(state_label: int, n_rois: int, bit_order: str = "little") -> str:
    return "".join(str(v) for v in state_label_to_bits(state_label, n_rois, bit_order))


def state_label_to_sigma(state_label: int, n_rois: int, bit_order: str = "little") -> np.ndarray:
    bits = state_label_to_bits(state_label, n_rois, bit_order)
    return np.array([1 if b else -1 for b in bits], dtype=int)


def hamming_distance_state_labels(a: int, b: int, n_rois: int, bit_order: str = "little") -> int:
    va = state_label_to_bits(a, n_rois, bit_order)
    vb = state_label_to_bits(b, n_rois, bit_order)
    return int(sum(x != y for x, y in zip(va, vb)))


def hypercube_neighbors(state_label: int, n_rois: int, bit_order: str = "little") -> List[int]:
    bits = state_label_to_bits(state_label, n_rois, bit_order)
    out = []
    for i in range(n_rois):
        b2 = bits.copy()
        b2[i] = 1 - b2[i]
        out.append(bits_to_state_label(b2, bit_order=bit_order))
    return out


# =============================================================================
# ROI names
# =============================================================================


def read_roi_names_file(path: Path) -> List[str]:
    """Read ROI/variable names from .dat, .txt, or .csv-ish single/multi-column file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROI names file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    # If CSV with multiple columns, flatten by row and take non-empty tokens.
    names: List[str] = []
    for ln in lines:
        try:
            row = next(csv.reader([ln]))
            row = [x.strip() for x in row if str(x).strip()]
            if len(row) == 1:
                names.append(row[0])
            else:
                names.extend(row)
        except Exception:
            names.append(ln)
    return names


def normalize_roi_names(n_rois: int, roi_names: Optional[Sequence[str]] = None) -> List[str]:
    names = [str(x).strip() for x in (roi_names or []) if str(x).strip()]
    if len(names) < n_rois:
        names = names + [f"ROI_{i}" for i in range(len(names) + 1, n_rois + 1)]
    if len(names) > n_rois:
        names = names[:n_rois]
    return names


# =============================================================================
# File manifest
# =============================================================================


@dataclass
class ELATFileManifest:
    ela_dir: str
    basin_list: Optional[str]
    result_file: Optional[str]
    dynamics_file: Optional[str]
    sn_files: List[str]
    bn_files: List[str]
    figures: List[str]
    timeline_files: List[str]
    feature_files: List[str]
    confirmatory_files: List[str]


def discover_elat_files(ela_dir: Path, recursive_optional: bool = True) -> ELATFileManifest:
    """Discover known ELAT and downstream analysis files."""
    ela_dir = Path(ela_dir).resolve()
    if not ela_dir.exists():
        raise FileNotFoundError(f"ELAT directory not found: {ela_dir}")
    if not ela_dir.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {ela_dir}")

    basin = latest_file(ela_dir, ["BasinList*.txt", "BasinList.txt"], recursive=False)
    result = latest_file(ela_dir, ["result*.txt", "results*.txt", "Result*.txt", "Results*.txt"], recursive=False)
    dynamics = latest_file(ela_dir, ["Dynamics*.csv", "Dynamics.csv", "dynamics*.csv"], recursive=False)
    sn_files = all_files(ela_dir, ["*_SN.csv"], recursive=False)
    bn_files = all_files(ela_dir, ["*_BN.csv"], recursive=False)
    figures = all_files(
        ela_dir,
        ["Figure_100.*", "Figure_101.*", "Figure_102.*", "Figure_103.*", "figure_100.*", "figure_101.*", "figure_102.*", "figure_103.*"],
        recursive=False,
    )

    timeline_files = all_files(
        ela_dir,
        ["timelines_merged*.csv", "timeline_merged*.csv", "*timeline*.csv"],
        recursive=recursive_optional,
    )

    feature_files = all_files(
        ela_dir,
        [
            "ELA_features*/*.csv",
            "ELA_features*/*.json",
            "*/ELA_features*/*.csv",
            "*/ELA_features*/*.json",
        ],
        recursive=False,
    )

    confirmatory_files = all_files(
        ela_dir,
        [
            "confirmatory*/*.csv",
            "confirmatory*/*.md",
            "confirmatory*/*.json",
            "*/confirmatory*/*.csv",
            "*/confirmatory*/*.md",
            "*/confirmatory*/*.json",
            "group_tests_confirmatory.csv",
            "group_tests_by_task_confirmatory.csv",
            "clinical_spearman_confirmatory.csv",
            "primary_results_for_manuscript.csv",
            "selected_confirmatory_features.csv",
            "confirmatory_analysis_summary.md",
        ],
        recursive=False,
    )

    return ELATFileManifest(
        ela_dir=str(ela_dir),
        basin_list=rel_or_none(basin, ela_dir),
        result_file=rel_or_none(result, ela_dir),
        dynamics_file=rel_or_none(dynamics, ela_dir),
        sn_files=[rel_or_none(p, ela_dir) or str(p) for p in sn_files],
        bn_files=[rel_or_none(p, ela_dir) or str(p) for p in bn_files],
        figures=[rel_or_none(p, ela_dir) or str(p) for p in figures],
        timeline_files=[rel_or_none(p, ela_dir) or str(p) for p in timeline_files],
        feature_files=[rel_or_none(p, ela_dir) or str(p) for p in feature_files],
        confirmatory_files=[rel_or_none(p, ela_dir) or str(p) for p in confirmatory_files],
    )


def manifest_path(manifest: ELATFileManifest, key: str) -> Optional[Path]:
    val = getattr(manifest, key)
    if not val:
        return None
    p = Path(val)
    return p if p.is_absolute() else Path(manifest.ela_dir) / p


def manifest_paths(manifest: ELATFileManifest, key: str) -> List[Path]:
    vals = getattr(manifest, key)
    out = []
    for val in vals:
        p = Path(val)
        out.append(p if p.is_absolute() else Path(manifest.ela_dir) / p)
    return out


# =============================================================================
# BasinList parser
# =============================================================================


def parse_basin_list(path: Path, n_rois: Optional[int] = None, bit_order: str = "little") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse ELAT BasinList*.txt.

    Returns
    -------
    basins_table:
      one row per basin.
    state_basin_membership:
      one row per state membership in a basin.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BasinList file not found: {path}")

    basin_rows: List[Dict[str, Any]] = []
    member_rows: List[Dict[str, Any]] = []
    max_state_seen = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            if line.lower().startswith("local minimum"):
                continue
            left, right = line.split(":", 1)
            left_match = re.search(r"-?\d+", left)
            if not left_match:
                continue
            local_minimum_state = int(left_match.group())
            states = [int(x) for x in re.findall(r"-?\d+", right)]
            if not states:
                continue

            basin = len(basin_rows) + 1
            max_state_seen = max(max_state_seen, local_minimum_state, max(states))
            basin_rows.append(
                {
                    "basin": basin,
                    "basin_label": f"B{basin}",
                    "local_minimum_state": local_minimum_state,
                    "basin_size": len(set(states)),
                    "member_states_json": json.dumps(states),
                }
            )

            for state in states:
                row = {
                    "state": int(state),
                    "basin": basin,
                    "basin_label": f"B{basin}",
                    "local_minimum_state": local_minimum_state,
                    "is_local_minimum": int(state) == int(local_minimum_state),
                }
                member_rows.append(row)

    if not basin_rows:
        raise ValueError(f"No basin rows parsed from {path}")

    inferred_n = int(math.ceil(math.log2(max_state_seen))) if max_state_seen > 0 else None
    n = n_rois or inferred_n
    if n is not None:
        for row in member_rows:
            row["binary_01"] = state_label_to_binary_string(row["state"], n, bit_order)
            row["hamming_to_min"] = hamming_distance_state_labels(row["state"], row["local_minimum_state"], n, bit_order)

    basins_table = pd.DataFrame(basin_rows).sort_values("basin").reset_index(drop=True)
    membership = pd.DataFrame(member_rows).sort_values(["basin", "state"]).reset_index(drop=True)
    return basins_table, membership


# =============================================================================
# result*.txt parser
# =============================================================================


def _parse_numeric_block_after_label(text: str, label: str) -> List[List[float]]:
    """Parse MATLAB-ish numeric block after a label such as h = or J =."""
    marker = re.search(rf"(?:^|\n)\s*{re.escape(label)}\s*=\s*\n", text)
    if not marker:
        return []
    rest = text[marker.end():]
    rows: List[List[float]] = []
    started = False
    for raw in rest.splitlines():
        stripped = raw.strip()
        if not stripped:
            if started:
                break
            continue
        # Stop when a new named section begins.
        if started and re.match(r"^[A-Za-z][A-Za-z0-9_ ]*\s*(=|:)?\s*$", stripped):
            break
        nums = re.findall(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", stripped)
        if nums:
            started = True
            rows.append([float(x) for x in nums])
        elif started:
            break
    return rows


def parse_result_file(path: Path, roi_names: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse ELAT result*.txt.

    Returns
    -------
    metadata: dict
    h_parameters: DataFrame
    J_parameters_long: DataFrame
    J_parameters_matrix: DataFrame
    local_minima_table: DataFrame
    basin_graph: DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"result file not found: {path}")
    text = safe_read_text(path)

    input_files = re.findall(r"File Name\s*=\s*(.+)", text)

    node_match = re.search(r"nodeNumber\s*=\s*(\d+)", text)
    n_rois = int(node_match.group(1)) if node_match else None

    h_block = _parse_numeric_block_after_label(text, "h")
    J_block = _parse_numeric_block_after_label(text, "J")

    if n_rois is None:
        if h_block:
            n_rois = len(h_block)
        elif J_block:
            n_rois = len(J_block)

    n = int(n_rois) if n_rois else max(len(h_block), len(J_block), 0)
    names = normalize_roi_names(n, roi_names)

    r_match = re.search(r"(?:^|\n)\s*r\s*=\s*\n\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)", text)
    fit_r = float(r_match.group(1)) if r_match else None

    # h vector
    h_values = [row[0] if row else np.nan for row in h_block]
    h_values = h_values[:n] + [np.nan] * max(0, n - len(h_values))
    h_parameters = pd.DataFrame(
        {
            "roi_index": list(range(1, n + 1)),
            "roi_name": names,
            "h": h_values,
        }
    )

    # J matrix and long table
    J_matrix = pd.DataFrame()
    J_long = pd.DataFrame(columns=["roi_i", "roi_j", "roi_i_name", "roi_j_name", "J"])
    if J_block:
        arr = np.full((n, n), np.nan, dtype=float)
        for i, row in enumerate(J_block[:n]):
            for j, val in enumerate(row[:n]):
                arr[i, j] = float(val)
        J_matrix = pd.DataFrame(arr, columns=names, index=names)
        J_matrix.insert(0, "roi_name", names)
        J_long = (
            pd.DataFrame(arr, index=range(1, n + 1), columns=range(1, n + 1))
            .reset_index(names="roi_i")
            .melt(id_vars="roi_i", var_name="roi_j", value_name="J")
        )
        J_long["roi_i"] = J_long["roi_i"].astype(int)
        J_long["roi_j"] = J_long["roi_j"].astype(int)
        J_long["roi_i_name"] = J_long["roi_i"].map(lambda i: names[i - 1] if 1 <= i <= len(names) else f"ROI_{i}")
        J_long["roi_j_name"] = J_long["roi_j"].map(lambda i: names[i - 1] if 1 <= i <= len(names) else f"ROI_{i}")
        J_long = J_long[["roi_i", "roi_j", "roi_i_name", "roi_j_name", "J"]]

    # LocalMinimum lines
    local_rows: List[Dict[str, Any]] = []
    for m in re.finditer(r"LocalMinimum\s+(\d+)\s*:\s*state\s+(-?\d+)", text, flags=re.I):
        lm_number = int(m.group(1))
        state = int(m.group(2))
        local_rows.append(
            {
                "local_minimum_number": lm_number,
                "local_minimum_state": state,
                "local_minimum_label": f"LM{lm_number}",
            }
        )
    local_minima_table = pd.DataFrame(local_rows)

    # BasinGraph block: columns = state, steepest neighbor/end node, local minimum state
    basin_graph = pd.DataFrame(columns=["state", "steepest_neighbor", "local_minimum_state"])
    matches = list(re.finditer(r"BasinGraph\s*=\s*\n", text))
    if matches:
        # The file has a descriptive line and then a second numeric BasinGraph block.
        marker = matches[-1]
        rest = text[marker.end():]
        rows: List[List[int]] = []
        started = False
        for raw in rest.splitlines():
            stripped = raw.strip()
            if not stripped:
                if started:
                    break
                continue
            nums = re.findall(r"-?\d+", stripped)
            if len(nums) >= 3:
                rows.append([int(nums[0]), int(nums[1]), int(nums[2])])
                started = True
            elif started:
                break
        if rows:
            basin_graph = pd.DataFrame(rows, columns=["state", "steepest_neighbor", "local_minimum_state"])

    metadata = {
        "result_file": str(path),
        "nodeNumber": n,
        "r": fit_r,
        "n_input_files": len(input_files),
        "input_files": input_files,
        "n_h": int(h_parameters["h"].notna().sum()) if not h_parameters.empty else 0,
        "n_J_rows": int(len(J_block)),
        "n_local_minima": int(local_minima_table["local_minimum_state"].nunique()) if not local_minima_table.empty else 0,
        "has_basin_graph": not basin_graph.empty,
    }
    return metadata, h_parameters, J_long, J_matrix, local_minima_table, basin_graph


# =============================================================================
# Energy and state table construction
# =============================================================================


def h_j_to_arrays(n_rois: int, h_parameters: pd.DataFrame, J_long: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return dense h and symmetrized J arrays.

    If both J_ij and J_ji are present, use their average. Diagonal forced to zero.
    """
    h = np.zeros(n_rois, dtype=float)
    if not h_parameters.empty and "h" in h_parameters:
        h_vals = pd.to_numeric(h_parameters["h"], errors="coerce").to_numpy(dtype=float)
        h[: min(n_rois, len(h_vals))] = np.nan_to_num(h_vals[: min(n_rois, len(h_vals))])

    J_raw = np.zeros((n_rois, n_rois), dtype=float)
    seen = np.zeros((n_rois, n_rois), dtype=bool)
    if not J_long.empty:
        for _, row in J_long.iterrows():
            i = int(row["roi_i"]) - 1
            j = int(row["roi_j"]) - 1
            if 0 <= i < n_rois and 0 <= j < n_rois:
                val = pd.to_numeric(pd.Series([row["J"]]), errors="coerce").iloc[0]
                if not pd.isna(val):
                    J_raw[i, j] = float(val)
                    seen[i, j] = True

    J = np.zeros((n_rois, n_rois), dtype=float)
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            vals = []
            if seen[i, j]:
                vals.append(J_raw[i, j])
            if seen[j, i]:
                vals.append(J_raw[j, i])
            if vals:
                v = float(np.mean(vals))
                J[i, j] = v
                J[j, i] = v
    np.fill_diagonal(J, 0.0)
    return h, J


def compute_state_energy(state: int, n_rois: int, h: np.ndarray, J: np.ndarray, bit_order: str = "little") -> float:
    """Pairwise maximum entropy / Ising-style energy for one state.

    E(sigma) = -sum_i h_i sigma_i - sum_{i<j} J_ij sigma_i sigma_j
    """
    sigma = state_label_to_sigma(state, n_rois, bit_order)
    field = float(np.dot(h, sigma))
    pair = 0.0
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            pair += float(J[i, j]) * int(sigma[i]) * int(sigma[j])
    return -field - pair


def build_states_table(
    n_rois: int,
    roi_names: Sequence[str],
    membership: Optional[pd.DataFrame] = None,
    basin_graph: Optional[pd.DataFrame] = None,
    h_parameters: Optional[pd.DataFrame] = None,
    J_long: Optional[pd.DataFrame] = None,
    bit_order: str = "little",
) -> pd.DataFrame:
    """Create one row per possible state with binary pattern, basin, energy, neighbors."""
    roi_names = normalize_roi_names(n_rois, roi_names)

    state_to_basin: Dict[int, int] = {}
    state_to_lm: Dict[int, int] = {}
    state_to_hamming: Dict[int, int] = {}

    if membership is not None and not membership.empty:
        for _, row in membership.iterrows():
            s = int(row["state"])
            state_to_basin[s] = int(row["basin"])
            state_to_lm[s] = int(row["local_minimum_state"])
            if "hamming_to_min" in membership.columns and not pd.isna(row.get("hamming_to_min")):
                state_to_hamming[s] = int(row["hamming_to_min"])

    # Fill local-minimum state from basin graph if BasinList membership was absent/incomplete.
    if basin_graph is not None and not basin_graph.empty:
        # Map local_minimum_state to a sequential basin number when missing.
        lm_to_basin = {lm: i + 1 for i, lm in enumerate(sorted(basin_graph["local_minimum_state"].dropna().astype(int).unique()))}
        for _, row in basin_graph.iterrows():
            s = int(row["state"])
            lm = int(row["local_minimum_state"])
            state_to_lm.setdefault(s, lm)
            state_to_basin.setdefault(s, lm_to_basin.get(lm, np.nan))

    h, J = None, None
    has_energy = h_parameters is not None and J_long is not None and not h_parameters.empty and not J_long.empty
    if has_energy:
        h, J = h_j_to_arrays(n_rois, h_parameters, J_long)

    rows: List[Dict[str, Any]] = []
    for state in range(1, 2 ** n_rois + 1):
        bits = state_label_to_bits(state, n_rois, bit_order)
        sigma = [1 if b else -1 for b in bits]
        lm = state_to_lm.get(state)
        basin = state_to_basin.get(state)
        row: Dict[str, Any] = {
            "state": state,
            "binary_01": "".join(str(b) for b in bits),
            "sigma_pattern": "".join("+" if s == 1 else "-" for s in sigma),
            "basin": basin,
            "basin_label": f"B{basin}" if basin is not None and not pd.isna(basin) else pd.NA,
            "local_minimum_state": lm,
            "is_local_minimum": bool(lm is not None and state == int(lm)),
            "hamming_to_min": state_to_hamming.get(state, hamming_distance_state_labels(state, lm, n_rois, bit_order) if lm is not None else np.nan),
            "neighbors_json": json.dumps(hypercube_neighbors(state, n_rois, bit_order)),
        }
        if has_energy and h is not None and J is not None:
            row["energy"] = compute_state_energy(state, n_rois, h, J, bit_order)
        else:
            row["energy"] = np.nan
        for i, (roi, b, s) in enumerate(zip(roi_names, bits, sigma), start=1):
            row[f"roi{i}_name"] = roi
            row[f"roi{i}_bit"] = b
            row[f"roi{i}_sigma"] = s
        rows.append(row)

    states = pd.DataFrame(rows)
    sort_cols = ["basin", "hamming_to_min", "energy", "state"]
    sort_cols = [c for c in sort_cols if c in states.columns]
    return states.sort_values(sort_cols, na_position="last").reset_index(drop=True)


# =============================================================================
# SN/BN parser
# =============================================================================


def read_numeric_vector_csv(path: Path) -> np.ndarray:
    """Read one-column numeric CSV, tolerating an accidental text/header row."""
    path = Path(path)
    df = pd.read_csv(path, header=None)
    if df.empty:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if vals.isna().any() and len(vals) > 1 and pd.isna(vals.iloc[0]) and vals.iloc[1:].notna().all():
        vals = vals.iloc[1:].reset_index(drop=True)
    if vals.isna().any():
        bad = vals[vals.isna()].index.tolist()[:5]
        raise ValueError(f"Nonnumeric values found in {path}; bad row indices: {bad}")
    arr = vals.to_numpy()
    if len(arr) and np.all(np.isclose(arr, np.round(arr))):
        arr = np.round(arr).astype(int)
    return arr


def pair_sn_bn_files(sn_files: Sequence[Path], bn_files: Sequence[Path]) -> List[Tuple[Path, Path, Tuple[str, str, str]]]:
    sn_map: Dict[Tuple[str, str, str], Path] = {parse_series_key(p): p for p in sn_files}
    pairs: List[Tuple[Path, Path, Tuple[str, str, str]]] = []
    for bn in bn_files:
        key = parse_series_key(bn)
        sn = sn_map.get(key)
        if sn is None:
            # Fallback: match subject + task ignoring session.
            candidates = [(k, p) for k, p in sn_map.items() if k[0] == key[0] and k[2] == key[2]]
            if len(candidates) == 1:
                key, sn = candidates[0]
        if sn is not None:
            pairs.append((sn, bn, key))
    return pairs


def parse_sn_bn_series(sn_files: Sequence[Path], bn_files: Sequence[Path], folder_group: Optional[str] = None) -> pd.DataFrame:
    """Parse paired *_SN.csv and *_BN.csv files into one long time-series table."""
    pairs = pair_sn_bn_files(sn_files, bn_files)
    rows: List[Dict[str, Any]] = []
    for sn_path, bn_path, key in pairs:
        subject, session, task = key
        state = read_numeric_vector_csv(sn_path).astype(int)
        basin = read_numeric_vector_csv(bn_path).astype(int)
        if len(state) != len(basin):
            raise ValueError(f"SN/BN length mismatch for {subject} {session} {task}: SN={len(state)} BN={len(basin)}")
        group = infer_group_from_subject(subject) or folder_group or "UNKNOWN"
        for i, (s, b) in enumerate(zip(state, basin)):
            rows.append(
                {
                    "group": group,
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "row_index_0": i,
                    "kept_row": i + 1,
                    "state": int(s),
                    "basin": int(b),
                    "sn_file": sn_path.name,
                    "bn_file": bn_path.name,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Dynamics.csv parser
# =============================================================================


def clean_dynamics_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def parse_dynamics(path: Path, folder_group: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse ELAT Dynamics*.csv into wide and long tables.

    Returns
    -------
    dynamics_wide_clean
    frequency_long
    direct_transition_long
    total_transition_long
    indirect_transition_long
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dynamics file not found: {path}")

    raw = clean_dynamics_columns(pd.read_csv(path))
    wide_rows: List[Dict[str, Any]] = []
    freq_rows: List[Dict[str, Any]] = []
    direct_rows: List[Dict[str, Any]] = []
    total_rows: List[Dict[str, Any]] = []

    freq_pat = re.compile(r"^Frequency of B(\d+)$", flags=re.I)
    direct_pat = re.compile(r"^Direct transitions from B(\d+) to B(\d+)$", flags=re.I)
    total_pat = re.compile(r"^Transitions from B(\d+) to B(\d+)$", flags=re.I)

    for idx, row in raw.iterrows():
        input_file = str(row.get("InputFile", ""))
        md = parse_inputfile_metadata(input_file, folder_group=folder_group)
        base = {
            "group": md["group"],
            "subject": md["subject"],
            "session": md["session"],
            "task": md["task"],
            "input_file": input_file,
        }
        wide_row: Dict[str, Any] = dict(base)

        for col in raw.columns:
            if col == "InputFile":
                continue
            c = str(col).strip()
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            wide_row[f"dyn_{safe_name(c)}"] = val

            fm = freq_pat.match(c)
            dm = direct_pat.match(c)
            tm = total_pat.match(c)
            if fm:
                b = int(fm.group(1))
                freq_rows.append({**base, "basin": b, "basin_label": f"B{b}", "value": val})
            elif dm:
                src, dst = int(dm.group(1)), int(dm.group(2))
                direct_rows.append({**base, "source": src, "target": dst, "source_label": f"B{src}", "target_label": f"B{dst}", "value": val})
            elif tm:
                src, dst = int(tm.group(1)), int(tm.group(2))
                total_rows.append({**base, "source": src, "target": dst, "source_label": f"B{src}", "target_label": f"B{dst}", "value": val})
        wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows)
    freq_df = pd.DataFrame(freq_rows)
    direct_df = pd.DataFrame(direct_rows)
    total_df = pd.DataFrame(total_rows)

    if not total_df.empty:
        if not direct_df.empty:
            direct_key = direct_df[["group", "subject", "session", "task", "source", "target", "value"]].rename(columns={"value": "direct_value"})
            indirect = total_df.merge(direct_key, on=["group", "subject", "session", "task", "source", "target"], how="left")
            indirect["direct_value"] = indirect["direct_value"].fillna(0.0)
            indirect["value"] = indirect["value"] - indirect["direct_value"]
            indirect = indirect.drop(columns=["direct_value"])
        else:
            indirect = total_df.copy()
    else:
        indirect = pd.DataFrame(columns=total_df.columns)

    return wide_df, freq_df, direct_df, total_df, indirect


# =============================================================================
# Timeline / downstream output discovery
# =============================================================================


def load_timeline_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Timeline file not found: {path}")
    df = pd.read_csv(path)
    for col in ["state", "basin", "orig_tr", "kept_row", "time_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["group", "subject", "session", "task"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def available_hrf_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in {"trial_type", "trial_type_onset"} or re.fullmatch(r"trial_type_hrf\d+(p\d+)?", str(c)):
            cols.append(str(c))
    return cols


def build_available_filters(
    timeline: Optional[pd.DataFrame],
    series: Optional[pd.DataFrame],
    states_table: Optional[pd.DataFrame],
    basins_table: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Summarize filter options for Streamlit/Dash UI controls."""
    out: Dict[str, Any] = {}

    def uniques(df: pd.DataFrame, col: str) -> List[str]:
        if df is None or df.empty or col not in df.columns:
            return []
        return sorted([str(x) for x in df[col].dropna().unique()])

    base = timeline if timeline is not None and not timeline.empty else series
    if base is not None and not base.empty:
        out["groups"] = uniques(base, "group")
        out["subjects"] = uniques(base, "subject")
        out["sessions"] = uniques(base, "session")
        out["tasks"] = uniques(base, "task")
        out["hrf_trial_type_columns"] = available_hrf_columns(base)
        epoch_cols = [c for c in base.columns if "epoch" in str(c).lower() or "window" in str(c).lower() or "block" in str(c).lower()]
        out["epoch_or_window_columns"] = epoch_cols
        out["epoch_or_window_values"] = {c: uniques(base, c) for c in epoch_cols[:30]}
    else:
        out["groups"] = []
        out["subjects"] = []
        out["sessions"] = []
        out["tasks"] = []
        out["hrf_trial_type_columns"] = []
        out["epoch_or_window_columns"] = []
        out["epoch_or_window_values"] = {}

    if basins_table is not None and not basins_table.empty:
        out["basins"] = [int(x) for x in basins_table["basin"].dropna().astype(int).unique()]
        out["basin_labels"] = uniques(basins_table, "basin_label")
    else:
        out["basins"] = []
        out["basin_labels"] = []

    if states_table is not None and not states_table.empty:
        out["states_min"] = int(states_table["state"].min())
        out["states_max"] = int(states_table["state"].max())
        out["n_states"] = int(states_table["state"].nunique())
    else:
        out["states_min"] = None
        out["states_max"] = None
        out["n_states"] = 0

    return out


def make_file_manifest_table(files: Sequence[Path], base: Path, category: str) -> pd.DataFrame:
    rows = []
    for p in files:
        rows.append(
            {
                "category": category,
                "file_name": p.name,
                "relative_path": rel_or_none(p, base),
                "size_bytes": p.stat().st_size if p.exists() else np.nan,
                "modified_time_epoch": p.stat().st_mtime if p.exists() else np.nan,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Main parse function
# =============================================================================


def infer_n_rois_from_available(
    result_metadata: Optional[Mapping[str, Any]],
    basin_membership: Optional[pd.DataFrame],
    series: Optional[pd.DataFrame],
) -> Optional[int]:
    if result_metadata and result_metadata.get("nodeNumber"):
        return int(result_metadata["nodeNumber"])
    max_state = 0
    if basin_membership is not None and not basin_membership.empty and "state" in basin_membership:
        max_state = max(max_state, int(pd.to_numeric(basin_membership["state"], errors="coerce").max()))
    if series is not None and not series.empty and "state" in series:
        max_state = max(max_state, int(pd.to_numeric(series["state"], errors="coerce").max()))
    if max_state > 0:
        return int(math.ceil(math.log2(max_state)))
    return None


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_df(path: Path, df: Optional[pd.DataFrame]) -> None:
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def parse_elat_directory(
    ela_dir: Path,
    out_dir: Optional[Path] = None,
    roi_names: Optional[Sequence[str]] = None,
    roi_names_file: Optional[Path] = None,
    bit_order: str = "little",
    recursive_optional: bool = True,
    copy_figures: bool = False,
) -> Dict[str, Any]:
    """Parse one ELAT output directory and write dashboard-ready tables."""
    ela_dir = Path(ela_dir).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else ela_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    folder_group = infer_group_from_path(ela_dir)

    if roi_names_file:
        names_from_file = read_roi_names_file(roi_names_file)
        roi_names = names_from_file

    manifest = discover_elat_files(ela_dir, recursive_optional=recursive_optional)
    write_json(out_dir / "file_manifest.json", asdict(manifest))

    result_path = manifest_path(manifest, "result_file")
    basin_path = manifest_path(manifest, "basin_list")
    dynamics_path = manifest_path(manifest, "dynamics_file")
    sn_paths = manifest_paths(manifest, "sn_files")
    bn_paths = manifest_paths(manifest, "bn_files")
    fig_paths = manifest_paths(manifest, "figures")
    timeline_paths = manifest_paths(manifest, "timeline_files")
    feature_paths = manifest_paths(manifest, "feature_files")
    confirm_paths = manifest_paths(manifest, "confirmatory_files")

    # 1. Parse result file first when available because it gives nodeNumber, h, J, r, BasinGraph.
    result_metadata: Dict[str, Any] = {}
    h_parameters = pd.DataFrame()
    J_long = pd.DataFrame()
    J_matrix = pd.DataFrame()
    local_minima_table = pd.DataFrame()
    basin_graph = pd.DataFrame()
    if result_path and result_path.exists():
        result_metadata, h_parameters, J_long, J_matrix, local_minima_table, basin_graph = parse_result_file(result_path, roi_names=roi_names)
    else:
        warnings.append("No result*.txt file found. h/J/r/BasinGraph tables will be empty.")

    n_rois = int(result_metadata.get("nodeNumber")) if result_metadata.get("nodeNumber") else None
    normalized_roi_names = normalize_roi_names(n_rois or len(roi_names or []), roi_names) if n_rois else list(roi_names or [])

    # Add names to h/J if result was parsed before names were completely known.
    if n_rois:
        normalized_roi_names = normalize_roi_names(n_rois, roi_names)
        if not h_parameters.empty:
            h_parameters["roi_name"] = normalized_roi_names[: len(h_parameters)]
        if not J_long.empty:
            J_long["roi_i_name"] = J_long["roi_i"].map(lambda i: normalized_roi_names[int(i) - 1] if 1 <= int(i) <= len(normalized_roi_names) else f"ROI_{i}")
            J_long["roi_j_name"] = J_long["roi_j"].map(lambda i: normalized_roi_names[int(i) - 1] if 1 <= int(i) <= len(normalized_roi_names) else f"ROI_{i}")

    # 2. Parse BasinList.
    basins_table = pd.DataFrame()
    membership = pd.DataFrame()
    if basin_path and basin_path.exists():
        basins_table, membership = parse_basin_list(basin_path, n_rois=n_rois, bit_order=bit_order)
        if n_rois is None and not membership.empty:
            n_rois = infer_n_rois_from_available(result_metadata, membership, None)
            normalized_roi_names = normalize_roi_names(n_rois or 0, roi_names)
            # Recompute binary/hamming now that n_rois is known.
            if n_rois:
                basins_table, membership = parse_basin_list(basin_path, n_rois=n_rois, bit_order=bit_order)
    else:
        warnings.append("No BasinList*.txt file found. Basin membership will rely on BasinGraph/SN-BN if available.")

    # 3. Parse SN/BN series.
    series_table = pd.DataFrame()
    if sn_paths and bn_paths:
        series_table = parse_sn_bn_series(sn_paths, bn_paths, folder_group=folder_group)
    else:
        warnings.append("No paired *_SN.csv and *_BN.csv files found. series_table will be empty.")

    # Infer n_rois if still unknown.
    if n_rois is None:
        n_rois = infer_n_rois_from_available(result_metadata, membership, series_table)
        if n_rois:
            normalized_roi_names = normalize_roi_names(n_rois, roi_names)
        else:
            warnings.append("Could not infer number of ROIs/nodes. states_table will be empty.")

    # If BasinList membership lacks binary columns due to initial unknown N, recompute.
    if basin_path and basin_path.exists() and n_rois and (membership.empty or "binary_01" not in membership.columns):
        basins_table, membership = parse_basin_list(basin_path, n_rois=n_rois, bit_order=bit_order)

    # 4. Build complete states table.
    states_table = pd.DataFrame()
    if n_rois:
        states_table = build_states_table(
            n_rois=n_rois,
            roi_names=normalize_roi_names(n_rois, normalized_roi_names),
            membership=membership,
            basin_graph=basin_graph,
            h_parameters=h_parameters,
            J_long=J_long,
            bit_order=bit_order,
        )

    # Add local minimum energy/pattern to local_minima_table if possible.
    if not local_minima_table.empty and not states_table.empty:
        lm_lookup = states_table[["state", "energy", "binary_01", "sigma_pattern", "basin", "basin_label"]].rename(
            columns={
                "state": "local_minimum_state",
                "energy": "local_minimum_energy",
                "binary_01": "local_minimum_binary_01",
                "sigma_pattern": "local_minimum_sigma_pattern",
            }
        )
        local_minima_table = local_minima_table.merge(lm_lookup, on="local_minimum_state", how="left")
    elif local_minima_table.empty and not basins_table.empty:
        local_minima_table = basins_table[["basin", "basin_label", "local_minimum_state"]].copy()
        if not states_table.empty:
            lm_lookup = states_table[["state", "energy", "binary_01", "sigma_pattern"]].rename(
                columns={
                    "state": "local_minimum_state",
                    "energy": "local_minimum_energy",
                    "binary_01": "local_minimum_binary_01",
                    "sigma_pattern": "local_minimum_sigma_pattern",
                }
            )
            local_minima_table = local_minima_table.merge(lm_lookup, on="local_minimum_state", how="left")

    # 5. Dynamics.
    dynamics_wide = pd.DataFrame()
    dyn_freq = pd.DataFrame()
    dyn_direct = pd.DataFrame()
    dyn_total = pd.DataFrame()
    dyn_indirect = pd.DataFrame()
    if dynamics_path and dynamics_path.exists():
        dynamics_wide, dyn_freq, dyn_direct, dyn_total, dyn_indirect = parse_dynamics(dynamics_path, folder_group=folder_group)
    else:
        warnings.append("No Dynamics*.csv file found. Dynamics-derived tables will be empty.")

    # 6. Optional timeline.
    timeline_table = pd.DataFrame()
    chosen_timeline: Optional[Path] = None
    if timeline_paths:
        # Prefer exact timelines_merged.csv, newest if multiple.
        exact = [p for p in timeline_paths if p.name == "timelines_merged.csv"]
        candidates = exact if exact else timeline_paths
        chosen_timeline = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        try:
            timeline_table = load_timeline_table(chosen_timeline)
        except Exception as exc:
            warnings.append(f"Failed to load timeline file {chosen_timeline}: {exc}")

    # 7. Optional manifests for feature/confirmatory outputs.
    feature_manifest = make_file_manifest_table(feature_paths, ela_dir, "feature") if feature_paths else pd.DataFrame()
    confirm_manifest = make_file_manifest_table(confirm_paths, ela_dir, "confirmatory") if confirm_paths else pd.DataFrame()

    # 8. Copy figures optionally. Useful for dashboard static assets.
    if copy_figures and fig_paths:
        fig_out = out_dir / "figures"
        fig_out.mkdir(exist_ok=True)
        for p in fig_paths:
            shutil.copy2(p, fig_out / p.name)

    # 9. Metadata and filters.
    model_metadata: Dict[str, Any] = {
        "ela_dir": str(ela_dir),
        "out_dir": str(out_dir),
        "bit_order": bit_order,
        "n_rois": int(n_rois) if n_rois else None,
        "roi_names": normalize_roi_names(n_rois or 0, normalized_roi_names),
        "folder_group_hint": folder_group,
        "basin_list_file": str(basin_path) if basin_path else None,
        "result_file": str(result_path) if result_path else None,
        "dynamics_file": str(dynamics_path) if dynamics_path else None,
        "timeline_file": str(chosen_timeline) if chosen_timeline else None,
        "n_basins": int(basins_table["basin"].nunique()) if not basins_table.empty else None,
        "n_states_possible": int(2 ** n_rois) if n_rois else None,
        "n_series_rows": int(len(series_table)) if not series_table.empty else 0,
        "n_timeline_rows": int(len(timeline_table)) if not timeline_table.empty else 0,
        "result_metadata": result_metadata,
    }
    filters = build_available_filters(timeline_table if not timeline_table.empty else None, series_table if not series_table.empty else None, states_table, basins_table)

    # 10. Export tables.
    write_json(out_dir / "model_metadata.json", model_metadata)
    write_json(out_dir / "parse_warnings.json", warnings)
    write_json(out_dir / "available_filters.json", filters)

    write_df(out_dir / "basins_table.csv", basins_table)
    write_df(out_dir / "state_basin_membership.csv", membership)
    write_df(out_dir / "local_minima_table.csv", local_minima_table)
    write_df(out_dir / "h_parameters.csv", h_parameters)
    write_df(out_dir / "J_parameters_long.csv", J_long)
    write_df(out_dir / "J_parameters_matrix.csv", J_matrix)
    write_df(out_dir / "basin_graph.csv", basin_graph)
    write_df(out_dir / "states_table.csv", states_table)
    write_df(out_dir / "series_table.csv", series_table)
    write_df(out_dir / "dynamics_wide_clean.csv", dynamics_wide)
    write_df(out_dir / "dynamics_frequency_long.csv", dyn_freq)
    write_df(out_dir / "dynamics_direct_transition_long.csv", dyn_direct)
    write_df(out_dir / "dynamics_total_transition_long.csv", dyn_total)
    write_df(out_dir / "dynamics_indirect_transition_long.csv", dyn_indirect)
    write_df(out_dir / "timeline_table.csv", timeline_table)
    if not feature_manifest.empty:
        write_df(out_dir / "feature_file_manifest.csv", feature_manifest)
    if not confirm_manifest.empty:
        write_df(out_dir / "confirmatory_file_manifest.csv", confirm_manifest)

    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "metadata": model_metadata,
        "warnings": warnings,
        "tables": {
            "basins_table": basins_table,
            "state_basin_membership": membership,
            "local_minima_table": local_minima_table,
            "h_parameters": h_parameters,
            "J_parameters_long": J_long,
            "J_parameters_matrix": J_matrix,
            "basin_graph": basin_graph,
            "states_table": states_table,
            "series_table": series_table,
            "dynamics_wide_clean": dynamics_wide,
            "dynamics_frequency_long": dyn_freq,
            "dynamics_direct_transition_long": dyn_direct,
            "dynamics_total_transition_long": dyn_total,
            "dynamics_indirect_transition_long": dyn_indirect,
            "timeline_table": timeline_table,
        },
    }


# =============================================================================
# CLI
# =============================================================================


def parse_cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse Ezaki/ELAT output folder into dashboard-ready CSV/JSON tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("ela_dir", type=Path, help="ELAT output directory containing BasinList, Dynamics, result, SN/BN files.")
    p.add_argument("--out_dir", type=Path, default=None, help="Output directory. Defaults to <ela_dir>/parsed.")
    p.add_argument("--roi_names", type=str, default=None, help="Comma-separated ROI names in the original ELAT .dat row order.")
    p.add_argument("--roi_names_file", type=Path, default=None, help="Optional text/.dat/.csv file containing ROI names.")
    p.add_argument("--bit_order", choices=["little", "big"], default="little", help="State decoding bit order.")
    p.add_argument("--no_recursive_optional", action="store_true", help="Do not recursively search for optional timeline files.")
    p.add_argument("--copy_figures", action="store_true", help="Copy Figure_100-103 assets into parsed/figures.")
    return p.parse_args()


def main() -> None:
    args = parse_cli_args()
    roi_names = None
    if args.roi_names:
        roi_names = [x.strip() for x in args.roi_names.split(",") if x.strip()]

    result = parse_elat_directory(
        ela_dir=args.ela_dir,
        out_dir=args.out_dir,
        roi_names=roi_names,
        roi_names_file=args.roi_names_file,
        bit_order=args.bit_order,
        recursive_optional=not args.no_recursive_optional,
        copy_figures=args.copy_figures,
    )

    meta = result["metadata"]
    warnings = result["warnings"]
    print("[OK] Parsed ELAT directory")
    print(f"Input:  {args.ela_dir}")
    print(f"Output: {result['out_dir']}")
    print(f"N ROIs: {meta.get('n_rois')}")
    print(f"N basins: {meta.get('n_basins')}")
    print(f"N possible states: {meta.get('n_states_possible')}")
    print(f"Series rows: {meta.get('n_series_rows')}")
    print(f"Timeline rows: {meta.get('n_timeline_rows')}")
    if warnings:
        print("[WARNINGS]")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()
