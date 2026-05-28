#!/usr/bin/env python3
"""
figures_timeline_v4_elatvis.py

ELAT-Vis timeline figure generator, v4.

Creates a compact event-aligned timeline from parser.py outputs:
  1) matched epoch row(s), kept separate from events
  2) event row(s): CS+/CS-/ITI
  3) basin table timeline: one row per basin, active cells colored by selected event label
  4) ROI activation heatmap: one row per ROI; active cells colored by selected event label

Key v4 fixes:
  - robust CS+/CS-/ITI normalization, including Unicode minus variants
  - CS+ = red, CS- = blue, ITI = dark/muted gray
  - basin table colors inherit event colors only for the active basin cell
  - ROI heatmap active cells inherit event colors; inactive cells are near-white
  - cleaner subplot titles and right-side legend to prevent text overlap
  - hover templates suppress the Plotly secondary <extra> box

Typical use:
  python figures_timeline_v4_elatvis.py \
    --parsed-dir merge_acquisition_ela_5ROI/parsed \
    --subject FC001 \
    --event-col trial_type_hrf4 \
    --preset fear4

Output defaults to <parsed-dir>/timeline_v4_elatvis.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Defaults / presets
# -----------------------------

EVENT_COLORS = {
    "CS+": "#D55E5E",        # red / threat cue
    "CS-": "#4C78A8",        # blue / safety cue
    "ITI": "#5F6368",        # dark muted gray / neutral interval
    "none": "#F2F2F2",       # background / missing
}

EPOCH_COLORS = {
    "early": "#F3DFA2",
    "late": "#CFCFCF",
    "early_renewal": "#F3DFA2",
    "later_renewal": "#CFCFCF",
    "non_cs_event": "#EFEFEF",
    "unclassified_task": "#EFEFEF",
    "iti": "#EFEFEF",
    "none": "#F7F7F7",
}

PRESETS = {
    "default": {
        "basin_labels": {},
        "event_colors": EVENT_COLORS,
        "epoch_colors": EPOCH_COLORS,
    },
    "fear4": {
        "basin_labels": {
            1: "Global off",
            2: "Threat-like",
            3: "Safety-like",
            4: "Global on",
        },
        "event_colors": EVENT_COLORS,
        "epoch_colors": EPOCH_COLORS,
    },
}

# Discrete code order used consistently across event, basin, and ROI rows.
EVENT_ORDER = ["none", "ITI", "CS-", "CS+"]
EPOCH_ORDER = ["none", "early", "late", "early_renewal", "later_renewal", "non_cs_event", "unclassified_task", "iti"]


# -----------------------------
# Loading helpers
# -----------------------------

def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_parsed_tables(parsed_dir: str | Path) -> Dict[str, pd.DataFrame]:
    parsed_dir = Path(parsed_dir)
    return {
        "timeline": _read_csv_if_exists(parsed_dir / "timeline_table.csv"),
        "series": _read_csv_if_exists(parsed_dir / "series_table.csv"),
        "states": _read_csv_if_exists(parsed_dir / "states_table.csv"),
        "basins": _read_csv_if_exists(parsed_dir / "basins_table.csv"),
        "basin_graph": _read_csv_if_exists(parsed_dir / "basin_graph.csv"),
    }


def get_base_timeline(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    timeline = tables.get("timeline", pd.DataFrame())
    if timeline is not None and not timeline.empty:
        return timeline.copy()
    series = tables.get("series", pd.DataFrame())
    if series is not None and not series.empty:
        return series.copy()
    raise FileNotFoundError("No non-empty timeline_table.csv or series_table.csv found in parsed directory.")


# -----------------------------
# Column detection / filtering
# -----------------------------

def choose_x_col(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested x column not found: {requested}")
        return requested
    for col in ["orig_tr", "kept_row", "tr", "TR", "row_index_0", "volume", "scan", "index"]:
        if col in df.columns:
            return col
    return "__row_number__"


def available_event_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        cl = c.lower()
        if c.startswith("trial_type") and "epoch" not in cl and "block" not in cl:
            cols.append(c)

    def key(c: str) -> Tuple[int, str]:
        cl = c.lower()
        if "onset" in cl:
            return (0, c)
        if "hrf2" in cl:
            return (1, c)
        if "hrf4" in cl:
            return (2, c)
        if "hrf6" in cl:
            return (3, c)
        return (9, c)

    return sorted(cols, key=key)


def suffix_from_event_col(event_col: str) -> Optional[str]:
    lowered = event_col.lower()
    for suffix in ["onset", "hrf2", "hrf4", "hrf6"]:
        if suffix in lowered:
            return suffix
    parts = event_col.split("_")
    return parts[-1] if parts else None


def find_epoch_col_for_event(df: pd.DataFrame, event_col: str, epoch_type: str = "trialblock") -> Optional[str]:
    suffix = suffix_from_event_col(event_col)
    candidates: List[str] = []
    if suffix:
        if epoch_type in ("trialblock", "auto"):
            candidates.append(f"trialblock_epoch_{suffix}")
        if epoch_type in ("hennings", "auto"):
            candidates.append(f"hennings_epoch_{suffix}")
    candidates.extend([f"{event_col}_epoch", f"{event_col}_block"])
    # Fallbacks only; these may not be HRF-matched.
    if epoch_type in ("trialblock", "auto"):
        candidates.append("trialblock_epoch_primary")
    if epoch_type in ("hennings", "auto"):
        candidates.append("hennings_epoch_primary")
    for c in candidates:
        if c in df.columns:
            return c
    return None


def filter_timeline(
    df: pd.DataFrame,
    group: Optional[str] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    query: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()

    def filt(col: str, value: Optional[str]) -> None:
        nonlocal out
        if value is None or col not in out.columns:
            return
        s = out[col].astype(str)
        v = str(value)
        out = out[(s == v) | (s.str.contains(v, regex=False, na=False))]

    filt("group", group)
    filt("subject", subject)
    filt("session", session)
    filt("task", task)

    if query:
        out = out.query(query)

    if out.empty:
        raise ValueError("No rows remain after filtering. Check --subject/--task/--group/--query.")
    return out.reset_index(drop=True)


# -----------------------------
# Label/color helpers
# -----------------------------

def clean_label(x) -> str:
    if pd.isna(x):
        return "none"
    s = str(x).strip()
    if s in {"", "<NA>", "nan", "None", "NaN", "NA", "N/A"}:
        return "none"
    return s


def normalize_event_label(x) -> str:
    """Map common event spelling variants to exactly CS+, CS-, ITI, or none."""
    s = clean_label(x)
    if s == "none":
        return "none"
    raw = s.strip()
    compact = raw.replace(" ", "").replace("_", "").replace("−", "-").replace("–", "-").replace("—", "-")
    lower = compact.lower()

    if lower in {"cs+", "csplus", "csp", "threat", "threatcue"} or "cs+" in lower:
        return "CS+"
    if lower in {"cs-", "csminus", "csm", "safety", "safetycue"} or "cs-" in lower:
        return "CS-"
    if lower in {"iti", "intertrialinterval", "fix", "fixation", "crosshair"} or "iti" in lower:
        return "ITI"
    return raw


def normalize_epoch_label(x) -> str:
    s = clean_label(x)
    if s == "none":
        return "none"
    lower = s.strip().lower().replace(" ", "_").replace("-", "_")
    if "early" in lower:
        return "early_renewal" if "renew" in lower else "early"
    if "late" in lower or "later" in lower:
        return "later_renewal" if "renew" in lower else "late"
    if "iti" in lower:
        return "iti"
    if "non" in lower and "cs" in lower:
        return "non_cs_event"
    return lower


def make_basin_config(
    basins: pd.DataFrame,
    preset: str = "default",
    basin_labels_json: Optional[str] = None,
) -> Tuple[Dict[int, str], List[int]]:
    preset_cfg = PRESETS.get(preset, PRESETS["default"])
    labels = {int(k): v for k, v in preset_cfg.get("basin_labels", {}).items()}

    if basin_labels_json:
        labels.update({int(k): v for k, v in json.loads(Path(basin_labels_json).read_text()).items()})

    if basins is not None and not basins.empty and "basin" in basins.columns:
        basin_order = sorted(pd.to_numeric(basins["basin"], errors="coerce").dropna().astype(int).unique().tolist())
    else:
        basin_order = sorted(labels.keys()) or [1, 2, 3, 4]

    for b in basin_order:
        labels.setdefault(b, f"B{b}")
    return labels, basin_order


def discrete_colorscale(order: Sequence[str], colors: Dict[str, str]) -> Tuple[List[List[float | str]], Dict[str, int]]:
    """Make an exact colorscale for integer category codes 0..n-1."""
    cats = list(dict.fromkeys(order))
    code = {cat: i for i, cat in enumerate(cats)}
    n = len(cats)
    if n == 1:
        c = colors.get(cats[0], "#F2F2F2")
        return [[0.0, c], [1.0, c]], code
    scale: List[List[float | str]] = []
    for i, cat in enumerate(cats):
        left = i / (n - 1)
        right = i / (n - 1)
        color = colors.get(cat, colors.get("none", "#F2F2F2"))
        # Tiny epsilon trick avoids interpolation bleed between adjacent categories.
        if i == 0:
            scale.append([0.0, color])
        else:
            scale.append([max(0.0, left - 1e-9), color])
        scale.append([min(1.0, right + 1e-9), color])
    return scale, code


def ordered_event_categories(values: Sequence[str]) -> List[str]:
    vals = [normalize_event_label(v) for v in values]
    seen = set(vals)
    return [c for c in EVENT_ORDER if c in seen] + sorted([c for c in seen if c not in EVENT_ORDER])


def ordered_epoch_categories(values: Sequence[str]) -> List[str]:
    vals = [normalize_epoch_label(v) for v in values]
    seen = set(vals)
    return [c for c in EPOCH_ORDER if c in seen] + sorted([c for c in seen if c not in EPOCH_ORDER])


# -----------------------------
# State / ROI activation helpers
# -----------------------------

def enrich_timeline_with_states(df: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "state" in out.columns:
        out["state"] = pd.to_numeric(out["state"], errors="coerce").astype("Int64")
    if "basin" in out.columns:
        out["basin"] = pd.to_numeric(out["basin"], errors="coerce").astype("Int64")

    if states is not None and not states.empty and "state" in states.columns:
        keep_cols = [c for c in states.columns if c == "state" or c.startswith("roi") or c in [
            "binary_01", "sigma_pattern", "basin", "local_minimum_state", "is_local_minimum", "energy"
        ]]
        st = states[keep_cols].copy()
        st["state"] = pd.to_numeric(st["state"], errors="coerce").astype("Int64")
        rename = {c: f"state_{c}" for c in keep_cols if c != "state"}
        st = st.rename(columns=rename)
        out = out.merge(st, on="state", how="left")
        if "basin" not in out.columns and "state_basin" in out.columns:
            out["basin"] = out["state_basin"]
        elif "basin" in out.columns and "state_basin" in out.columns:
            out["basin"] = out["basin"].fillna(out["state_basin"])
    return out


def get_roi_specs(states: pd.DataFrame) -> List[Tuple[int, str, str]]:
    """Return (roi_index, roi_name, bit_col_in_states_table)."""
    if states is None or states.empty:
        return []
    specs = []
    for col in states.columns:
        m = re.fullmatch(r"roi(\d+)_bit", col)
        if not m:
            continue
        idx = int(m.group(1))
        name_col = f"roi{idx}_name"
        if name_col in states.columns and states[name_col].notna().any():
            name = str(states[name_col].dropna().iloc[0])
        else:
            name = f"ROI {idx}"
        specs.append((idx, name, col))
    return sorted(specs, key=lambda x: x[0])


def build_roi_activation_event_matrix(
    df: pd.DataFrame,
    states: pd.DataFrame,
    x_col: str,
    event_col: Optional[str],
    epoch_col: Optional[str],
    event_code: Dict[str, int],
) -> Tuple[np.ndarray, List[str], List[List[str]]]:
    """Inactive=none code. Active cell receives the event code for that TR."""
    specs = get_roi_specs(states)
    if not specs:
        return np.empty((0, len(df))), [], []

    st = states.copy()
    st["state"] = pd.to_numeric(st["state"], errors="coerce").astype("Int64")
    bit_lookup = {}
    for _, r in st.iterrows():
        if pd.isna(r.get("state")):
            continue
        s = int(r["state"])
        bit_lookup[s] = [int(r[col]) if pd.notna(r.get(col)) else np.nan for _, _, col in specs]

    n_rois = len(specs)
    n_tr = len(df)
    z = np.full((n_rois, n_tr), event_code.get("none", 0), dtype=float)
    hover_text: List[List[str]] = [["" for _ in range(n_tr)] for _ in range(n_rois)]

    for j, (_, r) in enumerate(df.iterrows()):
        state = r.get("state", np.nan)
        bits = bit_lookup.get(int(state)) if pd.notna(state) else None
        ev = normalize_event_label(r.get(event_col, "none")) if event_col and event_col in df.columns else "none"
        ep = normalize_epoch_label(r.get(epoch_col, "none")) if epoch_col and epoch_col in df.columns else "none"
        for i, (_, roi_name, _) in enumerate(specs):
            val = np.nan if bits is None else bits[i]
            is_active = val == 1
            z[i, j] = event_code.get(ev, event_code.get("none", 0)) if is_active else event_code.get("none", 0)
            status = "active" if is_active else "inactive" if val == 0 else "unknown"
            hover_text[i][j] = (
                f"TR: {r.get(x_col, '')}<br>"
                f"ROI: {roi_name}<br>"
                f"Status: {status}<br>"
                f"State: {clean_label(r.get('state', 'none'))}<br>"
                f"Basin: {clean_label(r.get('basin', 'none'))}<br>"
                f"Event: {ev}<br>"
                f"Epoch: {ep}"
            )

    roi_labels = [name for _, name, _ in specs]
    return z, roi_labels, hover_text


# -----------------------------
# Plot builders
# -----------------------------

def add_categorical_row(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
    x_col: str,
    value_col: str,
    row_label: str,
    colors: Dict[str, str],
    category_type: str,
) -> Tuple[List[str], Dict[str, int]]:
    if category_type == "event":
        values = [normalize_event_label(v) for v in df[value_col].tolist()]
        cats = ordered_event_categories(values)
    else:
        values = [normalize_epoch_label(v) for v in df[value_col].tolist()]
        cats = ordered_epoch_categories(values)
    if "none" not in cats:
        cats = ["none"] + cats
    scale, code = discrete_colorscale(cats, colors)
    z = [[code.get(v, code.get("none", 0)) for v in values]]
    text = [[f"{row_label}<br>Column: {value_col}<br>TR: {x}<br>Label: {v}" for x, v in zip(df[x_col], values)]]

    fig.add_trace(
        go.Heatmap(
            x=df[x_col],
            y=[row_label],
            z=z,
            text=text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=scale,
            zmin=0,
            zmax=max(len(cats) - 1, 1),
            showscale=False,
            xgap=0,
            ygap=0,
            name=row_label,
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(showticklabels=True, ticks="", row=row, col=1)
    return cats, code


def add_legend_swatches(fig: go.Figure, row: int, colors: Dict[str, str], labels: Sequence[str], group: str, title: str) -> None:
    first = True
    for label in labels:
        if label in {"", "none"}:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=colors.get(label, "#BBBBBB"), symbol="square"),
                name=f"{label}",
                legendgroup=group,
                legendgrouptitle_text=title if first else None,
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        first = False


def make_basin_event_table(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
    x_col: str,
    event_col: Optional[str],
    epoch_col: Optional[str],
    basin_labels: Dict[int, str],
    basin_order: List[int],
    event_colors: Dict[str, str],
) -> None:
    event_values = [normalize_event_label(v) for v in df[event_col].tolist()] if event_col and event_col in df.columns else ["none" for _ in range(len(df))]
    cats = ordered_event_categories(event_values)
    if "none" not in cats:
        cats = ["none"] + cats
    scale, code = discrete_colorscale(cats, event_colors)

    z = np.full((len(basin_order), len(df)), code.get("none", 0), dtype=float)
    text: List[List[str]] = []
    for i, b in enumerate(basin_order):
        hrow = []
        for j, (_, r) in enumerate(df.iterrows()):
            active_b = int(r["basin"]) if pd.notna(r.get("basin")) else None
            ev = event_values[j]
            ep = normalize_epoch_label(r.get(epoch_col, "none")) if epoch_col and epoch_col in df.columns else "none"
            if active_b == b:
                z[i, j] = code.get(ev, code.get("none", 0))
                hrow.append(
                    f"TR: {r.get(x_col, '')}<br>"
                    f"Active basin: B{b} {basin_labels.get(b, '')}<br>"
                    f"State: {clean_label(r.get('state', 'none'))}<br>"
                    f"Event: {ev}<br>"
                    f"Epoch: {ep}"
                )
            else:
                hrow.append(
                    f"TR: {r.get(x_col, '')}<br>"
                    f"Basin row: B{b} {basin_labels.get(b, '')}<br>"
                    f"Active basin: B{active_b if active_b is not None else 'none'}<br>"
                    f"Event: {ev}"
                )
        text.append(hrow)

    y_labels = [f"B{b}: {basin_labels.get(b, f'B{b}')}" for b in basin_order]
    fig.add_trace(
        go.Heatmap(
            x=df[x_col],
            y=y_labels,
            z=z,
            text=text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=scale,
            zmin=0,
            zmax=max(len(cats) - 1, 1),
            showscale=False,
            xgap=0.4,
            ygap=1.0,
            name="Basin table timeline",
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", row=row, col=1)


def make_roi_activation_heatmap(
    fig: go.Figure,
    df: pd.DataFrame,
    states: pd.DataFrame,
    row: int,
    x_col: str,
    event_col: Optional[str],
    epoch_col: Optional[str],
    event_colors: Dict[str, str],
) -> None:
    # Use the same event code/colors as event rows. Inactive cells are coded as none.
    event_values = [normalize_event_label(v) for v in df[event_col].tolist()] if event_col and event_col in df.columns else ["none" for _ in range(len(df))]
    cats = ordered_event_categories(event_values)
    if "none" not in cats:
        cats = ["none"] + cats
    scale, code = discrete_colorscale(cats, event_colors)

    z, roi_labels, text = build_roi_activation_event_matrix(df, states, x_col, event_col, epoch_col, code)
    if z.size == 0:
        fig.add_annotation(
            text="No ROI bit columns found in states_table.csv",
            xref=f"x{row}", yref=f"y{row}", x=0.5, y=0.5, showarrow=False,
            row=row, col=1,
        )
        return

    fig.add_trace(
        go.Heatmap(
            x=df[x_col],
            y=roi_labels,
            z=z,
            text=text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=scale,
            zmin=0,
            zmax=max(len(cats) - 1, 1),
            showscale=False,
            xgap=0.4,
            ygap=0.5,
            name="ROI activation",
            showlegend=False,
        ),
        row=row,
        col=1,
    )


# -----------------------------
# Main figure
# -----------------------------

def make_elat_timeline_dashboard(
    timeline: pd.DataFrame,
    states: pd.DataFrame,
    basins: pd.DataFrame,
    event_col: Optional[str] = None,
    show_all_event_cols: bool = False,
    epoch_type: str = "trialblock",
    x_col: Optional[str] = None,
    preset: str = "default",
    basin_labels_json: Optional[str] = None,
    basin_event_col: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 1500,
    height: Optional[int] = None,
) -> go.Figure:
    df = enrich_timeline_with_states(timeline, states)

    x_col = choose_x_col(df, x_col)
    if x_col == "__row_number__":
        df[x_col] = np.arange(1, len(df) + 1)

    df = df.sort_values(x_col).reset_index(drop=True)

    basin_labels, basin_order = make_basin_config(basins, preset=preset, basin_labels_json=basin_labels_json)

    event_cols_all = available_event_columns(df)
    if show_all_event_cols:
        event_cols = event_cols_all
    elif event_col and event_col in df.columns:
        event_cols = [event_col]
    elif event_cols_all:
        event_cols = [event_cols_all[0]]
    else:
        event_cols = []

    # For each event col, put epoch row first and event row second so event is adjacent to basin table.
    row_specs: List[Tuple[str, str, Optional[str]]] = []
    matched_epochs: Dict[str, Optional[str]] = {}
    for ec in event_cols:
        ep = find_epoch_col_for_event(df, ec, epoch_type=epoch_type)
        matched_epochs[ec] = ep
        if ep:
            row_specs.append(("epoch", ep, ec))
        row_specs.append(("event", ec, None))

    selected_event_for_basin = basin_event_col if basin_event_col in df.columns else (event_cols[-1] if event_cols else None)
    selected_epoch_for_basin = matched_epochs.get(selected_event_for_basin) if selected_event_for_basin else None

    n_top_rows = len(row_specs)
    rows = n_top_rows + 2

    # Keep subplot titles sparse to prevent overlap; row labels carry event/epoch identity.
    subplot_titles: List[str] = ["" for _ in range(n_top_rows)]
    subplot_titles += ["Basin table timeline", "ROI activation heatmap"]
    row_heights: List[float] = [0.055 for _ in range(n_top_rows)] + [0.34, 0.38]

    if height is None:
        height = max(720, 340 + 42 * n_top_rows + 72 * max(len(basin_order), 4))

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    preset_cfg = PRESETS.get(preset, PRESETS["default"])
    event_colors = {**EVENT_COLORS, **preset_cfg.get("event_colors", {})}
    epoch_colors = {**EPOCH_COLORS, **preset_cfg.get("epoch_colors", {})}

    current_row = 1
    for kind, col_name, parent in row_specs:
        if kind == "epoch":
            add_categorical_row(fig, df, current_row, x_col, col_name, f"epoch\n{col_name}", epoch_colors, "epoch")
        else:
            add_categorical_row(fig, df, current_row, x_col, col_name, f"event\n{col_name}", event_colors, "event")
        current_row += 1

    basin_row = current_row
    make_basin_event_table(
        fig,
        df,
        basin_row,
        x_col,
        selected_event_for_basin,
        selected_epoch_for_basin,
        basin_labels,
        basin_order,
        event_colors,
    )
    current_row += 1

    roi_row = current_row
    make_roi_activation_heatmap(fig, df, states, roi_row, x_col, selected_event_for_basin, selected_epoch_for_basin, event_colors)

    # Right-side legends: avoids collision with figure title and top rows.
    add_legend_swatches(fig, roi_row, event_colors, ["CS+", "CS-", "ITI"], "event", "Event / active ROI color")
    add_legend_swatches(fig, roi_row, epoch_colors, ["early", "late"], "epoch", "Epoch")
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=event_colors["none"], symbol="square", line=dict(color="#BDBDBD", width=1)),
            name="inactive / none",
            legendgroup="event",
            hoverinfo="skip",
            showlegend=True,
        ),
        row=roi_row,
        col=1,
    )

    for r in range(1, rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False, row=r, col=1, tickfont=dict(size=11))

    if title is None:
        parts = []
        for c in ["group", "subject", "session", "task"]:
            if c in df.columns:
                vals = pd.Series(df[c].dropna().astype(str).unique()).head(3).tolist()
                if len(vals) == 1:
                    parts.append(vals[0])
        ev_text = f" | event: {selected_event_for_basin}" if selected_event_for_basin else ""
        title = "Event-aligned ELAT timeline" + (": " + " | ".join(parts) if parts else "") + ev_text

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", y=0.985, yanchor="top", font=dict(size=17)),
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=125, r=260, t=115, b=65),
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            tracegroupgap=14,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(title_text="TR / kept row", row=rows, col=1)
    return fig


# -----------------------------
# CLI
# -----------------------------

def print_columns(parsed_dir: Path) -> None:
    tables = load_parsed_tables(parsed_dir)
    df = get_base_timeline(tables)
    print("Available columns:")
    for c in df.columns:
        print(f"  {c}")
    print("\nDetected event columns:")
    for c in available_event_columns(df):
        epoch = find_epoch_col_for_event(df, c, epoch_type="auto")
        print(f"  {c}" + (f"  -> matched epoch: {epoch}" if epoch else ""))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create ELAT-Vis v4 stacked event/basin/ROI timeline figures from parser.py outputs."
    )
    p.add_argument("--parsed-dir", required=True, help="Directory produced by parser.py")
    p.add_argument("--out", default=None, help="Output HTML path. Default: <parsed-dir>/timeline_v4_elatvis.html")

    p.add_argument("--group", default=None)
    p.add_argument("--subject", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--query", default=None, help="Optional pandas query string")

    p.add_argument("--x-col", default=None, help="TR-like x column. Default: orig_tr > kept_row > row_index_0")
    p.add_argument("--event-col", default=None, help="Event column, e.g. trial_type_hrf4")
    p.add_argument("--basin-event-col", default=None, help="Event column used to color basin/ROI active cells. Defaults to selected event column.")
    p.add_argument("--show-all-event-cols", action="store_true", help="Show all detected trial_type_* rows with matched epoch rows")
    p.add_argument("--epoch-type", choices=["trialblock", "hennings", "auto"], default="trialblock")

    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="default",
                   help="Use fear4 for B1 off, B2 threat-like, B3 safety-like, B4 on labels.")
    p.add_argument("--basin-labels-json", default=None, help="Optional JSON mapping basin id to label")

    p.add_argument("--width", type=int, default=1500)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--list-columns", action="store_true", help="Print available timeline/event columns and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    parsed_dir = Path(args.parsed_dir)

    if args.list_columns:
        print_columns(parsed_dir)
        return 0

    tables = load_parsed_tables(parsed_dir)
    df = get_base_timeline(tables)
    df = filter_timeline(
        df,
        group=args.group,
        subject=args.subject,
        session=args.session,
        task=args.task,
        query=args.query,
    )

    fig = make_elat_timeline_dashboard(
        timeline=df,
        states=tables.get("states", pd.DataFrame()),
        basins=tables.get("basins", pd.DataFrame()),
        event_col=args.event_col,
        show_all_event_cols=args.show_all_event_cols,
        epoch_type=args.epoch_type,
        x_col=args.x_col,
        preset=args.preset,
        basin_labels_json=args.basin_labels_json,
        basin_event_col=args.basin_event_col,
        width=args.width,
        height=args.height,
    )

    out = Path(args.out) if args.out else parsed_dir / "timeline_v4_elatvis.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
