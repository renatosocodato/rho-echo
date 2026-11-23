"""GraphPad Prism XML ingestion helpers for governed FRET source data."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy import stats


def _tag_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _clean_text(text: Optional[str]) -> str:
    return (text or "").strip()


def _series_points(node: ET.Element) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for elem in node.iter():
        if _tag_name(elem.tag) not in {"d", "v"}:
            continue
        text = _clean_text(elem.text)
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        points.append(
            {
                "value": value,
                "excluded": elem.attrib.get("Excluded") == "1",
            }
        )
    return points


def _numeric_values(node: ET.Element) -> List[float]:
    return [point["value"] for point in _series_points(node)]


def _first_text(node: ET.Element, child_name: str) -> str:
    for child in node:
        if _tag_name(child.tag) == child_name:
            return _clean_text(child.text)
    return ""


def _series_name(node: ET.Element, fallback: str) -> str:
    for key in ("Title", "title", "Name", "name"):
        if key in node.attrib and _clean_text(node.attrib[key]):
            return _clean_text(node.attrib[key])
    title = _first_text(node, "Title")
    return title or fallback


def _subcolumns(node: ET.Element) -> List[ET.Element]:
    subs = [child for child in node if _tag_name(child.tag) == "Subcolumn"]
    return subs or [node]


def _infer_observable(path: Path) -> str:
    name = path.name.lower()
    if "rhoa" in name:
        return "RhoA"
    if "src" in name:
        return "Src"
    return "unknown"


def _normalize_condition_label(text: str) -> str:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()
    if not lowered:
        return "unknown"
    if "q63l" in lowered or lowered == "rhoa":
        return "Q63L"
    if lowered in {"mock", "wt", "control"} or re.search(r"\bwt\b", lowered):
        return "WT"
    return "unknown"


def _infer_condition(*texts: str) -> str:
    joined = " ".join(texts).lower()
    if "q63l" in joined or re.search(r"\brhoa\b", joined):
        return "Q63L"
    if "mock" in joined or re.search(r"\bwt\b", joined):
        return "WT"
    return "unknown"


def _table_kind(title: str) -> str:
    lowered = (title or "").lower()
    if "histo" in lowered or "hist" in lowered:
        return "histogram"
    return "timecourse"


def _collect_ycolumn_titles(root: ET.Element) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for table in root.iter():
        tag = _tag_name(table.tag)
        if tag not in {"XYTable", "Table"} and not tag.endswith("Table"):
            continue
        y_columns = [child for child in table if _tag_name(child.tag) == "YColumn"]
        if not y_columns:
            continue
        titles = [_series_name(y_col, "") for y_col in y_columns]
        if not any(title for title in titles):
            continue
        for index, title in enumerate(titles, start=1):
            if title and index not in mapping:
                mapping[index] = title
    return mapping


def parse_prism_file(path: Path) -> Dict[str, Any]:
    """Parse XY-style Prism XML into trace rows that can be summarized downstream."""
    root = ET.parse(path).getroot()
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    tables: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    ycolumn_titles = _collect_ycolumn_titles(root)
    default_condition = "WT" if "WT" in raw_text and "Q63L" not in raw_text else "unknown"

    for table in root.iter():
        tag = _tag_name(table.tag)
        if tag not in {"XYTable", "Table"} and not tag.endswith("Table"):
            continue

        x_columns = [child for child in table if _tag_name(child.tag) == "XColumn"]
        y_columns = [child for child in table if _tag_name(child.tag) == "YColumn"]
        if not x_columns or not y_columns:
            continue

        table_id = table.attrib.get("ID", f"table_{len(tables) + 1}")
        table_title = _series_name(table, table_id)
        table_kind = _table_kind(table_title)
        x_series_candidates: List[List[float]] = []
        for x_col in x_columns:
            for sub in _subcolumns(x_col):
                values = _numeric_values(sub)
                if values:
                    x_series_candidates.append(values)
        default_x = max(x_series_candidates, key=len) if x_series_candidates else []

        table_rows = 0
        for y_index, y_col in enumerate(y_columns, start=1):
            column_title = _series_name(y_col, f"series_{y_index}")
            raw_condition = ycolumn_titles.get(y_index) or column_title or default_condition
            condition = _normalize_condition_label(raw_condition)
            if condition == "unknown":
                condition = _infer_condition(raw_condition, default_condition)
            for sub_index, sub in enumerate(_subcolumns(y_col), start=1):
                y_points = _series_points(sub)
                if not y_points:
                    continue
                if default_x:
                    if len(y_points) > len(default_x):
                        y_points = y_points[: len(default_x)]
                    x_values = default_x[: len(y_points)]
                else:
                    x_values = list(range(len(y_points)))
                series_name = _series_name(sub, column_title)
                observable = _infer_observable(path)
                replicate_id = (
                    f"{condition}_rep{sub_index:02d}"
                    if condition != "unknown"
                    else f"{table_id}_y{y_index}_rep{sub_index:02d}"
                )
                for point_index, (x_val, point) in enumerate(zip(x_values, y_points), start=1):
                    trace_rows.append(
                        {
                            "source_file": path.name,
                            "source_path": str(path),
                            "observable": observable,
                            "condition": condition,
                            "raw_condition_label": raw_condition,
                            "table_id": table_id,
                            "table_title": table_title,
                            "table_kind": table_kind,
                            "series_name": condition if condition != "unknown" else series_name,
                            "ycolumn_index": y_index,
                            "subseries_index": sub_index,
                            "replicate_id": replicate_id,
                            "point_index": point_index,
                            "x": float(x_val),
                            "y": float(point["value"]),
                            "excluded": bool(point["excluded"]),
                        }
                    )
                table_rows += len(y_points)

        if table_rows:
            tables.append(
                {
                    "table_id": table_id,
                    "table_title": table_title,
                    "table_kind": table_kind,
                    "n_y_columns": len(y_columns),
                    "n_rows": table_rows,
                }
            )

    return {
        "source_file": path.name,
        "observable": _infer_observable(path),
        "tables": tables,
        "trace_rows": trace_rows,
    }


def summarize_trace_rows(rows: Iterable[Dict[str, Any]], baseline_points: int = 3) -> List[Dict[str, Any]]:
    """Compute condition-level summaries from Prism time-course rows."""
    replicate_groups: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("table_kind") != "timecourse":
            continue
        if row.get("excluded"):
            continue
        key = (
            row["source_file"],
            row["observable"],
            row["condition"],
            row["replicate_id"],
        )
        replicate_groups[key].append(row)

    per_replicate: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for key, group_rows in sorted(replicate_groups.items()):
        source_file, observable, condition, replicate_id = key
        ordered = sorted(group_rows, key=lambda row: (row["x"], row["point_index"]))
        xs = np.array([float(row["x"]) for row in ordered], dtype=float)
        ys = np.array([float(row["y"]) for row in ordered], dtype=float)
        baseline_idx = [i for i, x_val in enumerate(xs) if x_val <= 0.0]
        if not baseline_idx:
            baseline_idx = list(range(min(max(baseline_points, 1), len(xs))))
        endpoint_idx = [i for i, x_val in enumerate(xs) if 39.5 <= x_val <= 40.5]
        if not endpoint_idx:
            endpoint_n = min(max(baseline_points, 1), len(xs))
            endpoint_idx = list(range(len(xs) - endpoint_n, len(xs)))
        slope_mask = xs >= 0.0
        if int(np.sum(slope_mask)) < 2:
            slope_mask = np.ones_like(xs, dtype=bool)

        baseline = ys[baseline_idx]
        endpoint = ys[endpoint_idx]
        baseline_mean = float(np.mean(baseline))
        endpoint_mean = float(np.mean(endpoint))
        delta_abs = endpoint_mean - baseline_mean
        fold_change = float(endpoint_mean / baseline_mean) if not math.isclose(baseline_mean, 0.0) else None
        pct_change = float((delta_abs / baseline_mean) * 100.0) if not math.isclose(baseline_mean, 0.0) else None
        slope = float(np.polyfit(xs[slope_mask], ys[slope_mask], 1)[0]) if int(np.sum(slope_mask)) > 1 else 0.0
        variance = float(np.var(ys, ddof=1)) if len(ys) > 1 else 0.0
        per_replicate[(source_file, observable, condition)].append(
            {
                "replicate_id": replicate_id,
                "baseline_mean": baseline_mean,
                "endpoint_mean": endpoint_mean,
                "delta_abs": delta_abs,
                "fold_change": fold_change,
                "pct_change": pct_change,
                "linear_slope": slope,
                "variance": variance,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    pairwise_metrics: Dict[tuple[str, str], Dict[str, float]] = {}

    for key, replicate_metrics in sorted(per_replicate.items()):
        source_file, observable, condition = key
        baselines = np.array([row["baseline_mean"] for row in replicate_metrics], dtype=float)
        endpoints = np.array([row["endpoint_mean"] for row in replicate_metrics], dtype=float)
        deltas = np.array([row["delta_abs"] for row in replicate_metrics], dtype=float)
        fold_changes = np.array([row["fold_change"] for row in replicate_metrics if row["fold_change"] is not None], dtype=float)
        pct_changes = np.array([row["pct_change"] for row in replicate_metrics if row["pct_change"] is not None], dtype=float)
        slopes = np.array([row["linear_slope"] for row in replicate_metrics], dtype=float)
        variances = np.array([row["variance"] for row in replicate_metrics], dtype=float)
        slope_cv_percent = float(np.std(slopes, ddof=1) / abs(float(np.mean(slopes))) * 100.0) if len(slopes) > 1 and not math.isclose(float(np.mean(slopes)), 0.0) else None
        pct_change_cv_percent = (
            float(np.std(pct_changes, ddof=1) / abs(float(np.mean(pct_changes))) * 100.0)
            if len(pct_changes) > 1 and not math.isclose(float(np.mean(pct_changes)), 0.0)
            else None
        )
        endpoint_cv_percent = (
            float(np.std(endpoints, ddof=1) / abs(float(np.mean(endpoints))) * 100.0)
            if len(endpoints) > 1 and not math.isclose(float(np.mean(endpoints)), 0.0)
            else None
        )
        baseline_cv_percent = (
            float(np.std(baselines, ddof=1) / abs(float(np.mean(baselines))) * 100.0)
            if len(baselines) > 1 and not math.isclose(float(np.mean(baselines)), 0.0)
            else None
        )
        heterogeneity_descriptor = "low_response_heterogeneity"
        if pct_change_cv_percent is not None and pct_change_cv_percent >= 60.0:
            heterogeneity_descriptor = "high_response_heterogeneity"
        elif pct_change_cv_percent is not None and pct_change_cv_percent >= 30.0:
            heterogeneity_descriptor = "moderate_response_heterogeneity"
        summary_rows.append(
            {
                "source_file": source_file,
                "observable": observable,
                "condition": condition,
                "n_replicates": int(len(replicate_metrics)),
                "baseline_selector": "x<=0.0" if baselines.size else f"first_{baseline_points}",
                "endpoint_selector": "39.5<=x<=40.5" if endpoints.size else f"last_{baseline_points}",
                "baseline_mean": float(np.mean(baselines)),
                "baseline_sd": float(np.std(baselines, ddof=1)) if len(baselines) > 1 else 0.0,
                "baseline_cv_percent": baseline_cv_percent,
                "endpoint_mean": float(np.mean(endpoints)),
                "endpoint_sd": float(np.std(endpoints, ddof=1)) if len(endpoints) > 1 else 0.0,
                "endpoint_cv_percent": endpoint_cv_percent,
                "delta_abs_mean": float(np.mean(deltas)),
                "fold_change_mean": float(np.mean(fold_changes)) if len(fold_changes) else None,
                "pct_change_mean": float(np.mean(pct_changes)) if len(pct_changes) else None,
                "pct_change_sd": float(np.std(pct_changes, ddof=1)) if len(pct_changes) > 1 else 0.0,
                "pct_change_cv_percent": pct_change_cv_percent,
                "linear_slope_mean": float(np.mean(slopes)),
                "linear_slope_sd": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
                "linear_slope_cv_percent": slope_cv_percent,
                "variance_mean": float(np.mean(variances)),
                "variance_sd": float(np.std(variances, ddof=1)) if len(variances) > 1 else 0.0,
                "heterogeneity_descriptor": heterogeneity_descriptor,
            }
        )

    by_observable: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    replicate_lookup: Dict[tuple[str, str, str], List[Dict[str, Any]]] = per_replicate
    for row in summary_rows:
        by_observable[row["observable"]][row["condition"]] = row
    for observable, condition_map in by_observable.items():
        if "WT" not in condition_map or "Q63L" not in condition_map:
            continue
        wt_row = condition_map["WT"]
        q63l_row = condition_map["Q63L"]
        wt_reps = replicate_lookup[(wt_row["source_file"], observable, "WT")]
        q63l_reps = replicate_lookup[(q63l_row["source_file"], observable, "Q63L")]
        wt_slopes = np.array([row["linear_slope"] for row in wt_reps], dtype=float)
        q63l_slopes = np.array([row["linear_slope"] for row in q63l_reps], dtype=float)
        rate_p_value = float(stats.ttest_ind(wt_slopes, q63l_slopes, equal_var=False).pvalue) if len(wt_slopes) > 1 and len(q63l_slopes) > 1 else None
        baseline_delta_pct = (
            float((q63l_row["baseline_mean"] - wt_row["baseline_mean"]) / wt_row["baseline_mean"] * 100.0)
            if not math.isclose(float(wt_row["baseline_mean"]), 0.0)
            else None
        )
        pairwise_metrics[(observable, "WT")] = {
            "baseline_vs_wt_pct": 0.0,
            "rate_vs_wt_p_value": rate_p_value,
        }
        pairwise_metrics[(observable, "Q63L")] = {
            "baseline_vs_wt_pct": baseline_delta_pct,
            "rate_vs_wt_p_value": rate_p_value,
        }

    for row in summary_rows:
        pairwise = pairwise_metrics.get((row["observable"], row["condition"]), {})
        row["baseline_vs_wt_pct"] = pairwise.get("baseline_vs_wt_pct")
        row["rate_vs_wt_p_value"] = pairwise.get("rate_vs_wt_p_value")

    return summary_rows
