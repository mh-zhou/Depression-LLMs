# -*- coding: utf-8 -*-
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import re
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    hamming_loss,
)

warnings.filterwarnings("ignore")

SEED = 42
rng = np.random.default_rng(SEED)

PROJECT_ROOT = Path(r"C:/Users/zmh/Downloads/Depressed/Depressed")
LORA_ROOT = PROJECT_ROOT / "LlamaFactory" / "saves" / "Qwen3.5-9B-Base" / "lora"

OUTPUT_ROOT = PROJECT_ROOT / "result_IMG" / "nature_behavior_interpretability_9B_v3"
CSV_ROOT = OUTPUT_ROOT / "supporting_csv"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CSV_ROOT.mkdir(parents=True, exist_ok=True)

STAGES = ["Base", "Single-SFT", "Multi-SFT", "KTO"]

RUNS_4 = {
    "Base": LORA_ROOT / "Native_eval_26-03-18-08-43-30" / "generated_predictions.jsonl",
    "Single-SFT": LORA_ROOT / "eval_2026-03-08-20-34-30" / "generated_predictions.jsonl",
    "Multi-SFT": LORA_ROOT / "eval_2026-03-11-03-48-30_1" / "generated_predictions.jsonl",
    "KTO": LORA_ROOT / "eval_2026-04-25-14-39-38" / "generated_predictions.jsonl",
}

RUNS_8 = {
    "Base": LORA_ROOT / "Native_eval_26-03-18-11-36-32" / "generated_predictions.jsonl",
    "Single-SFT": LORA_ROOT / "eval_2026-03-10-04-50-16" / "generated_predictions.jsonl",
    "Multi-SFT": LORA_ROOT / "eval_2026-03-11-03-48-30_2" / "generated_predictions.jsonl",
    "KTO": LORA_ROOT / "eval_2026-04-24-10-46-15" / "generated_predictions.jsonl",
}

CODES_8 = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]
CODE_TO_INDEX = {c: i for i, c in enumerate(CODES_8)}
SYMPTOM_LABELS = [f"S{i + 1}" for i in range(8)]

def setup_matplotlib():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def light_blue_cmap():
    return LinearSegmentedColormap.from_list(
        "light_blues",
        ["#f9fcff", "#edf5fb", "#dcecf8", "#c7e0f3", "#a9cdea", "#7fb3dd", "#4f93c6"],
    )


def error_cmap():
    return LinearSegmentedColormap.from_list(
        "soft_error",
        ["#ffffff", "#f7e5e5", "#e6a3a3", "#c74a4a", "#8f1f1f"],
    )

def check_paths(run_dict, task_name):
    print(f"\n[Check paths | {task_name}]")
    ok = True

    for stage in STAGES:
        p = Path(run_dict[stage])
        exists = p.exists()
        print(f"{stage:10s}: {p} -> {exists}")
        if not exists:
            ok = False

    if not ok:
        raise FileNotFoundError(f"{task_name} path check failed.")

    return True


def read_jsonl(path):
    rows = []
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSON parse failed: {path}, line={line_id}, error={e}")

    return rows


def extract_fields_from_jsonl(path):
    rows = read_jsonl(path)

    pred_keys = ["predict", "prediction", "pred", "output", "response"]
    label_keys = ["label", "labels", "target", "answer", "gold"]

    preds = []
    labels = []

    for item in rows:
        pred_text = None
        label_text = None

        for k in pred_keys:
            if k in item:
                pred_text = item[k]
                break

        for k in label_keys:
            if k in item:
                label_text = item[k]
                break

        preds.append(pred_text)
        labels.append(label_text)

    return preds, labels


def clean_generation_text(x):
    if x is None:
        return ""

    s = str(x)
    s = s.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answer_region_after_think(x):
    s = "" if x is None else str(x)

    if "</think>" in s.lower():
        parts = re.split(r"</think>", s, flags=re.IGNORECASE)
        return clean_generation_text(parts[-1]), False

    if "<think>" in s.lower():
        return clean_generation_text(s[-3000:]), True

    s = re.sub(r"<think>[\s\S]*?</think>", " ", s, flags=re.IGNORECASE)
    return clean_generation_text(s), False


def parse_4cls_value(x):
    if x is None:
        return None

    if isinstance(x, (int, np.integer)):
        v = int(x)
        return v if v in [0, 1, 2, 3] else None

    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        v = int(x)
        return v if v in [0, 1, 2, 3] else None

    s, _ = answer_region_after_think(x)
    s_low = s.lower()

    patterns = [
        r"^\s*\(?\s*([0-3])\s*\)?\s*$",
        r"option\s*[:：]?\s*([0-3])",
        r"answer\s*[:：]?\s*([0-3])",
        r"label\s*[:：]?\s*([0-3])",
        r"class\s*[:：]?\s*([0-3])",
        r"depression\s*level\s*[:：]?\s*([0-3])",
        r"(?<!\d)([0-3])(?!\d)",
    ]

    for pat in patterns:
        m = re.search(pat, s_low)
        if m:
            return int(m.group(1))

    if "non-depressed" in s_low or "non depressed" in s_low or "not depressed" in s_low or "no depression" in s_low:
        return 0

    if "mildly depressed" in s_low or "mild depression" in s_low or re.search(r"\bmild\b", s_low):
        return 1

    if "moderately depressed" in s_low or "moderate depression" in s_low or re.search(r"\bmoderate\b", s_low):
        return 2

    if "severely depressed" in s_low or "severe depression" in s_low or re.search(r"\bsevere\b", s_low):
        return 3

    return None


def vector_from_codes(codes):
    v = np.zeros(8, dtype=int)

    for c in codes:
        if c in CODE_TO_INDEX:
            v[CODE_TO_INDEX[c]] = 1

    return v


def parse_numeric_8_vector(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        vals = list(x)
        if len(vals) >= 8:
            try:
                return np.array([1 if float(v) >= 0.5 else 0 for v in vals[:8]], dtype=int)
            except Exception:
                return None

    if isinstance(x, dict):
        for k in ["label", "labels", "target", "answer", "gold", "output", "predict"]:
            if k in x:
                out = parse_numeric_8_vector(x[k])
                if out is not None:
                    return out

    return None


def extract_code_line_from_unclosed_think(s):
    tail = clean_generation_text(s[-3000:])

    marker_patterns = [
        r"(?:final answer|answer|therefore|output|result|答案|输出|最终答案|结论)\s*[:：]?\s*([甲乙丙丁戊己庚辛](?:\s*[,，、;；和及]\s*[甲乙丙丁戊己庚辛])*)",
        r"(?:the emotions are|identified emotions are|selected labels are)\s*[:：]?\s*([甲乙丙丁戊己庚辛](?:\s*[,，、;；和及]\s*[甲乙丙丁戊己庚辛])*)",
    ]

    matches = []
    for pat in marker_patterns:
        matches.extend(re.findall(pat, tail, flags=re.IGNORECASE))

    if matches:
        return matches[-1]

    raw_lines = re.split(r"[\n\r]+", s[-3000:])
    candidates = []

    for line in raw_lines:
        line_clean = clean_generation_text(line)
        if re.fullmatch(r"[甲乙丙丁戊己庚辛](?:\s*[,，、;；和及]\s*[甲乙丙丁戊己庚辛])*", line_clean):
            candidates.append(line_clean)

    return candidates[-1] if candidates else ""


def parse_8label_vector(x):
    if x is None:
        return None

    numeric = parse_numeric_8_vector(x)
    if numeric is not None:
        return numeric

    s_raw = str(x)
    s_answer, unclosed = answer_region_after_think(s_raw)

    if unclosed:
        code_region = extract_code_line_from_unclosed_think(s_raw)
        if code_region == "":
            return None
    else:
        code_region = s_answer

    code_region = (
        code_region
        .replace("，", ",")
        .replace("、", ",")
        .replace("；", ",")
        .replace(";", ",")
        .replace("和", ",")
        .replace("及", ",")
    )
    code_region = clean_generation_text(code_region)

    low = code_region.lower()
    if any(t in low for t in ["none", "no emotion", "no label", "无", "没有", "未检测"]):
        if not any(c in code_region for c in CODES_8):
            return np.zeros(8, dtype=int)

    codes = [c for c in CODES_8 if c in code_region]
    if codes:
        return vector_from_codes(codes)

    nums = re.findall(r"(?<!\d)([01])(?:\.0)?(?!\d)", code_region)
    if len(nums) >= 8:
        return np.array([int(v) for v in nums[:8]], dtype=int)

    return None


def load_4task(run_dict):
    raw = {}

    for stage in STAGES:
        preds_text, labels_text = extract_fields_from_jsonl(run_dict[stage])

        parsed_preds = [parse_4cls_value(x) for x in preds_text]
        parsed_labels = [parse_4cls_value(x) for x in labels_text]

        preds = np.array([np.nan if x is None else x for x in parsed_preds], dtype=float)
        labels = np.array([np.nan if x is None else x for x in parsed_labels], dtype=float)

        print(f"\n[4-task | {stage}]")
        print(f"jsonl: {run_dict[stage]}")
        print(f"pred valid: {np.sum(~np.isnan(preds))}/{len(preds)}")
        print(f"label valid: {np.sum(~np.isnan(labels))}/{len(labels)}")

        raw[stage] = {"pred": preds, "label": labels}

    y_true = raw["Base"]["label"]
    min_len = min([len(y_true)] + [len(raw[s]["pred"]) for s in STAGES])

    y_true = y_true[:min_len]
    valid = ~np.isnan(y_true)
    preds = {}

    for stage in STAGES:
        p = raw[stage]["pred"][:min_len]
        preds[stage] = p
        valid &= ~np.isnan(p)

    y_true = y_true[valid].astype(int)

    for stage in STAGES:
        preds[stage] = preds[stage][valid].astype(int)

    if len(y_true) == 0:
        raise RuntimeError("No valid 4-task samples after alignment.")

    print(f"\n[4-task alignment] valid samples: {len(y_true)}")
    return y_true, preds


def load_8task(run_dict):
    raw = {}

    for stage in STAGES:
        preds_text, labels_text = extract_fields_from_jsonl(run_dict[stage])

        preds = [parse_8label_vector(x) for x in preds_text]
        labels = [parse_8label_vector(x) for x in labels_text]

        print(f"\n[8-task | {stage}]")
        print(f"jsonl: {run_dict[stage]}")
        print(f"pred valid: {sum(x is not None for x in preds)}/{len(preds)}")
        print(f"label valid: {sum(x is not None for x in labels)}/{len(labels)}")

        raw[stage] = {"pred": preds, "label": labels}

    labels0 = raw["Base"]["label"]
    min_len = min([len(labels0)] + [len(raw[s]["pred"]) for s in STAGES])

    valid_idx = []

    for i in range(min_len):
        ok = labels0[i] is not None
        for stage in STAGES:
            ok = ok and raw[stage]["pred"][i] is not None
        if ok:
            valid_idx.append(i)

    if not valid_idx:
        raise RuntimeError("No valid 8-task samples after alignment.")

    y_true = np.stack([labels0[i] for i in valid_idx], axis=0).astype(int)
    preds = {
        stage: np.stack([raw[stage]["pred"][i] for i in valid_idx], axis=0).astype(int)
        for stage in STAGES
    }

    print(f"\n[8-task alignment] valid samples: {len(y_true)}")
    return y_true, preds


def severity_state(y_true, y_pred):
    abs_err = np.abs(y_pred - y_true)

    states = np.empty(len(y_true), dtype=object)
    states[abs_err == 0] = "Correct"
    states[abs_err == 1] = "Adjacent"
    states[abs_err >= 2] = "Non-adjacent"

    return states


def severity_metrics(y_true, pred_dict):
    rows = []

    for stage in STAGES:
        y_pred = pred_dict[stage]
        abs_err = np.abs(y_pred - y_true)
        err = y_pred != y_true
        n_err = max(int(err.sum()), 1)

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        rows.append({
            "stage": stage,
            "n": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": p_macro,
            "macro_recall": r_macro,
            "macro_f1": f1_macro,
            "mae": float(abs_err.mean()),
            "error_rate": float(err.mean()),
            "correct_rate": float((abs_err == 0).mean()),
            "adjacent_error_rate": float((abs_err == 1).mean()),
            "non_adjacent_error_rate": float((abs_err >= 2).mean()),
            "adjacent_ratio_among_errors": float(((abs_err == 1) & err).sum() / n_err),
            "non_adjacent_ratio_among_errors": float(((abs_err >= 2) & err).sum() / n_err),
            "overestimation_rate": float((y_pred > y_true).mean()),
            "underestimation_rate": float((y_pred < y_true).mean()),
        })

    df = pd.DataFrame(rows)
    out = CSV_ROOT / "severity_structure_metrics.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out}")
    return df


def symptom_metrics(y_true, pred_dict):
    rows = []

    for stage in STAGES:
        y_pred = pred_dict[stage]

        exact = np.all(y_pred == y_true, axis=1).mean()
        ham = hamming_loss(y_true, y_pred)

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )

        fp_count = ((y_true == 0) & (y_pred == 1)).sum(axis=1)
        fn_count = ((y_true == 1) & (y_pred == 0)).sum(axis=1)

        rows.append({
            "stage": stage,
            "n": len(y_true),
            "exact_match": exact,
            "hamming_loss": ham,
            "macro_precision": p_macro,
            "macro_recall": r_macro,
            "macro_f1": f1_macro,
            "micro_precision": p_micro,
            "micro_recall": r_micro,
            "micro_f1": f1_micro,
            "mean_fp_count": float(fp_count.mean()),
            "mean_fn_count": float(fn_count.mean()),
        })

    df = pd.DataFrame(rows)
    out = CSV_ROOT / "symptom_prediction_metrics.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out}")
    return df


def jaccard_cooccurrence(Y):
    Y = np.asarray(Y).astype(int)
    n_labels = Y.shape[1]
    M = np.zeros((n_labels, n_labels), dtype=float)

    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                M[i, j] = Y[:, i].mean()
            else:
                inter = np.logical_and(Y[:, i] == 1, Y[:, j] == 1).sum()
                union = np.logical_or(Y[:, i] == 1, Y[:, j] == 1).sum()
                M[i, j] = inter / union if union > 0 else 0.0

    return M


def upper_triangle_values(M):
    idx = np.triu_indices_from(M, k=1)
    return M[idx]


def safe_pearson(a, b):
    a = np.asarray(a).astype(float)
    b = np.asarray(b).astype(float)

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def top_edge_overlap(M_ref, M_pred, top_k=8):
    idx = np.triu_indices_from(M_ref, k=1)

    ref_vals = M_ref[idx]
    pred_vals = M_pred[idx]

    ref_order = np.argsort(ref_vals)[::-1][:top_k]
    pred_order = np.argsort(pred_vals)[::-1][:top_k]

    return len(set(ref_order).intersection(set(pred_order))) / max(top_k, 1)


def symptom_network_metrics(y_true, pred_dict):
    M_true = jaccard_cooccurrence(y_true)
    true_vec = upper_triangle_values(M_true)

    rows = []
    matrices = {"Ground truth": M_true}

    for stage in STAGES:
        M_pred = jaccard_cooccurrence(pred_dict[stage])
        pred_vec = upper_triangle_values(M_pred)

        corr = safe_pearson(true_vec, pred_vec)
        rmse = float(np.sqrt(np.mean((true_vec - pred_vec) ** 2)))
        overlap = top_edge_overlap(M_true, M_pred, top_k=8)
        prev_err = float(np.mean(np.abs(np.diag(M_true) - np.diag(M_pred))))

        rows.append({
            "stage": stage,
            "network_corr": corr,
            "network_rmse": rmse,
            "top8_edge_overlap": overlap,
            "mean_prevalence_error": prev_err,
        })

        matrices[stage] = M_pred

    df = pd.DataFrame(rows)
    out = CSV_ROOT / "symptom_network_metrics.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out}")

    return M_true, matrices, df


def draw_ribbon(ax, x0, x1, y0_low, y0_high, y1_low, y1_high, color, alpha=0.58):
    dx = x1 - x0

    verts = [
        (x0, y0_low),
        (x0 + 0.45 * dx, y0_low),
        (x1 - 0.45 * dx, y1_low),
        (x1, y1_low),
        (x1, y1_high),
        (x1 - 0.45 * dx, y1_high),
        (x0 + 0.45 * dx, y0_high),
        (x0, y0_high),
        (x0, y0_low),
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]

    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=1,
    )

    ax.add_patch(patch)


def plot_wrong_only_error_repair_flow(y_true, pred4):
    setup_matplotlib()

    state_order = ["Correct", "Adjacent", "Non-adjacent"]
    state_colors = {
        "Correct": "#8BC7C4",
        "Adjacent": "#5B84B1",
        "Non-adjacent": "#E56B6F",
    }

    states_all = {
        stage: severity_state(y_true, pred4[stage])
        for stage in STAGES
    }

    base_wrong = states_all["Base"] != "Correct"
    n = int(base_wrong.sum())

    if n == 0:
        raise RuntimeError("Base has no wrong samples. Wrong-only flow cannot be drawn.")

    states = {
        stage: states_all[stage][base_wrong]
        for stage in STAGES
    }

    counts = {stage: {s: 0 for s in state_order} for stage in STAGES}
    for stage in STAGES:
        vals, cnts = np.unique(states[stage], return_counts=True)
        for v, c in zip(vals, cnts):
            counts[stage][v] = int(c)

    flows = {}
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        pair_counts = defaultdict(int)
        for sa, sb in zip(states[a], states[b]):
            pair_counts[(sa, sb)] += 1
        flows[(a, b)] = pair_counts

    flow_rows = []
    for (a, b), pair_counts in flows.items():
        for (sa, sb), c in pair_counts.items():
            flow_rows.append({
                "from_stage": a,
                "to_stage": b,
                "from_state": sa,
                "to_state": sb,
                "count": c,
                "ratio_among_base_errors": c / n,
            })

    flow_df = pd.DataFrame(flow_rows)
    flow_out = CSV_ROOT / "wrong_only_error_repair_flow.csv"
    flow_df.to_csv(flow_out, index=False, encoding="utf-8-sig")
    print(f"[Saved] {flow_out}")

    fig, ax = plt.subplots(figsize=(9.4, 4.9), constrained_layout=True, facecolor="white")

    x_pos = {stage: i for i, stage in enumerate(STAGES)}
    bar_w = 0.12
    gap = 0.020
    usable_h = 1.0 - gap * (len(state_order) - 1)

    segment_bounds = {}

    for stage in STAGES:
        y_top = 1.0
        segment_bounds[stage] = {}

        for s in state_order:
            h = counts[stage][s] / n * usable_h
            y_low = y_top - h
            y_high = y_top

            segment_bounds[stage][s] = [y_low, y_high]

            rect = Rectangle(
                (x_pos[stage] - bar_w / 2, y_low),
                bar_w,
                h,
                facecolor=state_colors[s],
                edgecolor="white",
                linewidth=1.2,
                zorder=4,
            )
            ax.add_patch(rect)

            if counts[stage][s] > 0 and h > 0.025:
                ax.text(
                    x_pos[stage],
                    (y_low + y_high) / 2,
                    f"{counts[stage][s]}\n{counts[stage][s] / n:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    zorder=5,
                )

            y_top = y_low - gap

    for a, b in zip(STAGES[:-1], STAGES[1:]):
        pair_counts = flows[(a, b)]

        source_cursor = {s: segment_bounds[a][s][0] for s in state_order}
        target_cursor = {s: segment_bounds[b][s][0] for s in state_order}

        for s0 in state_order:
            for s1 in state_order:
                c = pair_counts.get((s0, s1), 0)
                if c == 0:
                    continue

                h = c / n * usable_h

                y0_low = source_cursor[s0]
                y0_high = y0_low + h
                source_cursor[s0] = y0_high

                y1_low = target_cursor[s1]
                y1_high = y1_low + h
                target_cursor[s1] = y1_high

                draw_ribbon(
                    ax,
                    x_pos[a] + bar_w / 2,
                    x_pos[b] - bar_w / 2,
                    y0_low,
                    y0_high,
                    y1_low,
                    y1_high,
                    color=state_colors[s1],
                    alpha=0.42 if s0 == s1 else 0.62,
                )

    ax.set_xlim(-0.45, len(STAGES) - 0.55)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xticks([x_pos[s] for s in STAGES])
    ax.set_xticklabels(STAGES, fontsize=13)
    ax.set_ylabel("Proportion among Base errors", fontsize=14)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="both", labelsize=12, width=1.1, length=4)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=state_colors[s],
            markeredgecolor="none",
            markersize=12,
            label=s,
        )
        for s in state_order
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    out_png = OUTPUT_ROOT / "A_wrong_only_error_repair_flow.png"
    out_pdf = OUTPUT_ROOT / "A_wrong_only_error_repair_flow.pdf"

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def plot_ordinal_boundary_closure_core(sev_df):
    setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True, facecolor="white")

    stages = sev_df["stage"].tolist()
    x = np.arange(len(stages))

    correct = sev_df["correct_rate"].values
    adjacent = sev_df["adjacent_error_rate"].values
    non_adj = sev_df["non_adjacent_error_rate"].values

    axes[0].bar(
        x,
        correct,
        color="#8BC7C4",
        edgecolor="white",
        linewidth=1.0,
        label="Correct",
    )
    axes[0].bar(
        x,
        adjacent,
        bottom=correct,
        color="#5B84B1",
        edgecolor="white",
        linewidth=1.0,
        label="Adjacent error",
    )
    axes[0].bar(
        x,
        non_adj,
        bottom=correct + adjacent,
        color="#E56B6F",
        edgecolor="white",
        linewidth=1.0,
        label="Non-adjacent error",
    )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, fontsize=12)
    axes[0].set_ylabel("Sample proportion", fontsize=14)
    axes[0].set_ylim(0, 1.08)

    for i in range(len(stages)):
        axes[0].text(
            i,
            correct[i] / 2,
            f"{correct[i]:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

        if adjacent[i] >= 0.045:
            axes[0].text(
                i,
                correct[i] + adjacent[i] / 2,
                f"{adjacent[i]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

        if non_adj[i] >= 0.035:
            axes[0].text(
                i,
                correct[i] + adjacent[i] + non_adj[i] / 2,
                f"{non_adj[i]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
        elif non_adj[i] > 0:
            axes[0].text(
                i,
                1.015,
                f"{non_adj[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#E56B6F",
                clip_on=False,
            )

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    axes[1].plot(
        x,
        sev_df["mae"].values,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#123b63",
        label="Ordinal MAE",
    )
    axes[1].plot(
        x,
        sev_df["non_adjacent_error_rate"].values,
        marker="s",
        linewidth=2.2,
        markersize=7,
        color="#E56B6F",
        label="Non-adjacent error",
    )

    for i, v in enumerate(sev_df["mae"].values):
        axes[1].text(
            i,
            v + 0.018,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#123b63",
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages, fontsize=12)
    axes[1].set_ylabel("Rate / ordinal error", fontsize=14)
    axes[1].set_ylim(0, max(0.45, sev_df["mae"].max() * 1.18))

    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=12, width=1.1, length=4)

    out_png = OUTPUT_ROOT / "B_ordinal_boundary_closure_core.png"
    out_pdf = OUTPUT_ROOT / "B_ordinal_boundary_closure_core.pdf"

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")

def add_compact_panel_label(ax, text, fontsize=11):
    ax.text(
        0.5,
        1.06,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        transform=ax.transAxes,
        clip_on=False,
    )


def plot_symptom_network_matrices(matrices, net_df):
    setup_matplotlib()

    cmap = light_blue_cmap()
    panels = ["Ground truth"] + STAGES
    M_true = matrices["Ground truth"]
    vmax = max(0.45, np.max(M_true))

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.05 * len(panels), 2.95),
        constrained_layout=True,
        facecolor="white",
    )

    last_im = None

    for ax, name in zip(axes, panels):
        M = matrices[name]

        last_im = ax.imshow(
            M,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(SYMPTOM_LABELS, fontsize=8, rotation=45)
        ax.set_yticklabels(SYMPTOM_LABELS, fontsize=8)

        if name == "Ground truth":
            label = "Ground truth"
        else:
            row = net_df[net_df["stage"] == name].iloc[0]
            label = f"{name}\nr={row['network_corr']:.2f}, Top={row['top8_edge_overlap']:.2f}"

        add_compact_panel_label(ax, label, fontsize=10.5)

        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="#d9e6f2", linestyle="-", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.78, pad=0.012)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Co-occurrence strength", fontsize=11)

    out_png = OUTPUT_ROOT / "C1_symptom_network_matrices.png"
    out_pdf = OUTPUT_ROOT / "C1_symptom_network_matrices.pdf"

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def plot_symptom_network_error_matrices(matrices, net_df):
    setup_matplotlib()

    cmap = error_cmap()
    M_true = matrices["Ground truth"]

    err_vmax = 0.0
    for stage in STAGES:
        err_vmax = max(err_vmax, np.max(np.abs(matrices[stage] - M_true)))
    err_vmax = max(err_vmax, 0.05)

    fig, axes = plt.subplots(
        1,
        len(STAGES),
        figsize=(3.15 * len(STAGES), 2.95),
        constrained_layout=True,
        facecolor="white",
    )

    last_im = None

    for ax, stage in zip(axes, STAGES):
        E = np.abs(matrices[stage] - M_true)

        last_im = ax.imshow(
            E,
            cmap=cmap,
            vmin=0,
            vmax=err_vmax,
            interpolation="nearest",
        )

        row = net_df[net_df["stage"] == stage].iloc[0]
        add_compact_panel_label(ax, f"{stage}\nRMSE={row['network_rmse']:.2f}", fontsize=10.5)

        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(SYMPTOM_LABELS, fontsize=8, rotation=45)
        ax.set_yticklabels(SYMPTOM_LABELS, fontsize=8)

        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="#ead8d8", linestyle="-", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.78, pad=0.012)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Absolute error", fontsize=11)

    out_png = OUTPUT_ROOT / "C2_symptom_network_error_matrices.png"
    out_pdf = OUTPUT_ROOT / "C2_symptom_network_error_matrices.pdf"

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")

def plot_symptom_structure_fidelity_summary(net_df):
    setup_matplotlib()

    stages = net_df["stage"].tolist()
    x = np.arange(len(stages))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.65),
        constrained_layout=True,
        facecolor="white",
    )

    # Network correlation.
    vals = net_df["network_corr"].values
    axes[0].bar(
        x,
        vals,
        width=0.62,
        color="#5B84B1",
        edgecolor="#123b63",
        linewidth=1.1,
    )
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Correlation with ground truth", fontsize=13)
    axes[0].set_ylim(0, 1.08)

    # Top-edge overlap.
    vals = net_df["top8_edge_overlap"].values
    axes[1].bar(
        x,
        vals,
        width=0.62,
        color="#8BC7C4",
        edgecolor="#1f5f5b",
        linewidth=1.1,
    )
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Top-edge overlap", fontsize=13)
    axes[1].set_ylim(0, 1.08)

    # Prevalence error.
    vals = net_df["mean_prevalence_error"].values
    axes[2].bar(
        x,
        vals,
        width=0.62,
        color="#E56B6F",
        edgecolor="#7f1d1d",
        linewidth=1.1,
    )
    ymax = max(vals.max() * 1.35, 0.05)
    for i, v in enumerate(vals):
        axes[2].text(i, v + ymax * 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Mean prevalence error", fontsize=13)
    axes[2].set_ylim(0, ymax)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=11, rotation=18, ha="right")
        ax.set_xlabel("Stage", fontsize=13)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.tick_params(axis="both", labelsize=11, width=1.1, length=4)

    out_png = OUTPUT_ROOT / "D_symptom_structure_fidelity_summary.png"
    out_pdf = OUTPUT_ROOT / "D_symptom_structure_fidelity_summary.pdf"

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")

def write_counterfactual_probe_template():
    rows = [
        {
            "probe_id": "p01_level0",
            "edit_level": 0,
            "text": "I feel tired today, but I think I will be okay.",
            "expected_direction": "low severity; few or no symptoms",
        },
        {
            "probe_id": "p01_level1",
            "edit_level": 1,
            "text": "I feel tired and sad today, and it is hard to enjoy anything.",
            "expected_direction": "mild severity; sadness or emptiness may appear",
        },
        {
            "probe_id": "p01_level2",
            "edit_level": 2,
            "text": "I feel tired, empty, and hopeless, and I do not see things getting better.",
            "expected_direction": "moderate severity; emptiness and hopelessness should increase",
        },
        {
            "probe_id": "p01_level3",
            "edit_level": 3,
            "text": "I feel empty, worthless, and hopeless, and I keep thinking that I do not want to live.",
            "expected_direction": "severe severity; suicide intent and worthlessness should appear",
        },
    ]

    df = pd.DataFrame(rows)
    out = CSV_ROOT / "counterfactual_probe_template.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out}")


def main():
    print("[Project root]", PROJECT_ROOT)
    print("[LORA root]", LORA_ROOT)
    print("[Output root]", OUTPUT_ROOT)

    setup_matplotlib()

    check_paths(RUNS_4, "4-task")
    check_paths(RUNS_8, "8-task")

    y4_true, pred4 = load_4task(RUNS_4)
    y8_true, pred8 = load_8task(RUNS_8)

    sev_df = severity_metrics(y4_true, pred4)
    sym_df = symptom_metrics(y8_true, pred8)
    M_true, matrices, net_df = symptom_network_metrics(y8_true, pred8)

    merged = pd.DataFrame({"stage": STAGES})
    sev_map = sev_df.set_index("stage")
    sym_map = sym_df.set_index("stage")
    net_map = net_df.set_index("stage")

    merged["severity_accuracy"] = [sev_map.loc[s, "accuracy"] for s in STAGES]
    merged["severity_macro_f1"] = [sev_map.loc[s, "macro_f1"] for s in STAGES]
    merged["severity_mae"] = [sev_map.loc[s, "mae"] for s in STAGES]
    merged["severity_non_adjacent_error_rate"] = [sev_map.loc[s, "non_adjacent_error_rate"] for s in STAGES]
    merged["symptom_exact_match"] = [sym_map.loc[s, "exact_match"] for s in STAGES]
    merged["symptom_macro_f1"] = [sym_map.loc[s, "macro_f1"] for s in STAGES]
    merged["symptom_hamming_loss"] = [sym_map.loc[s, "hamming_loss"] for s in STAGES]
    merged["symptom_network_corr"] = [net_map.loc[s, "network_corr"] for s in STAGES]
    merged["symptom_top8_edge_overlap"] = [net_map.loc[s, "top8_edge_overlap"] for s in STAGES]
    merged["symptom_mean_prevalence_error"] = [net_map.loc[s, "mean_prevalence_error"] for s in STAGES]

    merged_out = CSV_ROOT / "core_interpretability_summary.csv"
    merged.to_csv(merged_out, index=False, encoding="utf-8-sig")
    print(f"[Saved] {merged_out}")

    plot_wrong_only_error_repair_flow(y4_true, pred4)
    plot_ordinal_boundary_closure_core(sev_df)
    plot_symptom_network_matrices(matrices, net_df)
    plot_symptom_network_error_matrices(matrices, net_df)
    plot_symptom_structure_fidelity_summary(net_df)

    write_counterfactual_probe_template()

    print("\nDone.")
    print(f"Core interpretability figures saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()