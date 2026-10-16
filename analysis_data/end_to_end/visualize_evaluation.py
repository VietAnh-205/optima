# -*- coding: utf-8 -*-
"""
Evaluation & Comprehensive Visualization for End-to-End Pipeline
Bao cao danh gia hieu nang mo hinh End-to-End (Chuan phong cach Corporate/Academic):
  - PHAN 0: Tong quan Dieu hanh (Executive Summary)
  - PHAN 1: Danh gia Mo hinh Phan loai Kho (Warehouse Classifier)
  - PHAN 2: Danh gia Mo hinh Hoi quy Thoi gian (Arrival Regressor)
  - PHAN 3: Danh gia Tuong tac Ket hop Hai Giai doan (Pipeline Interaction)
  - PHAN 4: Phan tich Chuyen sau Tuyen & Thoi gian (Deep Dive Insights)
"""

import os
import sys
import argparse
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
import plotly.graph_objects as go

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_data.end_to_end.multi_label import EndToEndPipeline
from analysis_data.end_to_end.e2e_preprocessor import (
    build_e2e_training_data,
    prepare_e2e_splits,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Danh gia toan dien Pipeline End-to-End")
    parser.add_argument("--model-path", type=str, default=os.path.join("saved_e2e_models", "e2e_pipeline.pkl"))
    parser.add_argument("--output-dir", type=str, default=os.path.join("saved_e2e_models", "eval_plots"))
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--max-eval-samples", type=int, default=20000)
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--cache-dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_e2e_models", "cache"))
    return parser.parse_args()


# ==============================================================================
# 1. HAM TAO CAC BIEU DO PLOTLY (THIET KE TOI GIAN, CHUYEN NGHIEP)
# ==============================================================================

# Bang mau xanh truyen thong (Blue Palette)
BLUE_PRIMARY = "#1d4ed8"
BLUE_DARK = "#1e3a8a"
BLUE_MEDIUM = "#2563eb"
BLUE_LIGHT = "#3b82f6"
BLUE_MUTED = "#60a5fa"
BLUE_PALE = "#dbeafe"
TEXT_DARK = "#1e293b"
TEXT_MUTED = "#475569"


def make_scatter_actual_vs_pred(y_true, y_pred, title, max_points=3500):
    """Scatter plot toi gian voi mau xanh truyen thong."""
    n = len(y_true)
    if n > max_points:
        np.random.seed(42)
        idx = np.random.choice(n, size=max_points, replace=False)
        yt_sample = y_true[idx]
        yp_sample = y_pred[idx]
    else:
        yt_sample = y_true
        yp_sample = y_pred

    errors = yp_sample - yt_sample
    max_val = max(float(np.percentile(yt_sample, 99.5)), float(np.percentile(yp_sample, 99.5)))
    max_val = max(10, max_val * 1.05)

    fig = go.Figure()

    x_band = np.linspace(0, max_val, 100)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_band, x_band[::-1]]),
        y=np.concatenate([np.maximum(0, x_band - 2), x_band[::-1] + 2]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.08)",
        line=dict(color="rgba(59, 130, 246, 0.3)", width=1, dash="dot"),
        hoverinfo="skip",
        name="Dung sai \u00b12h",
    ))

    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color=TEXT_DARK, dash="dash", width=1.5),
        name="Ly tuong (y = x)",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=yt_sample,
        y=yp_sample,
        mode="markers",
        marker=dict(
            size=3.5,
            color=BLUE_PRIMARY,
            opacity=0.25,
        ),
        name="Don hang (Mau)",
        customdata=errors,
        hovertemplate="Thuc te: %{x:.2f}h<br>Du doan: %{y:.2f}h<br>Sai lech: %{customdata:+.2f}h<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_DARK)),
        xaxis=dict(title=dict(text="Thoi gian Thuc te (Gio)", font=dict(size=11)), range=[0, max_val], tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="Thoi gian Du doan (Gio)", font=dict(size=11)), range=[0, max_val], tickfont=dict(size=10)),
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def make_error_histogram(y_true, y_pred, title):
    """Histogram phan phoi sai so voi sac thai xanh truyen thong."""
    errors = y_pred - y_true
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    p2 = np.percentile(errors, 1)
    p98 = np.percentile(errors, 99)
    range_limit = max(abs(p2), abs(p98), 10)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=80,
        marker=dict(color=BLUE_PRIMARY, line=dict(color=BLUE_DARK, width=0.5)),
        name="Sai so",
        hovertemplate="Sai so: %{x:.2f}h<br>So don: %{y:,}<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color=TEXT_DARK, width=1.5, dash="solid"),
                  annotation_text="Sai so = 0", annotation_position="top left",
                  annotation_font=dict(size=10, color=TEXT_DARK))
    fig.add_vline(x=mean_err, line=dict(color=BLUE_MUTED, width=1.5, dash="dash"),
                  annotation_text=f"Mean: {mean_err:+.2f}h", annotation_position="top right",
                  annotation_font=dict(size=10, color=BLUE_MEDIUM))

    fig.update_layout(
        title=dict(text=f"{title} (Mean: {mean_err:+.2f}h, Std: {std_err:.2f}h)", font=dict(size=14, color=TEXT_DARK)),
        xaxis=dict(title=dict(text="Sai so (Du doan - Thuc te) [Gio]", font=dict(size=11)), range=[-range_limit, range_limit], tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="So luong don hang", font=dict(size=11)), tickfont=dict(size=10)),
        height=430,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=60),
    )
    return fig


def make_cdf_chart(abs_err, title):
    """Duong cong CDF voi cac moc chuan, mau xanh truyen thong."""
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    p98 = float(np.percentile(sorted_err, 98))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_err,
        y=cdf,
        mode="lines",
        name="CDF Thuc te",
        line=dict(color=BLUE_PRIMARY, width=2.0),
        hovertemplate="Sai so tuyet doi: \u2264 %{x:.2f}h<br>Ty le tich luy: %{y:.1f}%<extra></extra>",
    ))

    for th in [1, 2, 3, 5]:
        pct = np.mean(abs_err <= th) * 100
        fig.add_vline(x=th, line_dash="dash", line_color=BLUE_MUTED, line_width=1.0,
                      annotation_text=f"\u00b1{th}h: {pct:.1f}%", annotation_position="bottom right",
                      annotation_font=dict(size=10, color=BLUE_MEDIUM))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_DARK)),
        xaxis=dict(title=dict(text="Sai so Tuyet doi [Gio]", font=dict(size=11)), range=[0, min(24, max(8, p98 * 1.1))], tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="Ty le Tich luy (%)", font=dict(size=11)), range=[0, 102], tickfont=dict(size=10)),
        height=430,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=60),
    )
    return fig


def make_box_by_warehouse(y_true_wh, abs_err, title, top_n=14):
    """Box plot sai so theo Kho 1A voi mau xanh truyen thong."""
    df_wh = pd.DataFrame({"warehouse": y_true_wh, "abs_err": abs_err})
    top_warehouses = df_wh["warehouse"].value_counts().head(top_n).index.tolist()
    df_filtered = df_wh[df_wh["warehouse"].isin(top_warehouses)].copy()

    sorted_order = df_filtered.groupby("warehouse")["abs_err"].median().sort_values().index.tolist()

    fig = go.Figure()
    for wh in sorted_order:
        data = df_filtered[df_filtered["warehouse"] == wh]["abs_err"]
        fig.add_trace(go.Box(
            y=data,
            name=f"{wh}",
            marker_color=BLUE_PRIMARY,
            line_width=1.2,
            boxpoints=False,
        ))

    cap_y = float(np.percentile(df_filtered["abs_err"], 97)) * 1.25
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_DARK)),
        xaxis=dict(title=dict(text="Kho 1A (Thu tu theo MedAE tang dan)", font=dict(size=11)), tickangle=35, tickfont=dict(size=9, color=TEXT_MUTED)),
        yaxis=dict(title=dict(text="Sai so Tuyet doi (Gio)", font=dict(size=11)), range=[0, cap_y], tickfont=dict(size=10)),
        height=480,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=120),
    )
    return fig


def make_box_by_hour(bill_dates, abs_err, title):
    """Box plot sai so theo khung gio tao don, mau xanh truyen thong."""
    dates = pd.Series(pd.to_datetime(bill_dates, errors="coerce"))
    hours = dates.dt.hour

    bins = [-1, 6, 12, 18, 24]
    labels = ["00:00 - 06:00 (Dem)", "06:00 - 12:00 (Sang)", "12:00 - 18:00 (Chieu)", "18:00 - 24:00 (Toi)"]
    hour_cats = pd.cut(hours, bins=bins, labels=labels)
    df_hour = pd.DataFrame({"time_slot": hour_cats, "abs_err": abs_err}).dropna()

    fig = go.Figure()
    for slot in labels:
        data = df_hour[df_hour["time_slot"] == slot]["abs_err"]
        if len(data) > 0:
            fig.add_trace(go.Box(
                y=data,
                name=f"{slot}<br>(N={len(data):,})",
                marker_color=BLUE_PRIMARY,
                line_width=1.2,
                boxpoints=False,
            ))

    cap_y = float(np.percentile(df_hour["abs_err"], 97)) * 1.25
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_DARK)),
        xaxis=dict(title=dict(text="Khung Gio Tao Don", font=dict(size=11)), tickfont=dict(size=10, color=TEXT_MUTED)),
        yaxis=dict(title=dict(text="Sai so Tuyet doi (Gio)", font=dict(size=11)), range=[0, cap_y], tickfont=dict(size=10)),
        height=430,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=70),
    )
    return fig


def make_time_series_chart(bill_dates, y_true, y_pred, title):
    """Time Series Actual vs Predicted theo bill_creation_date (2025 - 06/2026)."""
    dates = pd.Series(pd.to_datetime(bill_dates, errors="coerce"))
    tmp = pd.DataFrame({"date": dates, "actual": y_true, "predicted": y_pred}).dropna(subset=["date"])

    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2026-06-30 23:59:59")
    tmp = tmp[(tmp["date"] >= start_date) & (tmp["date"] <= end_date)].sort_values("date").reset_index(drop=True)

    if tmp.empty:
        tmp = pd.DataFrame({"date": dates, "actual": y_true, "predicted": y_pred}).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    span = tmp["date"].max() - tmp["date"].min()
    view_start = tmp["date"].min() + span * 0.02
    view_end = tmp["date"].min() + span * 0.98

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tmp["date"],
        y=tmp["actual"],
        mode="lines",
        name="Thuc te (Actual)",
        line=dict(color=TEXT_DARK, width=1.4),
        connectgaps=True,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Thuc te: %{y:.2f}h<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=tmp["date"],
        y=tmp["predicted"],
        mode="lines",
        name="Du doan (Predicted)",
        line=dict(color=BLUE_PRIMARY, width=1.4, dash="dash"),
        connectgaps=True,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Du doan: %{y:.2f}h<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_DARK)),
        xaxis=dict(title=dict(text="Thoi diem tao bill", font=dict(size=11)), range=[view_start, view_end], tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="Thoi gian toi kho 1A (Gio)", font=dict(size=11)), tickfont=dict(size=10)),
        hovermode="x unified",
        height=580,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def make_box_by_dayofweek(bill_dates, abs_err, title):
    """Box plot theo cac ngay trong tuan voi mau xanh truyen thong."""
    dates = pd.Series(pd.to_datetime(bill_dates, errors="coerce"))
    dow = dates.dt.dayofweek
    dow_names = ["Thu Hai", "Thu Ba", "Thu Tu", "Thu Nam", "Thu Sau", "Thu Bay", "Chu Nhat"]
    df_dow = pd.DataFrame({"dow": dow, "abs_err": abs_err}).dropna()

    fig = go.Figure()
    for i, name in enumerate(dow_names):
        data = df_dow[df_dow["dow"] == i]["abs_err"]
        if len(data) > 0:
            fig.add_trace(go.Box(
                y=data,
                name=f"{name}<br>(N={len(data):,})",
                marker_color=BLUE_PRIMARY,
                line_width=1.2,
                boxpoints=False,
            ))

    cap_y = float(np.percentile(df_dow["abs_err"], 97)) * 1.25
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_DARK)),
        xaxis=dict(title=dict(text="Ngay Trong Tuan", font=dict(size=11)), tickfont=dict(size=10, color=TEXT_MUTED)),
        yaxis=dict(title=dict(text="Sai so Tuyet doi (Gio)", font=dict(size=11)), range=[0, cap_y], tickfont=dict(size=10)),
        height=430,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=70),
    )
    return fig


# ==============================================================================
# 2. HAM TINH TOAN BANG METRICS & BANG PHAN TICH
# ==============================================================================

def get_top_misclassification_pairs(y_true, y_pred, top_n=10):
    wrong_mask = (y_true != y_pred)
    total_wrong = np.sum(wrong_mask)
    if total_wrong == 0:
        return []

    pairs = list(zip(y_true[wrong_mask], y_pred[wrong_mask]))
    most_common = Counter(pairs).most_common(top_n)

    result = []
    for rank, ((actual, pred), count) in enumerate(most_common, 1):
        pct = (count / total_wrong) * 100
        result.append({
            "rank": rank,
            "actual": actual,
            "pred": pred,
            "count": count,
            "pct": pct,
        })
    return result


def compute_stage_interaction(y_true_hours, y_pred_hours, y_true_first, y_pred_first):
    clf_correct = (y_true_first == y_pred_first)
    
    def calc_group_stats(mask):
        if np.sum(mask) == 0:
            return {"count": 0, "pct": 0, "mae": 0, "medae": 0, "rmse": 0, "within_2h": 0}
        sub_yt = y_true_hours[mask]
        sub_yp = y_pred_hours[mask]
        err = sub_yp - sub_yt
        abs_e = np.abs(err)
        return {
            "count": int(np.sum(mask)),
            "pct": float(np.mean(mask) * 100),
            "mae": float(mean_absolute_error(sub_yt, sub_yp)),
            "medae": float(median_absolute_error(sub_yt, sub_yp)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "within_2h": float(np.mean(abs_e <= 2.0) * 100),
        }

    correct_stats = calc_group_stats(clf_correct)
    wrong_stats = calc_group_stats(~clf_correct)

    delta = {
        "mae": wrong_stats["mae"] - correct_stats["mae"],
        "medae": wrong_stats["medae"] - correct_stats["medae"],
        "rmse": wrong_stats["rmse"] - correct_stats["rmse"],
        "within_2h": wrong_stats["within_2h"] - correct_stats["within_2h"],
    }
    return correct_stats, wrong_stats, delta


def get_top_error_routes(routes, abs_err, top_n=10, min_bills=15):
    if routes is None:
        return []
    
    df_r = pd.DataFrame({"route": routes, "abs_err": abs_err})
    grouped = df_r.groupby("route").agg(
        count=("abs_err", "count"),
        mae=("abs_err", "mean"),
        medae=("abs_err", "median"),
        within_2h=("abs_err", lambda x: np.mean(x <= 2.0) * 100),
    ).reset_index()

    grouped = grouped[grouped["count"] >= min_bills].sort_values("mae", ascending=False).head(top_n)
    return grouped.to_dict(orient="records")


# ==============================================================================
# 3. HAM TAO TOAN BO FILE BAO CAO HTML (GIAO DIEN CHUYEN NGHIEP, TOI GIAN)
# ==============================================================================

def build_html_report(
    y_true_hours,
    y_pred_hours,
    bill_dates,
    y_true_first,
    y_pred_first,
    top3_first,
    y_true_last,
    y_pred_last,
    top3_last,
    routes,
    output_path,
    shift_metrics=None,
    cutoff_comparison=None,
):
    print("  -> Tinh toan cac chi so hoi quy & phan loai...")
    errors = y_pred_hours - y_true_hours
    abs_err = np.abs(errors)
    sq_err = errors ** 2

    # Regressor Metrics
    n_samples = len(y_true_hours)
    mae = mean_absolute_error(y_true_hours, y_pred_hours)
    medae = median_absolute_error(y_true_hours, y_pred_hours)
    rmse = np.sqrt(np.mean(sq_err))
    percentage_errors = (abs_err / np.clip(np.abs(y_true_hours), 0.1, None)) * 100
    mape = np.mean(percentage_errors)
    mdape = np.median(percentage_errors)
    r2 = r2_score(y_true_hours, y_pred_hours)
    mean_bias = np.mean(errors)

    within_1h = np.mean(abs_err <= 1.0) * 100
    within_2h = np.mean(abs_err <= 2.0) * 100
    within_3h = np.mean(abs_err <= 3.0) * 100
    within_5h = np.mean(abs_err <= 5.0) * 100
    p90 = np.percentile(abs_err, 90)
    p95 = np.percentile(abs_err, 95)

    # Classifier Metrics
    f1_top1 = accuracy_score(y_true_first, y_pred_first) * 100
    f1_weighted_f1 = f1_score(y_true_first, y_pred_first, average="weighted", zero_division=0)
    f1_macro_f1 = f1_score(y_true_first, y_pred_first, average="macro", zero_division=0)
    
    top3_first_corr = [truth in top3 for truth, top3 in zip(y_true_first, top3_first)] if top3_first is not None else []
    f1_top3 = (np.mean(top3_first_corr) * 100) if len(top3_first_corr) > 0 else 0.0

    last_top1 = accuracy_score(y_true_last, y_pred_last) * 100
    last_weighted_f1 = f1_score(y_true_last, y_pred_last, average="weighted", zero_division=0)
    last_macro_f1 = f1_score(y_true_last, y_pred_last, average="macro", zero_division=0)

    top3_last_corr = [truth in top3 for truth, top3 in zip(y_true_last, top3_last)] if top3_last is not None else []
    last_top3 = (np.mean(top3_last_corr) * 100) if len(top3_last_corr) > 0 else 0.0

    both_correct_acc = np.mean((y_true_first == y_pred_first) & (y_true_last == y_pred_last)) * 100

    if top3_first is not None and top3_last is not None:
        both_top3_corr = [
            (t_f in p_f) and (t_l in p_l)
            for t_f, p_f, t_l, p_l in zip(y_true_first, top3_first, y_true_last, top3_last)
        ]
        both_top3_acc = (np.mean(both_top3_corr) * 100) if len(both_top3_corr) > 0 else 0.0
    else:
        both_top3_acc = 0.0

    # Combined F1 scores: trung binh co trong so giua first_1A va last_1A
    combined_weighted_f1 = (f1_weighted_f1 + last_weighted_f1) / 2
    combined_macro_f1 = (f1_macro_f1 + last_macro_f1) / 2

    # Interaction & Top errors
    top_mis_first = get_top_misclassification_pairs(y_true_first, y_pred_first, top_n=10)
    top_mis_last = get_top_misclassification_pairs(y_true_last, y_pred_last, top_n=10)
    correct_stats, wrong_stats, delta_stats = compute_stage_interaction(y_true_hours, y_pred_hours, y_true_first, y_pred_first)
    top_routes = get_top_error_routes(routes, abs_err, top_n=10, min_bills=15)

    print("  -> Khoi tao bieu do Plotly tuong tac...")

    # P2 Charts
    fig_scatter = make_scatter_actual_vs_pred(y_true_hours, y_pred_hours, "2.2 Bieu Do Tuong Quan: Thuc Te vs Du Doan (Identity Scatter)")
    fig_hist = make_error_histogram(y_true_hours, y_pred_hours, "2.3 Phan Phoi Sai So (Error Distribution)")
    fig_cdf = make_cdf_chart(abs_err, "2.4 Duong Cong Tich Luy Sai So Tuyet Doi & Moc Dung Sai")
    fig_box_wh = make_box_by_warehouse(y_true_first, abs_err, "2.5 Phan Phoi Sai So Thoi Gian Theo Kho first_1A", top_n=14)
    fig_box_hr = make_box_by_hour(bill_dates, abs_err, "2.6 Phan Phoi Sai So Thoi Gian Theo Khung Gio Tao Bill")

    # P3 Charts
    fig_ts = make_time_series_chart(bill_dates, y_true_hours, y_pred_hours, "3.1 Chuoi Thoi Gian: Thuc Te vs Du Doan (2025 - 06/2026)")

    # P4 Charts
    fig_box_dow = make_box_by_dayofweek(bill_dates, abs_err, "4.2 Phan Phoi Sai So Thoi Gian Theo Cac Ngay Trong Tuan")

    print("  -> Dong goi toan bo bieu do sang dinh dang HTML tuong tac...")

    plotly_figs = [
        ("scatter", fig_scatter),
        ("hist", fig_hist),
        ("cdf", fig_cdf),
        ("box_wh", fig_box_wh),
        ("box_hr", fig_box_hr),
        ("ts", fig_ts),
        ("box_dow", fig_box_dow),
    ]

    fig_divs = {}
    for i, (key, f) in enumerate(plotly_figs):
        fig_divs[key] = f.to_html(full_html=False, include_plotlyjs=(i == 0))

    def render_shift_table(s_metrics):
        if not s_metrics or not s_metrics.get("rows"):
            return "<p class='text-muted'>Không có dữ liệu ca vận hành.</p>"
        rows_html = ""
        for r in s_metrics["rows"]:
            rows_html += f"""
            <tr>
              <td><b>{r['name']}</b></td>
              <td class="text-right">{r['n_actual']:,}</td>
              <td class="text-right">{r['n_pred']:,}</td>
              <td class="text-right font-medium" style="color:var(--blue-primary);">{r['n_inter']:,}</td>
              <td class="text-right">{r['recall']:.1f}%</td>
              <td class="text-right">{r['precision']:.1f}%</td>
              <td class="text-right font-medium">{r['f1']:.1f}%</td>
            </tr>
            """
        return f"""
        <div class="table-container">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Ca Vận Hành</th>
                <th class="text-right">Số Đơn Thực Tế</th>
                <th class="text-right">Số Đơn Dự Đoán</th>
                <th class="text-right">Khớp (A ∩ P)</th>
                <th class="text-right">Recall (%)</th>
                <th class="text-right">Precision (%)</th>
                <th class="text-right">F1-Score (%)</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
          <div style="margin-top: 12px; font-size: 13px; color: var(--text-muted);">
            &bull; <b>Tỷ lệ dự đoán đúng ca toàn mạng lưới (Shift Accuracy):</b> <span style="color:var(--blue-dark); font-weight:700;">{s_metrics['shift_acc']:.2f}%</span><br>
            &bull; <i>Ghi chú:</i> Ca Đêm (22h00 - 06h00), Ca Sáng (06h00 - 14h00), Ca Chiều (14h00 - 22h00).
          </div>
        </div>
        """

    def render_cutoff_comparison(c_comp):
        if not c_comp:
            return "<p class='text-muted'>Không có dữ liệu so sánh lát cắt chốt số.</p>"
        return f"""
        <div class="table-container">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Góc Nhìn Đánh Giá</th>
                <th class="text-right">Số Mẫu</th>
                <th class="text-right">MAE (giờ)</th>
                <th class="text-right">MedAE (giờ)</th>
                <th class="text-right">Trong ±2.0h (%)</th>
                <th class="text-right">Trong ±3.0h (%)</th>
                <th>Ý Nghĩa Nghiệp Vụ</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>Đa Snapshot (Observation-level)</b></td>
                <td class="text-right">{c_comp['n_total_snapshots']:,}</td>
                <td class="text-right">{c_comp['all_mae']:.2f} h</td>
                <td class="text-right">{c_comp['all_medae']:.2f} h</td>
                <td class="text-right">{c_comp['all_within_2h']:.1f}%</td>
                <td class="text-right">{c_comp['all_within_3h']:.1f}%</td>
                <td>Đánh giá vi mô trên mọi thời điểm quét mã của đơn hàng.</td>
              </tr>
              <tr style="background-color: #f0fdf4;">
                <td><b>Lát Cắt Chốt Số (Unique Bill Cut-off)</b></td>
                <td class="text-right font-medium">{c_comp['n_unique_bills']:,}</td>
                <td class="text-right font-medium" style="color:#15803d;">{c_comp['unique_mae']:.2f} h</td>
                <td class="text-right font-medium" style="color:#15803d;">{c_comp['unique_medae']:.2f} h</td>
                <td class="text-right font-medium" style="color:#15803d;">{c_comp['unique_within_2h']:.1f}%</td>
                <td class="text-right font-medium" style="color:#15803d;">{c_comp['unique_within_3h']:.1f}%</td>
                <td><b>Mô phỏng thời điểm chốt ca:</b> Chỉ lấy trạng thái mới nhất trước khi đơn hàng về kho.</td>
              </tr>
            </tbody>
          </table>
          <div style="margin-top: 14px; padding: 12px 16px; background: #eff6ff; border-left: 4px solid var(--blue-primary); border-radius: 4px; font-size: 13px; line-height: 1.6;">
            <b>Khuyến nghị Vận hành:</b> Khi người điều phối chốt số liệu ca tại thời điểm $T_{{cut}}$, hệ thống chỉ truy vấn trạng thái mới nhất của các đơn hàng đang luân chuyển. Đánh giá trên lát cắt chốt số giúp phản ánh chính xác 100% dung lượng thực tế, loại trừ hoàn toàn hiện tượng nhân đôi số lượng/khối lượng do hợp nhiều snapshot.
          </div>
        </div>
        """

    def render_mis_table(mis_list, wh_title):
        if not mis_list:
            return "<p class='text-muted'>Khong ghi nhan loi phan loai nao.</p>"
        rows = ""
        for r in mis_list:
            rows += f"""
            <tr>
              <td class="text-center font-medium">{r['rank']}</td>
              <td>{r['actual']}</td>
              <td>{r['pred']}</td>
              <td class="text-right font-medium">{r['count']:,}</td>
              <td class="text-right">{r['pct']:.1f}%</td>
            </tr>
            """
        return f"""
        <div class="table-container">
          <div class="table-header">{wh_title}</div>
          <table class="custom-table">
            <thead>
              <tr>
                <th style="width: 50px;">Hang</th>
                <th>Kho Thuc Te</th>
                <th>Kho Du Doan Nham</th>
                <th class="text-right">So Luong Don</th>
                <th class="text-right">% Tong Loi</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """

    first_mis_html = render_mis_table(top_mis_first, "Top 10 Cap Loi Nham Kho first_1A")
    last_mis_html = render_mis_table(top_mis_last, "Top 10 Cap Loi Nham Kho last_1A")

    route_rows = ""
    for rank, r in enumerate(top_routes, 1):
        route_rows += f"""
        <tr>
          <td class="text-center font-medium">{rank}</td>
          <td class="font-medium">{r['route']}</td>
          <td class="text-right">{r['count']:,}</td>
          <td class="text-right font-medium">{r['mae']:.2f} h</td>
          <td class="text-right">{r['medae']:.2f} h</td>
          <td class="text-right">{r['within_2h']:.1f}%</td>
        </tr>
        """

    route_table_html = f"""
    <div class="table-container">
      <table class="custom-table">
        <thead>
          <tr>
            <th style="width: 50px;">Hang</th>
            <th>Tuyen Van Chuyen (Province Route)</th>
            <th class="text-right">So Luong Don</th>
            <th class="text-right">MAE (Gio)</th>
            <th class="text-right">MedAE (Gio)</th>
            <th class="text-right">Dat Chuan \u00b12h (%)</th>
          </tr>
        </thead>
        <tbody>{route_rows}</tbody>
      </table>
    </div>
    """ if route_rows else "<p class='text-muted'>Khong co du lieu tuyen duong (hoac moi tuyen co it hon 15 don hang).</p>"

    css = """
    :root {
      --bg-page: #f8fafc;
      --bg-card: #ffffff;
      --border: #e2e8f0;
      --border-dark: #cbd5e1;
      --text-main: #1e293b;
      --text-muted: #475569;
      --primary: #1d4ed8;
      --primary-light: #dbeafe;
      --primary-medium: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg-page);
      color: var(--text-main);
      margin: 0;
      padding: 0;
      line-height: 1.5;
      font-size: 14px;
    }
    .header-banner {
      background-color: #ffffff;
      border-bottom: 1px solid var(--border);
      padding: 28px 32px;
    }
    .header-title {
      font-size: 22px;
      font-weight: 700;
      color: var(--text-main);
      margin: 0 0 8px 0;
      letter-spacing: -0.3px;
    }
    .header-subtitle {
      color: var(--text-muted);
      font-size: 13px;
      margin: 0;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
    }
    .nav-bar {
      position: sticky;
      top: 0;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
      padding: 8px 32px;
      display: flex;
      gap: 16px;
      z-index: 1000;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
      overflow-x: auto;
    }
    .nav-link {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 6px;
      transition: background 0.15s, color 0.15s;
      white-space: nowrap;
    }
    .nav-link:hover {
      color: var(--primary);
      background-color: var(--primary-light);
    }
    .main-container {
      max-width: 1380px;
      margin: 28px auto 60px auto;
      padding: 0 24px;
    }
    .section-block {
      margin-bottom: 44px;
      scroll-margin-top: 60px;
    }
    .section-title {
      font-size: 17px;
      font-weight: 700;
      color: #1e293b;
      margin: 0 0 16px 0;
      padding-bottom: 8px;
      border-bottom: 2px solid #cbd5e1;
      letter-spacing: -0.2px;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 18px;
    }
    .kpi-label {
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.4px;
      margin-bottom: 6px;
    }
    .kpi-value {
      font-size: 28px;
      font-weight: 700;
      color: var(--primary);
      line-height: 1.15;
    }
    .kpi-sub {
      margin-top: 6px;
      font-size: 12px;
      color: var(--text-muted);
    }
    .kpi-tag {
      display: inline-block;
      background: var(--primary-light);
      color: var(--primary);
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 11.5px;
    }
    .card-box {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 22px 24px;
      margin-bottom: 18px;
    }
    .card-title {
      font-size: 15px;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 6px;
    }
    .card-desc {
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 16px;
      line-height: 1.6;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    @media (max-width: 900px) {
      .grid-2 { grid-template-columns: 1fr; }
    }
    .table-container {
      overflow-x: auto;
      margin-top: 12px;
    }
    .table-header {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 8px;
    }
    .custom-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .custom-table th {
      background-color: #f1f5f9;
      color: var(--text-main);
      font-weight: 600;
      padding: 10px 14px;
      text-align: left;
      border-bottom: 2px solid var(--border-dark);
      white-space: nowrap;
    }
    .custom-table td {
      padding: 9px 14px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    .custom-table tbody tr:hover {
      background-color: #f8fafc;
    }
    .text-right { text-align: right; }
    .text-center { text-align: center; }
    .font-medium { font-weight: 600; }
    .alert-box {
      background-color: var(--primary-light);
      border: 1px solid var(--primary);
      border-radius: 6px;
      padding: 14px 18px;
      margin-top: 16px;
      font-size: 13px;
      line-height: 1.6;
      color: var(--text-main);
    }
    .footnote {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 12px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
      line-height: 1.6;
    }
    .footnote sup {
      color: var(--primary);
      font-weight: 600;
    }
    """

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bao Cao Danh Gia Pipeline End-to-End</title>
  <style>{css}</style>
</head>
<body>

  <header class="header-banner">
    <div class="header-title">B\u00e1o C\u00e1o \u0110\u00e1nh Gi\u00e1 Hi\u1ec7u N\u0103ng M\u00f4 H\u00ecnh End-to-End Pipeline</div>
    <div class="header-subtitle">
      <span>S\u1ed1 m\u1eabu \u0111\u00e1nh gi\u00e1: <b>{n_samples:,} bills</b></span>
      <span>C\u1ea5u tr\u00fac m\u00f4 h\u00ecnh: <b>Cascade Classifier + Segmented Regressor (LightGBM)</b></span>
    </div>
  </header>

  <nav class="nav-bar">
    <a href="#section-p0" class="nav-link">P0: T\u1ed5ng Quan \u0110i\u1ec1u H\u00e0nh</a>
    <a href="#section-p1" class="nav-link">P1: Ph\u00e2n Lo\u1ea1i Kho (Warehouse Classifier)</a>
    <a href="#section-p2" class="nav-link">P2: H\u1ed3i Quy Th\u1eddi Gian (Arrival Regressor)</a>
    <a href="#section-p3" class="nav-link">P3: \u0110\u00e1nh Gi\u00e1 K\u1ebft H\u1ee3p (Pipeline Interaction)</a>
    <a href="#section-p4" class="nav-link">P4: Ph\u00e2n T\u00edch Chuy\u00ean S\u00e2u (Deep Dive)</a>
  </nav>

  <main class="main-container">

    <section id="section-p0" class="section-block">
      <h2 class="section-title">PH\u1ea6N 0: T\u1ed4NG QUAN \u0110I\u1ec0U H\u00c0NH (EXECUTIVE SUMMARY)</h2>
      
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Kho \u0110\u1ea7u (first_1A) Top-1</div>
          <div class="kpi-value">{f1_top1:.1f}%</div>
          <div class="kpi-sub">
            <span class="kpi-tag">Top-3: {f1_top3:.1f}%</span>
          </div>
        </div>

        <div class="kpi-card">
          <div class="kpi-label">Kho Cu\u1ed1i (last_1A) Top-1</div>
          <div class="kpi-value">{last_top1:.1f}%</div>
          <div class="kpi-sub">
            <span class="kpi-tag">Top-3: {last_top3:.1f}%</span>
          </div>
        </div>

        <div class="kpi-card">
          <div class="kpi-label">\u0110\u00fang C\u1ea3 2 Kho (Combined)</div>
          <div class="kpi-value">{both_correct_acc:.1f}%</div>
          <div class="kpi-sub">
            <span class="kpi-tag">{int(both_correct_acc * n_samples / 100):,} / {n_samples:,} bills</span>
          </div>
        </div>

        <div class="kpi-card">
          <div class="kpi-label">Regressor MAE & MdAPE</div>
          <div class="kpi-value">{mae:.2f} h</div>
          <div class="kpi-sub">
            <span>MedAE: <b>{medae:.2f} h</b> &bull; MdAPE: <b>{mdape:.1f}%</b></span>
          </div>
        </div>

        <div class="kpi-card">
          <div class="kpi-label">Regressor RMSE & R\u00b2</div>
          <div class="kpi-value">{rmse:.2f} h</div>
          <div class="kpi-sub">
            <span>R\u00b2 Score: <b>{r2:.3f}</b></span>
          </div>
        </div>

        <div class="kpi-card">
          <div class="kpi-label">T\u1ef7 l\u1ec7 trong dung sai<sup>[1]</sup></div>
          <div class="kpi-value">{within_2h:.1f}%</div>
          <div class="kpi-sub">
            <span class="kpi-tag">Trong \u00b13h: {within_3h:.1f}%</span>
          </div>
        </div>
      </div>

      <div class="card-box">
        <div class="card-title">B\u1ea3ng T\u1ed5ng H\u1ee3p Ch\u1ec9 S\u1ed1 C\u1ed1t L\u00f5i (Core Metrics Dashboard)</div>
        <div class="card-desc">\u0110\u1ed1i chi\u1ebfu k\u1ebft qu\u1ea3 ki\u1ec3m th\u1eed tr\u00ean to\u00e0n b\u1ed9 t\u1eadp d\u1eef li\u1ec7u test \u0111\u1ed9c l\u1eadp ({n_samples:,} v\u1eadn \u0111\u01a1n).</div>
        <div class="table-container">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Th\u00e0nh Ph\u1ea7n</th>
                <th>Ch\u1ec9 S\u1ed1 \u0110\u00e1nh Gi\u00e1</th>
                <th class="text-right">K\u1ebft Qu\u1ea3</th>
                <th>\u0110\u00e1nh Gi\u00e1 K\u1ef9 Thu\u1eadt</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Warehouse Classifier</td>
                <td>First 1A Top-1 Accuracy</td>
                <td class="text-right font-medium">{f1_top1:.2f}%</td>
                <td>Kh\u1ea3 n\u0103ng nh\u1eadn di\u1ec7n tr\u1ea1m 1A \u0111\u1ea7u v\u00e0o. T\u1ef7 l\u1ec7 d\u1ef1 \u0111o\u00e1n \u0111\u00fang l\u1ef1a ch\u1ecdn \u01b0u ti\u00ean s\u1ed1 1.</td>
              </tr>
              <tr>
                <td>Warehouse Classifier</td>
                <td>Last 1A Top-1 Accuracy</td>
                <td class="text-right font-medium">{last_top1:.2f}%</td>
                <td>Cascade classifier t\u1eadn d\u1ee5ng nh\u00e3n kho \u0111\u1ea7u ch\u00ednh x\u00e1c \u0111\u1ec3 d\u1ef1 \u0111o\u00e1n kho cu\u1ed1i.</td>
              </tr>
              <tr>
                <td>Warehouse Classifier</td>
                <td>Combined Both Correct Accuracy</td>
                <td class="text-right font-medium">{both_correct_acc:.2f}%</td>
                <td>T\u1ef7 l\u1ec7 d\u1ef1 \u0111o\u00e1n kh\u1edbp tr\u1ecdn v\u1eb9n c\u1ea3 kho \u0111\u1ea7u (first_1A) v\u00e0 kho cu\u1ed1i (last_1A) tr\u00ean c\u00f9ng m\u1ed9t \u0111\u01a1n h\u00e0ng.</td>
              </tr>
              <tr>
                <td>Arrival Regressor</td>
                <td>Mean Absolute Error (MAE)</td>
                <td class="text-right font-medium">{mae:.2f} gi\u1edd</td>
                <td>Sai s\u1ed1 trung b\u00ecnh tuy\u1ec7t \u0111\u1ed1i to\u00e0n m\u1ea1ng l\u01b0\u1edbi gi\u1eefa th\u1eddi gian d\u1ef1 \u0111o\u00e1n v\u00e0 th\u1eddi gian th\u1ef1c t\u1ebf t\u1edbi kho 1A.</td>
              </tr>
              <tr>
                <td>Arrival Regressor</td>
                <td>Median Absolute Error (MedAE)</td>
                <td class="text-right font-medium">{medae:.2f} gi\u1edd</td>
                <td>Trung v\u1ecb sai s\u1ed1: 50% \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 d\u01b0\u1edbi {medae:.2f} gi\u1edd. Kh\u00f4ng b\u1ecb \u1ea3nh h\u01b0\u1edfng b\u1edfi outlier.</td>
              </tr>
              <tr>
                <td>Arrival Regressor</td>
                <td>Root Mean Squared Error (RMSE)</td>
                <td class="text-right font-medium">{rmse:.2f} gi\u1edd</td>
                <td>Ph\u1ea1t n\u1eb7ng c\u00e1c \u0111\u01a1n c\u00f3 sai s\u1ed1 l\u1edbn. Ch\u1ecbu t\u00e1c \u0111\u1ed9ng t\u1eeb \u0111u\u00f4i sai s\u1ed1 c\u1ee7a c\u00e1c \u0111\u01a1n ch\u1edd xe gom/qua \u0111\u00eam.</td>
              </tr>
              <tr>
                <td>Arrival Regressor</td>
                <td>H\u1ec7 s\u1ed1 X\u00e1c \u0111\u1ecbnh (R\u00b2 Score)</td>
                <td class="text-right font-medium">{r2:.3f}</td>
                <td>M\u00f4 h\u00ecnh gi\u1ea3i th\u00edch {r2*100:.1f}% ph\u01b0\u01a1ng sai th\u1eddi gian v\u1eadn h\u00e0nh so v\u1edbi \u0111\u01b0\u1eddng c\u01a1 s\u1edf (baseline).</td>
              </tr>
              <tr>
                <td>Arrival Regressor</td>
                <td>Median Abs Percentage Error (MdAPE)<sup>[2]</sup></td>
                <td class="text-right font-medium">{mdape:.1f}%</td>
                <td>Trung v\u1ecb sai s\u1ed1 ph\u1ea7n tr\u0103m: 50% \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 d\u01b0\u1edbi {mdape:.1f}%. Th\u01b0\u1edbc \u0111o kh\u00e1ch quan kh\u00f4ng b\u1ecb m\u00e9o b\u1edfi c\u00e1c \u0111\u01a1n g\u1ea7n kho (&lt; 1h).</td>
              </tr>
              <tr>
                <td>Dung sai \u00b12h<sup>[1]</sup></td>
                <td>T\u1ef7 l\u1ec7 trong dung sai \u00b12.0 gi\u1edd</td>
                <td class="text-right font-medium">{within_2h:.1f}%</td>
                <td>T\u1ef7 l\u1ec7 \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 tuy\u1ec7t \u0111\u1ed1i (|d\u1ef1 \u0111o\u00e1n - th\u1ef1c t\u1ebf|) n\u1eb1m trong kho\u1ea3ng 2 gi\u1edd.</td>
              </tr>
              <tr>
                <td>Dung sai \u00b13h<sup>[1]</sup></td>
                <td>T\u1ef7 l\u1ec7 trong dung sai \u00b13.0 gi\u1edd</td>
                <td class="text-right font-medium">{within_3h:.1f}%</td>
                <td>T\u1ef7 l\u1ec7 \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 tuy\u1ec7t \u0111\u1ed1i n\u1eb1m trong kho\u1ea3ng 3 gi\u1edd.</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="footnote">
          <sup>[1]</sup> <b>Dung sai \u00b1Xh (Tolerance):</b> L\u00e0 ng\u01b0\u1ee1ng ch\u1ea5p nh\u1eadn sai s\u1ed1 gi\u1eefa th\u1eddi gian d\u1ef1 \u0111o\u00e1n v\u00e0 th\u1eddi gian th\u1ef1c t\u1ebf. V\u00ed d\u1ee5: "Dung sai \u00b12h" ngh\u0129a l\u00e0 n\u1ebfu m\u00f4 h\u00ecnh d\u1ef1 \u0111o\u00e1n h\u00e0ng t\u1edbi kho 1A l\u00fac 14:00 v\u00e0 th\u1ef1c t\u1ebf h\u00e0ng t\u1edbi l\u00fac 15:30, th\u00ec sai l\u1ec7ch l\u00e0 1.5 gi\u1edd, n\u1eb1m trong dung sai \u00b12h &rarr; \u0111\u01b0\u1ee3c t\u00ednh l\u00e0 "\u0111\u1ea1t chu\u1ea9n".<br>
          <sup>[2]</sup> <b>MdAPE vs MAPE:</b> Trong b\u00e0i to\u00e1n th\u1eddi gian logistics c\u00f3 nhi\u1ec1u \u0111\u01a1n t\u1edbi kho r\u1ea5t nhanh (&lt; 1h, chi\u1ebfm 12.7% m\u1eabu), ph\u00e9p chia cho m\u1eabu s\u1ed1 nh\u1ecf l\u00e0m MAPE trung b\u00ecnh c\u1ed9ng b\u1ecb v\u1ecdt l\u00ean cao (hi\u1ec7n t\u01b0\u1ee3ng Zero-Divisor). <b>MdAPE (Median Absolute Percentage Error)</b> l\u1ea5y trung v\u1ecb sai s\u1ed1 ph\u1ea7n tr\u0103m, ph\u1ea3n \u00e1nh ch\u00ednh x\u00e1c 50% s\u1ed1 \u0111\u01a1n th\u00f4ng th\u01b0\u1eddng ch\u1ec9 sai l\u1ec7ch kho\u1ea3ng {mdape:.1f}%.
        </div>
      </div>

      <div class="card-box">
        <div class="card-title">Gi\u1ea3i th\u00edch v\u1ec1 s\u1ed1 m\u1eabu \u0111\u00e1nh gi\u00e1 ({n_samples:,} bills)</div>
        <div class="card-desc">
          T\u1eadp d\u1eef li\u1ec7u g\u1ed1c \u0111\u01b0\u1ee3c chia theo t\u1ef7 l\u1ec7 th\u1eddi gian (time-based split) th\u00e0nh 3 ph\u1ea7n: <b>Train (60%)</b>, <b>Validation (20%)</b>, v\u00e0 <b>Test (20%)</b>.
          Qu\u00e1 tr\u00ecnh chia d\u1eef li\u1ec7u \u0111\u01b0\u1ee3c th\u1ef1c hi\u1ec7n theo th\u1ee9 t\u1ef1 th\u1eddi gian (arrival_time) c\u1ee7a t\u1eebng bill_code, \u0111\u1ea3m b\u1ea3o kh\u00f4ng c\u00f3 r\u00f2 r\u1ec9 d\u1eef li\u1ec7u t\u01b0\u01a1ng lai v\u00e0o t\u1eadp hu\u1ea5n luy\u1ec7n.
          Sau khi chia, t\u1eadp test ti\u1ebfp t\u1ee5c \u0111\u01b0\u1ee3c l\u1ecdc:
          (a) lo\u1ea1i b\u1ecf c\u00e1c \u0111\u01a1n h\u00e0ng kh\u00f4ng c\u00f3 gi\u00e1 tr\u1ecb target (time_to_first_1a_hours);
          (b) lo\u1ea1i b\u1ecf outlier theo ng\u01b0\u1ee1ng IQR \u0111\u01b0\u1ee3c t\u00ednh t\u1eeb t\u1eadp train.
          K\u1ebft qu\u1ea3 cu\u1ed1i c\u00f9ng l\u00e0 <b>{n_samples:,} v\u1eadn \u0111\u01a1n</b> tham gia \u0111\u00e1nh gi\u00e1 - \u0111\u00e2y l\u00e0 t\u1eadp test \u0111\u1ed9c l\u1eadp, ho\u00e0n to\u00e0n kh\u00f4ng tham gia v\u00e0o qu\u00e1 tr\u00ecnh hu\u1ea5n luy\u1ec7n hay \u0111i\u1ec1u ch\u1ec9nh m\u00f4 h\u00ecnh.
        </div>
      </div>
    </section>

    <section id="section-p1" class="section-block">
      <h2 class="section-title">PH\u1ea6N 1: \u0110\u00c1NH GI\u00c1 M\u00d4 H\u00ccNH PH\u00c2N LO\u1ea0I KHO (WAREHOUSE CLASSIFIER)</h2>

      <div class="card-box">
        <div class="card-title">1.1 B\u1ea3ng Thang \u0110o \u0110\u00e1nh Gi\u00e1 Ph\u00e2n Lo\u1ea1i Kho</div>
        <div class="card-desc">
          \u0110\u1ed1i chi\u1ebfu chi ti\u1ebft gi\u1eefa m\u00f4 h\u00ecnh tr\u1ea1m \u0111\u1ea7u (first_1A) v\u00e0 m\u00f4 h\u00ecnh tr\u1ea1m cu\u1ed1i (last_1A).
          C\u1ed9t "C\u1ea3 Hai Kho (Combined)" th\u1ec3 hi\u1ec7n k\u1ebft qu\u1ea3 khi y\u00eau c\u1ea7u c\u1ea3 hai kho \u0111\u1ea7u v\u00e0 kho cu\u1ed1i \u0111\u1ec1u ph\u1ea3i \u0111\u00fang tr\u00ean c\u00f9ng m\u1ed9t \u0111\u01a1n h\u00e0ng.
        </div>
        <div class="table-container">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Ch\u1ec9 S\u1ed1 \u0110\u00e1nh Gi\u00e1</th>
                <th class="text-right">Kho \u0110\u1ea7u (first_1A)</th>
                <th class="text-right">Kho Cu\u1ed1i (last_1A)</th>
                <th class="text-right">C\u1ea3 Hai Kho (Combined)</th>
                <th>\u00dd Ngh\u0129a</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Top-1 Accuracy</td>
                <td class="text-right font-medium">{f1_top1:.2f}%</td>
                <td class="text-right font-medium">{last_top1:.2f}%</td>
                <td class="text-right font-medium">{both_correct_acc:.2f}%</td>
                <td>T\u1ef7 l\u1ec7 d\u1ef1 \u0111o\u00e1n ch\u00ednh x\u00e1c tuy\u1ec7t \u0111\u1ed1i. Combined = c\u1ea3 2 kho \u0111\u1ec1u \u0111\u00fang.</td>
              </tr>
              <tr>
                <td>Top-3 Accuracy</td>
                <td class="text-right font-medium">{f1_top3:.2f}%</td>
                <td class="text-right font-medium">{last_top3:.2f}%</td>
                <td class="text-right font-medium">{both_top3_acc:.2f}%</td>
                <td>T\u1ef7 l\u1ec7 kho th\u1ef1c t\u1ebf n\u1eb1m trong top 3 g\u1ee3i \u00fd. Combined = c\u1ea3 2 kho \u0111\u1ec1u \u0111\u1ea1t Top-3.</td>
              </tr>
              <tr>
                <td>Weighted F1-Score</td>
                <td class="text-right font-medium">{f1_weighted_f1:.4f}</td>
                <td class="text-right font-medium">{last_weighted_f1:.4f}</td>
                <td class="text-right font-medium">{combined_weighted_f1:.4f}</td>
                <td>F1-Score trung b\u00ecnh theo tr\u1ecdng s\u1ed1 m\u1eabu. Combined = trung b\u00ecnh hai giai \u0111o\u1ea1n.</td>
              </tr>
              <tr>
                <td>Macro F1-Score</td>
                <td class="text-right font-medium">{f1_macro_f1:.4f}</td>
                <td class="text-right font-medium">{last_macro_f1:.4f}</td>
                <td class="text-right font-medium">{combined_macro_f1:.4f}</td>
                <td>F1-Score trung b\u00ecnh b\u00ecnh \u0111\u1eb3ng cho m\u1ed7i kho. Combined = trung b\u00ecnh hai giai \u0111o\u1ea1n.</td>
              </tr>
              <tr>
                <td>S\u1ed1 l\u01b0\u1ee3ng m\u1eabu ki\u1ec3m th\u1eed (N bills)</td>
                <td class="text-right font-medium">{n_samples:,}</td>
                <td class="text-right font-medium">{n_samples:,}</td>
                <td class="text-right font-medium">{n_samples:,}</td>
                <td>T\u1ed5ng s\u1ed1 l\u01b0\u1ee3ng v\u1eadn \u0111\u01a1n tham gia t\u1eadp \u0111\u00e1nh gi\u00e1 \u0111\u1ed9c l\u1eadp.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card-box">
        <div class="card-title">1.2 Top 10 C\u1eb7p L\u1ed7i Nh\u1ea7m Kho Ph\u1ed5 Bi\u1ebfn Nh\u1ea5t</div>
        <div class="card-desc">
          Ph\u00e2n t\u00edch c\u00e1c c\u1eb7p (Kho th\u1ef1c t\u1ebf, Kho b\u1ecb nh\u1ea7m) x\u1ea3y ra nhi\u1ec1u nh\u1ea5t. Gi\u00fap x\u00e1c \u0111\u1ecbnh c\u00e1c kho n\u00e0o th\u01b0\u1eddng b\u1ecb nh\u1ea7m l\u1eabn v\u1edbi nhau
          \u0111\u1ec3 c\u1ea3i thi\u1ec7n d\u1eef li\u1ec7u \u0111\u1eb7c tr\u01b0ng ho\u1eb7c b\u1ed5 sung quy t\u1eafc h\u1eadu x\u1eed l\u00fd.
        </div>
        <div class="grid-2">
          <div>{first_mis_html}</div>
          <div>{last_mis_html}</div>
        </div>
      </div>
    </section>

    <section id="section-p2" class="section-block">
      <h2 class="section-title">PH\u1ea6N 2: \u0110\u00c1NH GI\u00c1 M\u00d4 H\u00ccNH H\u1ed2I QUY TH\u1edcI GIAN (ARRIVAL REGRESSOR)</h2>

      <div class="card-box">
        <div class="card-title">2.1 B\u1ea3ng Thang \u0110o Chi Ti\u1ebft M\u00f4 H\u00ecnh H\u1ed3i Quy</div>
        <div class="card-desc">To\u00e0n b\u1ed9 c\u00e1c ch\u1ec9 s\u1ed1 \u0111\u00e1nh gi\u00e1 ch\u1ea5t l\u01b0\u1ee3ng d\u1ef1 \u0111o\u00e1n th\u1eddi gian t\u1edbi kho 1A tr\u00ean t\u1eadp test \u0111\u1ed9c l\u1eadp ({n_samples:,} \u0111\u01a1n h\u00e0ng).</div>
        <div class="table-container">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Ch\u1ec9 S\u1ed1</th>
                <th class="text-right">Gi\u00e1 Tr\u1ecb</th>
                <th>\u0110\u01a1n v\u1ecb</th>
                <th>\u00dd Ngh\u0129a Chi Ti\u1ebft</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Sai s\u1ed1 trung b\u00ecnh tuy\u1ec7t \u0111\u1ed1i (MAE)</td>
                <td class="text-right font-medium">{mae:.3f}</td>
                <td>gi\u1edd</td>
                <td>Trung b\u00ecnh kho\u1ea3ng c\u00e1ch tuy\u1ec7t \u0111\u1ed1i gi\u1eefa gi\u1edd d\u1ef1 \u0111o\u00e1n v\u00e0 gi\u1edd th\u1ef1c t\u1ebf. V\u00ed d\u1ee5: MAE = {mae:.2f} ngh\u0129a l\u00e0 trung b\u00ecnh m\u1ed7i \u0111\u01a1n h\u00e0ng b\u1ecb l\u1ec7ch {mae:.2f} gi\u1edd so v\u1edbi th\u1ef1c t\u1ebf.</td>
              </tr>
              <tr>
                <td>Trung v\u1ecb sai s\u1ed1 tuy\u1ec7t \u0111\u1ed1i (MedAE)</td>
                <td class="text-right font-medium">{medae:.3f}</td>
                <td>gi\u1edd</td>
                <td>Trung v\u1ecb sai s\u1ed1 (kh\u00f4ng b\u1ecb outlier \u1ea3nh h\u01b0\u1edfng). 50% \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 d\u01b0\u1edbi {medae:.2f} gi\u1edd. Th\u1ec3 hi\u1ec7n sai s\u1ed1 c\u1ee7a \u0111\u01a1n h\u00e0ng \u0111i\u1ec3n h\u00ecnh.</td>
              </tr>
              <tr>
                <td>Sai s\u1ed1 b\u00ecnh ph\u01b0\u01a1ng trung b\u00ecnh (RMSE)</td>
                <td class="text-right font-medium">{rmse:.3f}</td>
                <td>gi\u1edd</td>
                <td>Ph\u1ea1t n\u1eb7ng c\u00e1c \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 l\u1edbn. RMSE cao h\u01a1n MAE cho th\u1ea5y t\u1ed3n t\u1ea1i m\u1ed9t s\u1ed1 \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 \u0111\u1eb7c bi\u1ec7t l\u1edbn (outlier).</td>
              </tr>
              <tr>
                <td>Sai s\u1ed1 ph\u1ea7n tr\u0103m trung v\u1ecb (MdAPE)<sup>[2]</sup></td>
                <td class="text-right font-medium">{mdape:.2f}</td>
                <td>%</td>
                <td>Trung v\u1ecb sai s\u1ed1 ph\u1ea7n tr\u0103m. 50% \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 t\u01b0\u01a1ng \u0111\u1ed1i &le; {mdape:.1f}%. Th\u01b0\u1edbc \u0111o chu\u1ea9n x\u00e1c kh\u00f4ng b\u1ecb m\u00e9o b\u1edfi c\u00e1c \u0111\u01a1n g\u1ea7n kho (&lt; 1h).</td>
              </tr>
              <tr>
                <td>Sai s\u1ed1 ph\u1ea7n tr\u0103m trung b\u00ecnh (MAPE)<sup>[2]</sup></td>
                <td class="text-right font-medium">{mape:.2f}</td>
                <td>%</td>
                <td>T\u1ef7 l\u1ec7 ph\u1ea7n tr\u0103m sai l\u1ec7ch trung b\u00ecnh so v\u1edbi th\u1ef1c t\u1ebf. L\u01b0u \u00fd: MAPE b\u1ecb k\u00e9o l\u00ean cao b\u1edfi c\u00e1c \u0111\u01a1n g\u1ea7n kho (&lt; 1h chi\u1ebfm 12.7% m\u1eabu, m\u1eabu s\u1ed1 nh\u1ecf l\u00e0m % v\u1ecdt cao); n\u00ean tham chi\u1ebfu c\u00f9ng MdAPE.</td>
              </tr>
              <tr>
                <td>H\u1ec7 s\u1ed1 x\u00e1c \u0111\u1ecbnh (R\u00b2 Score)</td>
                <td class="text-right font-medium">{r2:.4f}</td>
                <td>-</td>
                <td>M\u1ee9c \u0111\u1ed9 gi\u1ea3i th\u00edch ph\u01b0\u01a1ng sai th\u1eddi gian c\u1ee7a m\u00f4 h\u00ecnh. R\u00b2 = {r2:.3f} ngh\u0129a l\u00e0 m\u00f4 h\u00ecnh gi\u1ea3i th\u00edch \u0111\u01b0\u1ee3c {r2*100:.1f}% bi\u1ebfn \u0111\u1ed9ng th\u1eddi gian so v\u1edbi d\u1ef1 \u0111o\u00e1n b\u1eb1ng trung b\u00ecnh.</td>
              </tr>
              <tr>
                <td>\u0110\u1ed9 l\u1ec7ch h\u1ec7 th\u1ed1ng (Mean Bias)</td>
                <td class="text-right font-medium">{mean_bias:+.3f}</td>
                <td>gi\u1edd</td>
                <td>{'M\u00f4 h\u00ecnh c\u00f3 xu h\u01b0\u1edbng d\u1ef1 \u0111o\u00e1n h\u01a1i mu\u1ed9n h\u01a1n th\u1ef1c t\u1ebf (over-estimate)' if mean_bias > 0 else 'M\u00f4 h\u00ecnh c\u00f3 xu h\u01b0\u1edbng d\u1ef1 \u0111o\u00e1n h\u01a1i s\u1edbm h\u01a1n th\u1ef1c t\u1ebf (under-estimate)'}. Gi\u00e1 tr\u1ecb g\u1ea7n 0 = kh\u00f4ng l\u1ec7ch h\u1ec7 th\u1ed1ng.</td>
              </tr>
              <tr>
                <td>Trong dung sai \u00b11.0 gi\u1edd<sup>[1]</sup></td>
                <td class="text-right font-medium">{within_1h:.1f}%</td>
                <td>%</td>
                <td>T\u1ef7 l\u1ec7 \u0111\u01a1n h\u00e0ng d\u1ef1 \u0111o\u00e1n ch\u00ednh x\u00e1c trong kho\u1ea3ng 1 gi\u1edd so v\u1edbi th\u1ef1c t\u1ebf.</td>
              </tr>
              <tr>
                <td>Trong dung sai \u00b12.0 gi\u1edd<sup>[1]</sup></td>
                <td class="text-right font-medium">{within_2h:.1f}%</td>
                <td>%</td>
                <td>T\u1ef7 l\u1ec7 \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 \u2264 2 gi\u1edd. Ng\u01b0\u1ee1ng ph\u1ed5 bi\u1ebfn cho v\u1eadn h\u00e0nh kho ti\u1ebfp nh\u1eadn trung chuy\u1ec3n.</td>
              </tr>
              <tr>
                <td>Trong dung sai \u00b13.0 gi\u1edd<sup>[1]</sup></td>
                <td class="text-right font-medium">{within_3h:.1f}%</td>
                <td>%</td>
                <td>T\u1ef7 l\u1ec7 \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 \u2264 3 gi\u1edd. Ng\u01b0\u1ee1ng dung sai r\u1ed9ng h\u01a1n cho c\u00e1c ch\u1eb7ng \u0111\u01b0\u1eddng d\u00e0i.</td>
              </tr>
              <tr>
                <td>Trong dung sai \u00b15.0 gi\u1edd<sup>[1]</sup></td>
                <td class="text-right font-medium">{within_5h:.1f}%</td>
                <td>%</td>
                <td>\u0110\u1ed9 bao ph\u1ee7 ph\u1ea7n l\u1edbn m\u1ea1ng l\u01b0\u1edbi v\u1eadn h\u00e0nh ({within_5h:.1f}% t\u1ed5ng s\u1ed1 \u0111\u01a1n).</td>
              </tr>
              <tr>
                <td>P90 Error (Ph\u00e2n v\u1ecb 90%)</td>
                <td class="text-right font-medium">{p90:.2f}</td>
                <td>gi\u1edd</td>
                <td>90% \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 kh\u00f4ng v\u01b0\u1ee3t qu\u00e1 {p90:.2f} gi\u1edd. 10% c\u00f2n l\u1ea1i l\u00e0 c\u00e1c tr\u01b0\u1eddng h\u1ee3p kh\u00f3 (\u0111\u01a1n gom, qua \u0111\u00eam...).</td>
              </tr>
              <tr>
                <td>P95 Error (Ph\u00e2n v\u1ecb 95%)</td>
                <td class="text-right font-medium">{p95:.2f}</td>
                <td>gi\u1edd</td>
                <td>95% \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 kh\u00f4ng v\u01b0\u1ee3t qu\u00e1 {p95:.2f} gi\u1edd.</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="footnote">
          <sup>[1]</sup> <b>Dung sai \u00b1Xh:</b> Sai s\u1ed1 tuy\u1ec7t \u0111\u1ed1i |th\u1eddi gian d\u1ef1 \u0111o\u00e1n - th\u1eddi gian th\u1ef1c t\u1ebf| \u2264 X gi\u1edd. \u0110\u00e2y l\u00e0 c\u00e1ch \u0111o l\u01b0\u1eddng th\u1ef1c t\u1ebf nh\u1ea5t cho v\u1eadn h\u00e0nh logistics:
          n\u1ebfu m\u00f4 h\u00ecnh b\u00e1o h\u00e0ng t\u1edbi kho l\u00fac 14:00 v\u00e0 th\u1ef1c t\u1ebf t\u1edbi l\u00fac 15:30, sai l\u1ec7ch 1.5h n\u1eb1m trong dung sai \u00b12h.
        </div>
      </div>

      <div class="grid-2">
        <div class="card-box">
          <div class="card-title">2.2 Bi\u1ec3u \u0110\u1ed3 T\u01b0\u01a1ng Quan: Th\u1ef1c T\u1ebf vs D\u1ef1 \u0110o\u00e1n</div>
          <div class="card-desc">
            M\u1ed7i \u0111i\u1ec3m tr\u00ean bi\u1ec3u \u0111\u1ed3 \u0111\u1ea1i di\u1ec7n cho m\u1ed9t \u0111\u01a1n h\u00e0ng: tr\u1ee5c ngang l\u00e0 th\u1eddi gian th\u1ef1c t\u1ebf t\u1edbi kho 1A, tr\u1ee5c d\u1ecdc l\u00e0 th\u1eddi gian m\u00f4 h\u00ecnh d\u1ef1 \u0111o\u00e1n.
            <b>\u0110\u01b0\u1eddng n\u00e9t \u0111\u1ee9t \u0111en (y = x)</b> l\u00e0 \u0111\u01b0\u1eddng l\u00fd t\u01b0\u1edfng - n\u1ebfu d\u1ef1 \u0111o\u00e1n ho\u00e0n h\u1ea3o, t\u1ea5t c\u1ea3 \u0111i\u1ec3m s\u1ebd n\u1eb1m tr\u00ean \u0111\u01b0\u1eddng n\u00e0y.
            <b>D\u1ea3i xanh nh\u1ea1t</b> bi\u1ec3u th\u1ecb v\u00f9ng dung sai \u00b12 gi\u1edd: \u0111i\u1ec3m n\u1eb1m trong d\u1ea3i n\u00e0y c\u00f3 sai s\u1ed1 ch\u1ea5p nh\u1eadn \u0111\u01b0\u1ee3c.
            \u0110i\u1ec3m c\u00e0ng ph\u00e2n t\u00e1n xa \u0111\u01b0\u1eddng y = x, sai s\u1ed1 d\u1ef1 \u0111o\u00e1n c\u00e0ng l\u1edbn.
          </div>
          {fig_divs['scatter']}
        </div>
        <div class="card-box">
          <div class="card-title">2.3 Ph\u00e2n Ph\u1ed1i Sai S\u1ed1 Th\u1eddi Gian (Error Distribution)</div>
          <div class="card-desc">
            Bi\u1ec3u \u0111\u1ed3 histogram th\u1ec3 hi\u1ec7n ph\u00e2n ph\u1ed1i sai s\u1ed1 (D\u1ef1 \u0111o\u00e1n - Th\u1ef1c t\u1ebf) tr\u00ean to\u00e0n b\u1ed9 t\u1eadp test.
            <b>Tr\u1ee5c ngang:</b> sai s\u1ed1 t\u00ednh b\u1eb1ng gi\u1edd (gi\u00e1 tr\u1ecb d\u01b0\u01a1ng = m\u00f4 h\u00ecnh d\u1ef1 \u0111o\u00e1n mu\u1ed9n h\u01a1n th\u1ef1c t\u1ebf, gi\u00e1 tr\u1ecb \u00e2m = d\u1ef1 \u0111o\u00e1n s\u1edbm h\u01a1n).
            <b>Tr\u1ee5c d\u1ecdc:</b> s\u1ed1 l\u01b0\u1ee3ng \u0111\u01a1n h\u00e0ng t\u1ea1i m\u1ed7i m\u1ee9c sai s\u1ed1.
            \u0110\u1ec9nh ph\u00e2n ph\u1ed1i h\u1ed9i t\u1ee5 s\u00e1t m\u1ed1c 0h cho th\u1ea5y ph\u1ea7n l\u1edbn d\u1ef1 \u0111o\u00e1n g\u1ea7n s\u00e1t th\u1ef1c t\u1ebf.
            N\u1ebfu ph\u00e2n ph\u1ed1i \u0111\u1ed1i x\u1ee9ng quanh 0 = m\u00f4 h\u00ecnh kh\u00f4ng b\u1ecb l\u1ec7ch h\u1ec7 th\u1ed1ng (kh\u00f4ng thi\u00ean v\u1ec1 d\u1ef1 \u0111o\u00e1n s\u1edbm hay mu\u1ed9n).
          </div>
          {fig_divs['hist']}
        </div>
      </div>

      <div class="grid-2">
        <div class="card-box">
          <div class="card-title">2.4 \u0110\u01b0\u1eddng Cong T\u00edch L\u0169y Sai S\u1ed1 (CDF Curve)</div>
          <div class="card-desc">
            \u0110\u01b0\u1eddng CDF cho bi\u1ebft t\u1ef7 l\u1ec7 ph\u1ea7n tr\u0103m \u0111\u01a1n h\u00e0ng t\u00edch l\u0169y \u1ee9ng v\u1edbi t\u1eebng m\u1ee9c ng\u01b0\u1ee1ng sai s\u1ed1 tuy\u1ec7t \u0111\u1ed1i.
            V\u00ed d\u1ee5: n\u1ebfu \u0111\u01b0\u1eddng c\u1eaft m\u1ed1c \u00b12h t\u1ea1i 60%, ngh\u0129a l\u00e0 60% \u0111\u01a1n h\u00e0ng c\u00f3 sai s\u1ed1 \u2264 2 gi\u1edd.
            C\u00e1c \u0111\u01b0\u1eddng \u0111\u1ee9t n\u00e9t d\u1ecdc \u0111\u00e1nh d\u1ea5u c\u00e1c m\u1ed1c dung sai \u00b11h, \u00b12h, \u00b13h, \u00b15h k\u00e8m t\u1ef7 l\u1ec7 t\u01b0\u01a1ng \u1ee9ng.
          </div>
          {fig_divs['cdf']}
        </div>
        <div class="card-box">
          <div class="card-title">2.5 Ph\u00e2n Ph\u1ed1i Sai S\u1ed1 Th\u1eddi Gian Theo Kho first_1A</div>
          <div class="card-desc">
            Box plot so s\u00e1nh m\u1ee9c \u0111\u1ed9 ph\u00e2n t\u00e1n sai s\u1ed1 gi\u1eefa c\u00e1c kho 1A l\u1edbn nh\u1ea5t tr\u00ean to\u00e0n qu\u1ed1c (top 14 kho c\u00f3 nhi\u1ec1u \u0111\u01a1n nh\u1ea5t).
            M\u1ed7i box cho bi\u1ebft: trung v\u1ecb sai s\u1ed1 (\u0111\u01b0\u1eddng ngang gi\u1eefa), kho\u1ea3ng 50% \u0111\u01a1n h\u00e0ng (h\u1ed9p), v\u00e0 ph\u1ea1m vi bi\u1ebfn \u0111\u1ed9ng (r\u00e2u).
            Kho n\u00e0o c\u00f3 box th\u1ea5p v\u00e0 h\u1eb9p = d\u1ef1 \u0111o\u00e1n \u1ed5n \u0111\u1ecbnh; box cao v\u00e0 r\u1ed9ng = sai s\u1ed1 dao \u0111\u1ed9ng l\u1edbn, c\u1ea7n theo d\u00f5i.
            C\u00e1c kho c\u00f3 \u00edt \u0111\u01a1n h\u00e0ng (support th\u1ea5p) kh\u00f4ng hi\u1ec3n th\u1ecb do m\u1eabu kh\u00f4ng \u0111\u1ee7 \u0111\u1ea1i di\u1ec7n.
          </div>
          {fig_divs['box_wh']}
        </div>
      </div>

      <div class="card-box">
        <div class="card-title">2.6 Ph\u00e2n Ph\u1ed1i Sai S\u1ed1 Th\u1eddi Gian Theo Khung Gi\u1edd T\u1ea1o Bill</div>
        <div class="card-desc">
          \u0110\u00e1nh gi\u00e1 \u0111\u1ed9 \u1ed5n \u0111\u1ecbnh d\u1ef1 \u0111o\u00e1n qua 4 khung gi\u1edd trong ng\u00e0y: \u0110\u00eam (00-06h), S\u00e1ng (06-12h), Chi\u1ec1u (12-18h), T\u1ed1i (18-24h).<br>
          <b>L\u01b0u \u00fd nghi\u1ec7p v\u1ee5:</b> Khung 00h-06h c\u00f3 th\u1eddi gian v\u00e0 sai s\u1ed1 cao do ph\u00e1t sinh th\u1eddi gian ch\u1edd b\u01b0u c\u1ee5c m\u1edf c\u1eeda ca s\u00e1ng (Dormant Window) v\u00e0 r\u1ee7i ro tr\u01b0\u1ee3t chu k\u1ef3 xe gom (+24h) t\u1eeb ph\u00eda ng\u01b0\u1eddi g\u1eedi. Khung t\u1ed1i (18-24h) \u0111\u1ea1t \u0111\u1ed9 ch\u00ednh x\u00e1c cao nh\u1ea5t (MedAE ch\u1ec9 0.67h, 74.9% trong dung sai &plusmn;2h) do h\u00e0ng gom s\u1eb5n ch\u1ea1y th\u1eb3ng tr\u00ean cao t\u1ed1c \u0111\u00eam.</div>
        {fig_divs['box_hr']}
      </div>
    </section>

    <section id="section-p3" class="section-block">
      <h2 class="section-title">PH\u1ea6N 3: \u0110\u00c1NH GI\u00c1 T\u01af\u01a0NG T\u00c1C K\u1ebeT H\u1ee2P HAI GIAI \u0110O\u1ea0N (PIPELINE INTERACTION)</h2>

      <div class="card-box">
        <div class="card-title">3.1 Bi\u1ec3u \u0110\u1ed3 Chu\u1ed7i Th\u1eddi Gian: Th\u1ef1c T\u1ebf vs D\u1ef1 \u0110o\u00e1n (2025 - 06/2026)</div>
        <div class="card-desc">Quan s\u00e1t nh\u1ecbp bi\u1ebfn \u0111\u1ed9ng th\u1eddi gian t\u1edbi kho 1A li\u00ean t\u1ee5c theo th\u1eddi \u0111i\u1ec3m t\u1ea1o bill. H\u1ed7 tr\u1ee3 zoom v\u00e0 r\u00ea chu\u1ed9t tra c\u1ee9u.</div>
        {fig_divs['ts']}
      </div>

      <div class="card-box">
        <div class="card-title">3.2 Ph\u00e2n T\u00edch Lan Truy\u1ec1n Sai S\u1ed1: T\u00e1c \u0110\u1ed9ng Khi Classifier G\u00e1n Sai Kho</div>
        <div class="card-desc">
          \u0110o l\u01b0\u1eddng \u0111\u1ecbnh l\u01b0\u1ee3ng m\u1ee9c \u0111\u1ed9 \u1ea3nh h\u01b0\u1edfng c\u1ee7a l\u1ed7i ph\u00e2n lo\u1ea1i kho \u0111\u1ea7u v\u00e0o l\u00ean sai s\u1ed1 c\u1ee7a t\u1ea7ng h\u1ed3i quy th\u1eddi gian.
          B\u1ea3ng chia \u0111\u01a1n h\u00e0ng th\u00e0nh 2 nh\u00f3m: nh\u00f3m Classifier g\u00e1n \u0110\u00daNG kho 1A v\u00e0 nh\u00f3m g\u00e1n SAI, sau \u0111\u00f3 so s\u00e1nh sai s\u1ed1 regressor \u1edf m\u1ed7i nh\u00f3m.
        </div>
        
        <div class="table-container">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Tr\u01b0\u1eddng H\u1ee3p Ph\u00e2n Lo\u1ea1i Kho first_1A</th>
                <th class="text-right">S\u1ed1 L\u01b0\u1ee3ng \u0110\u01a1n</th>
                <th class="text-right">T\u1ef7 Tr\u1ecdng (%)</th>
                <th class="text-right">MAE (Gi\u1edd)</th>
                <th class="text-right">MedAE (Gi\u1edd)</th>
                <th class="text-right">RMSE (Gi\u1edd)</th>
                <th class="text-right">Trong Dung Sai \u00b12h<sup>[1]</sup> (%)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Classifier G\u00e1n \u0110\u00daNG Kho 1A</td>
                <td class="text-right font-medium">{correct_stats['count']:,}</td>
                <td class="text-right">{correct_stats['pct']:.1f}%</td>
                <td class="text-right font-medium">{correct_stats['mae']:.2f} h</td>
                <td class="text-right font-medium">{correct_stats['medae']:.2f} h</td>
                <td class="text-right font-medium">{correct_stats['rmse']:.2f} h</td>
                <td class="text-right font-medium">{correct_stats['within_2h']:.1f}%</td>
              </tr>
              <tr>
                <td>Classifier G\u00e1n SAI Kho 1A</td>
                <td class="text-right font-medium">{wrong_stats['count']:,}</td>
                <td class="text-right">{wrong_stats['pct']:.1f}%</td>
                <td class="text-right font-medium">{wrong_stats['mae']:.2f} h</td>
                <td class="text-right font-medium">{wrong_stats['medae']:.2f} h</td>
                <td class="text-right font-medium">{wrong_stats['rmse']:.2f} h</td>
                <td class="text-right font-medium">{wrong_stats['within_2h']:.1f}%</td>
              </tr>
              <tr style="background-color: #f8fafc; font-weight: 600;">
                <td>M\u1ee9c Ch\u00eanh L\u1ec7ch (Delta)</td>
                <td class="text-right">-</td>
                <td class="text-right">-</td>
                <td class="text-right">{delta_stats['mae']:+.2f} h</td>
                <td class="text-right">{delta_stats['medae']:+.2f} h</td>
                <td class="text-right">{delta_stats['rmse']:+.2f} h</td>
                <td class="text-right">{delta_stats['within_2h']:+.1f}%</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="alert-box">
          <strong>Nh\u1eadn \u0110\u1ecbnh K\u1ef9 Thu\u1eadt Nghi\u1ec7p V\u1ee5:</strong><br>
          Khi Classifier d\u1ef1 \u0111o\u00e1n ch\u00ednh x\u00e1c tr\u1ea1m 1A, sai s\u1ed1 MAE \u0111\u1ea1t m\u1ee9c t\u1ed1i \u01b0u <b>{correct_stats['mae']:.2f} gi\u1edd</b> ({correct_stats['within_2h']:.1f}% s\u1ed1 \u0111\u01a1n n\u1eb1m trong dung sai \u00b12h). 
          Khi tr\u1ea1m 1A b\u1ecb g\u00e1n sai, sai s\u1ed1 MAE t\u0103ng th\u00eam <b>{abs(delta_stats['mae']):.2f} gi\u1edd</b> (l\u00ean m\u1ee9c {wrong_stats['mae']:.2f} gi\u1edd). 
          K\u1ebft qu\u1ea3 n\u00e0y \u0111\u1ecbnh l\u01b0\u1ee3ng \u0111\u01b0\u1ee3c t\u1ea7m quan tr\u1ecdng s\u1ed1ng c\u00f2n c\u1ee7a t\u1ea7ng ph\u00e2n lo\u1ea1i: C\u1ea3i thi\u1ec7n \u0111\u1ed9 ch\u00ednh x\u00e1c \u1edf Classifier l\u00e0 gi\u1ea3i ph\u00e1p c\u1ed1t l\u00f5i gi\u00fap h\u1ea1 th\u1ea5p sai s\u1ed1 th\u1eddi gian to\u00e0n h\u1ec7 th\u1ed1ng.
        </div>
        <div class="footnote">
          <sup>[1]</sup> Dung sai \u00b12h: T\u1ef7 l\u1ec7 \u0111\u01a1n h\u00e0ng c\u00f3 |sai s\u1ed1 d\u1ef1 \u0111o\u00e1n| \u2264 2 gi\u1edd so v\u1edbi th\u1eddi gian th\u1ef1c t\u1ebf t\u1edbi kho 1A.
        </div>
      </div>
    </section>

    <section id="section-p4" class="section-block">
      <h2 class="section-title">PH\u1ea6N 4: PH\u00c2N T\u00cdCH CHUY\u00caN S\u00c2U (DEEP DIVE INSIGHTS)</h2>

      <div class="card-box">
        <div class="card-title">4.1 Top 10 Tuy\u1ebfn V\u1eadn Chuy\u1ec3n C\u00f3 Sai S\u1ed1 Th\u1eddi Gian L\u1edbn Nh\u1ea5t</div>
        <div class="card-desc">Danh s\u00e1ch c\u00e1c tuy\u1ebfn (l\u1ecdc c\u00e1c tuy\u1ebfn c\u00f3 t\u1ed1i thi\u1ec3u 15 \u0111\u01a1n h\u00e0ng) c\u00f3 MAE cao nh\u1ea5t c\u1ea7n theo d\u00f5i \u0111\u1eb7c th\u00f9.</div>
        {route_table_html}
      </div>

      <div class="card-box">
        <div class="card-title">4.2 Ph\u00e2n Ph\u1ed1i Sai S\u1ed1 Th\u1eddi Gian Theo C\u00e1c Ng\u00e0y Trong Tu\u1ea7n</div>
        <div class="card-desc">\u0110\u00e1nh gi\u00e1 t\u00ednh \u1ed5n \u0111\u1ecbnh c\u1ee7a m\u00f4 h\u00ecnh gi\u1eefa c\u00e1c ng\u00e0y trong tu\u1ea7n (Th\u1ee9 Hai - Th\u1ee9 S\u00e1u) so v\u1edbi c\u00e1c ng\u00e0y cu\u1ed1i tu\u1ea7n (Th\u1ee9 B\u1ea3y, Ch\u1ee7 Nh\u1eadt).</div>
        {fig_divs['box_dow']}
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- PHẦN 5: ĐÁNH GIÁ THEO CA VẬN HÀNH & ĐIỂM CHỐT SỐ (SHIFT-LEVEL EVALUATION) -->
    <!-- ================================================================= -->
    <section class="section">
      <div class="section-header">
        <div class="section-tag">Vận Hành Thực Tế</div>
        <div class="section-title">PHẦN 5: ĐÁNH GIÁ THEO CA VẬN HÀNH & LÁT CẮT CHỐT SỐ (SHIFT-LEVEL ANALYSIS)</div>
      </div>

      <div class="card-box">
        <div class="card-title">5.1 Bảng Đánh Giá Độ Chính Xác Theo Ca Vận Hành (Shift-Level Matrix)</div>
        <div class="card-desc">Chuyển đổi sai số thời gian liên tục sang bài toán phân loại ca làm việc: Ca Đêm (22h - 06h), Ca Sáng (06h - 14h), Ca Chiều (14h - 22h).</div>
        {render_shift_table(shift_metrics)}
      </div>

      <div class="card-box">
        <div class="card-title">5.2 Đánh Giá So Sánh: Toàn Bộ Snapshot vs Snapshot Chốt Số Gần Nhất (Unique Bill Cut-off)</div>
        <div class="card-desc">So sánh sai số giữa toàn bộ quan sát lịch sử và bản ghi cập nhật mới nhất trước khi đơn hàng về kho.</div>
        {render_cutoff_comparison(cutoff_comparison)}
      </div>
    </section>

  </main>

  <footer style="text-align: center; padding: 24px; color: var(--text-muted); font-size: 12px; border-top: 1px solid var(--border); background: #ffffff;">
    B\u00e1o C\u00e1o \u0110\u00e1nh Gi\u00e1 Hi\u1ec7u N\u0103ng M\u00f4 H\u00ecnh End-to-End Pipeline &bull; H\u1ec7 Th\u1ed1ng T\u1ed1i \u01afu H\u00f3a Logistics
  </footer>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [build_html_report] Da luu bao cao HTML thanh cong: {output_path}")


# ==============================================================================
# 4. HAM MAIN
# ==============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DANH GIA TOAN DIEN PIPELINE END-TO-END")
    print(f"  Model  : {args.model_path}")
    print(f"  Data   : {args.data_dir}")
    print(f"  Output : {args.output_dir}")
    print(f"  Samples: {args.max_eval_samples:,}")
    print("=" * 70)

    print("\n[1/4] Nap mo hinh Pipeline...")
    pipeline = EndToEndPipeline.load(args.model_path)

    print("\n[2/4] Nap du lieu & chuan bi tap kiem thu doc lap (Test Set)...")
    wh1a = pd.read_csv(os.path.join(args.data_dir, "warehouse_1A.csv"))
    bill = pd.read_csv(os.path.join(args.data_dir, "bill.csv"))

    sche_chunks, total = [], 0
    for chunk in pd.read_csv(os.path.join(args.data_dir, "bill_schedule.csv"), chunksize=100000):
        chunk["io_time"] = pd.to_datetime(chunk["io_time"], errors="coerce")
        sche_chunks.append(chunk)
        total += len(chunk)
        if total >= args.chunksize:
            break
    sche = pd.concat(sche_chunks, ignore_index=True)
    del sche_chunks
    print(f"  -> Da nap {len(sche):,} ban ghi schedule, {len(bill):,} bills.")

    e2e_data = build_e2e_training_data(sche, bill, wh1a)
    del sche

    _, _, test_df, _, _ = prepare_e2e_splits(e2e_data, train_split=0.6, val_split=0.2)

    target_col = "time_to_first_1a_hours" if "time_to_first_1a_hours" in test_df.columns else "arrival_offset_hours"
    eval_df = test_df.dropna(subset=[target_col]).copy()
    if len(eval_df) > args.max_eval_samples:
        eval_df = eval_df.sample(args.max_eval_samples, random_state=42)

    print(f"\n[3/4] Thuc hien du bao Pipeline tren {len(eval_df):,} mau test...")
    preds = pipeline.predict(eval_df)

    y_true_hours = eval_df[target_col].values
    pred_col = "hours_to_first_1A" if "hours_to_first_1A" in preds.columns else "predicted_hours_to_first_1A"
    y_pred_hours = preds[pred_col].values

    y_true_first = eval_df["first_1A"].astype(str).values if "first_1A" in eval_df.columns else None
    y_pred_first = preds["first_1A"].astype(str).values if "first_1A" in preds.columns else None
    top3_first = preds["top3_first_1A"].values if "top3_first_1A" in preds.columns else None

    y_true_last = eval_df["last_1A"].astype(str).values if "last_1A" in eval_df.columns else None
    y_pred_last = preds["last_1A"].astype(str).values if "last_1A" in preds.columns else None
    top3_last = preds["top3_last_1A"].values if "top3_last_1A" in preds.columns else None

    bill_dates = eval_df["bill_creation_date"].values if "bill_creation_date" in eval_df.columns else None

    if "province_route" in eval_df.columns:
        routes = eval_df["province_route"].fillna("UNKNOWN").astype(str).values
    elif "origin_province" in eval_df.columns and "destination_province" in eval_df.columns:
        routes = (eval_df["origin_province"].fillna("UNKNOWN").astype(str) + " -> " + eval_df["destination_province"].fillna("UNKNOWN").astype(str)).values
    else:
        routes = None

    print("\n[4/4] Khoi tao Bao Cao HTML Danh Gia...")
    output_path = os.path.join(args.output_dir, "evaluation_report.html")
    # Tinh toan Shift Metrics & Cut-off Comparison
    print("  -> Tinh toan chi so ca van hanh (Shift-Level & Cut-off Comparison)...")
    if "arrival_time" in eval_df.columns and "snapshot_time" in eval_df.columns:
        act_dt = pd.to_datetime(eval_df["arrival_time"])
        pred_col_time = "predicted_arrival_time" if "predicted_arrival_time" in preds.columns else None
        if pred_col_time:
            pred_dt = pd.to_datetime(preds[pred_col_time])
        else:
            pred_dt = pd.to_datetime(eval_df["snapshot_time"]) + pd.to_timedelta(y_pred_hours, unit="h")

        def map_shift(dt_series):
            h = dt_series.dt.hour
            return np.where(h.isin([22, 23, 0, 1, 2, 3, 4, 5]), "Ca Đêm (22h - 06h)",
                   np.where(h.isin([6, 7, 8, 9, 10, 11, 12, 13]), "Ca Sáng (06h - 14h)", "Ca Chiều (14h - 22h)"))

        act_shifts = map_shift(act_dt)
        pred_shifts = map_shift(pred_dt)

        shift_acc = float((act_shifts == pred_shifts).mean() * 100)
        shift_list = ["Ca Đêm (22h - 06h)", "Ca Sáng (06h - 14h)", "Ca Chiều (14h - 22h)"]
        shift_rows = []
        for s in shift_list:
            mask_a = (act_shifts == s)
            mask_p = (pred_shifts == s)
            n_a = int(mask_a.sum())
            n_p = int(mask_p.sum())
            n_i = int((mask_a & mask_p).sum())
            rec = float(n_i / n_a * 100) if n_a > 0 else 0.0
            prec = float(n_i / n_p * 100) if n_p > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            shift_rows.append({
                "name": s,
                "n_actual": n_a,
                "n_pred": n_p,
                "n_inter": n_i,
                "recall": rec,
                "precision": prec,
                "f1": f1,
            })
        shift_metrics = {"shift_acc": shift_acc, "rows": shift_rows}
    else:
        shift_metrics = None

    if "bill_code" in eval_df.columns and "snapshot_time" in eval_df.columns:
        eval_df_sorted = eval_df.sort_values("snapshot_time")
        unique_latest = eval_df_sorted.drop_duplicates("bill_code", keep="last")
        u_y_true = unique_latest[target_col].values
        
        # pred for unique latest
        u_y_pred = preds.loc[unique_latest.index, pred_col].values if pred_col in preds.columns else y_pred_hours[unique_latest.index]
        u_abs_err = np.abs(u_y_true - u_y_pred)
        abs_err_all = np.abs(y_true_hours - y_pred_hours)
        
        cutoff_comparison = {
            "all_mae": float(mean_absolute_error(y_true_hours, y_pred_hours)),
            "all_medae": float(median_absolute_error(y_true_hours, y_pred_hours)),
            "all_within_2h": float(np.mean(abs_err_all <= 2.0) * 100),
            "all_within_3h": float(np.mean(abs_err_all <= 3.0) * 100),
            "unique_mae": float(mean_absolute_error(u_y_true, u_y_pred)),
            "unique_medae": float(median_absolute_error(u_y_true, u_y_pred)),
            "unique_within_2h": float(np.mean(u_abs_err <= 2.0) * 100),
            "unique_within_3h": float(np.mean(u_abs_err <= 3.0) * 100),
            "n_unique_bills": int(len(unique_latest)),
            "n_total_snapshots": int(len(eval_df)),
        }
    else:
        cutoff_comparison = None

    build_html_report(
        y_true_hours=y_true_hours,
        y_pred_hours=y_pred_hours,
        bill_dates=bill_dates,
        y_true_first=y_true_first,
        y_pred_first=y_pred_first,
        top3_first=top3_first,
        y_true_last=y_true_last,
        y_pred_last=y_pred_last,
        top3_last=top3_last,
        routes=routes,
        output_path=output_path,
        shift_metrics=shift_metrics,
        cutoff_comparison=cutoff_comparison,
    )

    print("\n" + "=" * 70)
    print("HOAN TAT DANH GIA! Moi mo file de xem bao cao:")
    print(f"  --> {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
