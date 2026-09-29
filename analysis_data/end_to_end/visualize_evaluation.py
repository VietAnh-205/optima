import os
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=os.path.join("saved_e2e_models", "e2e_pipeline.pkl"))
    parser.add_argument("--output-dir", type=str, default=os.path.join("saved_e2e_models", "eval_plots"))
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--max-eval-samples", type=int, default=20000)
    parser.add_argument("--chunksize", type=int, default=500000)
    return parser.parse_args()


def build_html_report(y_true, y_pred, bill_dates, output_path):
    errors = y_pred - y_true
    abs_err = np.abs(errors)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean(errors ** 2))
    medae = float(np.median(abs_err))
    r2 = r2_score(y_true, y_pred)
    within_1h = np.mean(abs_err <= 1) * 100
    within_2h = np.mean(abs_err <= 2) * 100
    within_3h = np.mean(abs_err <= 3) * 100

    # ---- Chart 1: Time Series liên tục theo bill_creation_date (2025 - 06/2026) ----
    dates = pd.to_datetime(bill_dates, errors="coerce")
    tmp = pd.DataFrame({"date": dates, "actual": y_true, "predicted": y_pred}).dropna(subset=["date"])
    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2026-06-30 23:59:59")
    tmp = tmp[(tmp["date"] >= start_date) & (tmp["date"] <= end_date)].sort_values("date").reset_index(drop=True)

    # Khung hiển thị mặc định = 25%-75% của dải ngày (vd: ngày 1->10 thì show 2.5->7.5)
    span = tmp["date"].max() - tmp["date"].min()
    view_start = tmp["date"].min() + span * 0.01
    view_end = tmp["date"].min() + span * 0.99

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=tmp["date"], y=tmp["actual"],
        mode="lines", name="Actual",
        line=dict(width=1.5), connectgaps=True,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual: %{y:.2f}h<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=tmp["date"], y=tmp["predicted"],
        mode="lines", name="Predicted",
        line=dict(width=1.5, dash="dash"), connectgaps=True,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Predicted: %{y:.2f}h<extra></extra>",
    ))
    fig1.update_layout(
        title="Time Series: Thời gian tới kho 1A theo bill",
        xaxis_title="Thời điểm tạo bill", yaxis_title="Giờ tới kho 1A",
        xaxis_range=[view_start, view_end],
        hovermode="x unified",
        height=800, width=1400,
    )

    # ---- Chart 2: CDF of Absolute Error (giữ nguyên) ----
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sorted_err, y=cdf, mode="lines", name="CDF",
        hovertemplate="Error: %{x:.2f}h<br>Cumulative: %{y:.1f}%<extra></extra>",
    ))
    for th in [1, 2, 3, 5]:
        pct = np.mean(abs_err <= th) * 100
        fig2.add_vline(x=th, line_dash="dot", line_color="gray",
                       annotation_text=f"{pct:.1f}%")
    fig2.update_layout(
        title="CDF of Absolute Error",
        xaxis_title="Absolute Error (h)", yaxis_title="Cumulative %",
        xaxis_range=[0, min(16, float(np.percentile(sorted_err, 98)))],
        height=450,
    )

    # ---- Ghép HTML ----
    metrics_html = f"""
    <h1>Evaluation Report</h1>
    <table border="1" cellpadding="6" cellspacing="0"
           style="border-collapse:collapse;font-family:Arial;">
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>N samples</td><td>{len(y_true):,}</td></tr>
      <tr><td>MAE</td><td>{mae:.3f} h</td></tr>
      <tr><td>RMSE</td><td>{rmse:.3f} h</td></tr>
      <tr><td>Median AE</td><td>{medae:.3f} h</td></tr>
      <tr><td>R²</td><td>{r2:.4f}</td></tr>
      <tr><td>Within ±1h</td><td>{within_1h:.1f}%</td></tr>
      <tr><td>Within ±2h</td><td>{within_2h:.1f}%</td></tr>
      <tr><td>Within ±3h</td><td>{within_3h:.1f}%</td></tr>
    </table>
    """

    chart_divs = ""
    for i, fig in enumerate([fig1, fig2], 1):
        chart_divs += fig.to_html(full_html=False, include_plotlyjs=(i == 1))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Evaluation Report</title></head>
<body style="font-family:Arial;max-width:1500px;margin:20px auto;">
{metrics_html}
{chart_divs}
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved report: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TRỰC QUAN HÓA ĐÁNH GIÁ MÔ HÌNH DỰ BÁO THỜI GIAN")
    print(f"  Model  : {args.model_path}")
    print(f"  Data   : {args.data_dir}")
    print(f"  Output : {args.output_dir}")
    print("=" * 60)

    print("\n[1/4] Nạp mô hình...")
    pipeline = EndToEndPipeline.load(args.model_path)

    print("\n[2/4] Nạp dữ liệu...")
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
    print(f"  Đã nạp {len(sche):,} bản ghi schedule, {len(bill):,} bills.")

    e2e_data = build_e2e_training_data(sche, bill, wh1a)
    del sche

    _, _, test_df, _, _ = prepare_e2e_splits(e2e_data, train_split=0.6, val_split=0.2)

    target_col = "time_to_first_1a_hours" if "time_to_first_1a_hours" in test_df.columns else "arrival_offset_hours"
    eval_df = test_df.dropna(subset=[target_col]).copy()
    if len(eval_df) > args.max_eval_samples:
        eval_df = eval_df.sample(args.max_eval_samples, random_state=42)

    print(f"\n[3/4] Dự báo trên {len(eval_df):,} mẫu...")
    preds = pipeline.predict(eval_df)

    y_true = eval_df[target_col].values
    pred_col = "hours_to_first_1A" if "hours_to_first_1A" in preds.columns else "predicted_hours_to_first_1A"
    y_pred = preds[pred_col].values

    bill_dates = eval_df["bill_creation_date"].values if "bill_creation_date" in eval_df.columns else None

    print(f"\n[4/4] Tạo báo cáo HTML...")
    output_path = os.path.join(args.output_dir, "evaluation_report.html")
    build_html_report(y_true, y_pred, bill_dates, output_path)

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT! Mở file để xem: {os.path.abspath(output_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
