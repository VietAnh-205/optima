# -*- coding: utf-8 -*-
"""
Danh Gia Thuc Nghiem Khung Gio Ca Dem (22h00 - 06h00) Theo Cap Kho.
Tap trung vao bieu do duong ket hop 3 duong khoi luong (kg):
  - Duong Thuc Te  (Actual)      : Tong khoi luong hang hoa ve thuc te trong ca dem
  - Duong Du Doan (Predict)      : Tong khoi luong hang hoa mo hinh du doan ve trong ca dem
  - Duong Giao    (Intersection) : Tong khoi luong cac bill trung khop trong ca 2 tap

Moi ngay trong thang cat khoang 22h00 -> 06h00 sang hom sau,
tim toan bo bills thuoc tung tap -> tinh tong khoi luong (kg) -> moi ngay = 1 diem.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_data.end_to_end.e2e_preprocessor import build_e2e_training_data
from analysis_data.end_to_end.multi_label import EndToEndPipeline

# Bang mau Corporate Blue
BLUE_DARK    = "#1e3a8a"  # Xanh navy dam (Thuc te)
BLUE_PRIMARY = "#1d4ed8"  # Xanh coban (Du doan)
BLUE_LIGHT   = "#0284c7"  # Xanh lam sang (Giao)
BLUE_PALE    = "#dbeafe"
AMBER        = "#d97706"  # Ho phach (Bo sot)
TEXT_DARK    = "#0f172a"
TEXT_MUTED   = "#475569"
BORDER       = "#cbd5e1"


# ==============================================================================
# 1. ARGUMENTS
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Bieu do duong khoi luong (kg) ca dem: Actual vs Predict vs Giao"
    )
    p.add_argument("--model-path",  type=str,
                   default=os.path.join(PROJECT_ROOT, "saved_e2e_models", "e2e_pipeline.pkl"))
    p.add_argument("--data-dir",    type=str, default=PROJECT_ROOT)
    p.add_argument("--output-dir",  type=str,
                   default=os.path.join(PROJECT_ROOT, "saved_e2e_models", "eval_plots"))
    p.add_argument("--output-file", type=str, default="operational_window_comparison.html")
    p.add_argument("--month",       type=str, default="2026-03",
                   help="YYYY-MM hoac 'auto'")
    p.add_argument("--window-start-hour", type=int, default=22)
    p.add_argument("--window-end-hour",   type=int, default=6)
    p.add_argument("--chunksize",   type=int, default=500000)
    p.add_argument("--top-pairs",   type=int, default=5,
                   help="So cap kho lon nhat duoc phan tich rieng")
    p.add_argument("--cache-dir",   type=str,
                   default=os.path.join(PROJECT_ROOT, "saved_e2e_models", "cache"))
    return p.parse_args()


# ==============================================================================
# 2. NAP DU LIEU & DU BAO
# ==============================================================================
def load_and_predict(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    target_month = args.month if args.month != "auto" else "2026-03"
    full_cache = os.path.join(args.cache_dir, f"e2e_data_month_{target_month}_FULL_ALL_BILLS.joblib")
    if os.path.exists(full_cache):
        cache_path = full_cache
    else:
        cache_path = os.path.join(args.cache_dir, f"e2e_data_chunk_{args.chunksize}.joblib")

    if os.path.exists(cache_path):
        print(f"  [Cache] Dang nap e2e_data tu: {cache_path}")
        e2e_data = joblib.load(cache_path)
        print(f"  [Cache] Nap thanh cong {len(e2e_data):,} ban ghi.")
    else:
        print(f"  [Data] Doc du lieu tu {args.data_dir}...")
        wh1a = pd.read_csv(os.path.join(args.data_dir, "warehouse_1A.csv"))
        bill = pd.read_csv(os.path.join(args.data_dir, "bill.csv"))
        chunks, total = [], 0
        for chunk in pd.read_csv(os.path.join(args.data_dir, "bill_schedule.csv"), chunksize=100_000):
            chunk["io_time"] = pd.to_datetime(chunk["io_time"], errors="coerce")
            chunks.append(chunk)
            total += len(chunk)
            if total >= args.chunksize:
                break
        sche = pd.concat(chunks, ignore_index=True)
        del chunks
        print(f"  [Data] Da nap {total:,} ban ghi schedule.")
        e2e_data = build_e2e_training_data(sche, bill, wh1a)
        del sche, bill, wh1a
        try:
            joblib.dump(e2e_data, cache_path, compress=3)
            print(f"  [Cache] Da luu: {cache_path}")
        except Exception as err:
            print(f"  [Cache] Khong the ghi: {err}")

    e2e_data["arrival_time"] = pd.to_datetime(e2e_data["arrival_time"], errors="coerce")
    e2e_data = e2e_data.dropna(subset=["arrival_time"]).copy()
    e2e_data["year_month"] = e2e_data["arrival_time"].dt.to_period("M").astype(str)

    month_counts = e2e_data["year_month"].value_counts()
    if args.month == "auto" or args.month not in month_counts.index:
        selected = str(month_counts.idxmax())
    else:
        selected = args.month

    print(f"\n  [Thang] Su dung thang: {selected} ({month_counts.get(selected, 0):,} ban ghi)")

    year, month_num = map(int, selected.split("-"))
    start_dt = pd.Timestamp(year, month_num, 1)
    end_dt   = pd.Timestamp(year + (month_num // 12), month_num % 12 + 1, 3) if month_num < 12 \
               else pd.Timestamp(year + 1, 1, 3)

    df = e2e_data[(e2e_data["arrival_time"] >= start_dt) & (e2e_data["arrival_time"] < end_dt)].copy()
    if len(df) == 0:
        raise ValueError(f"Khong co du lieu cho thang {selected}!")

    cache_dir = os.path.join(os.path.dirname(args.model_path), "cache")
    cache_pred_file = os.path.join(cache_dir, f"e2e_predicted_{selected}_op_window.joblib")
    if os.path.exists(cache_pred_file):
        print(f"  [Cache] Nap DataFrame da du bao tu: {cache_pred_file}")
        df = joblib.load(cache_pred_file)
        return df, selected

    print(f"  [Pipeline] Nap mo hinh tu {args.model_path}...")
    pipeline = EndToEndPipeline.load(args.model_path)

    print(f"  [Pipeline] Du bao {len(df):,} mau thang {selected}...")
    preds = pipeline.predict(df)

    df["pred_first_1A"] = preds["first_1A"].astype(str).values
    df["pred_last_1A"]  = preds["last_1A"].astype(str).values
    df["predicted_arrival_time"] = pd.to_datetime(preds["predicted_arrival_time"], errors="coerce")

    df["actual_pair"] = df["first_1A"].astype(str) + " -> " + df["last_1A"].astype(str)
    df["pred_pair"]   = df["pred_first_1A"].astype(str) + " -> " + df["pred_last_1A"].astype(str)

    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    df = df.sort_values("snapshot_time")

    try:
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(df, cache_pred_file)
        print(f"  [Cache] Da luu DataFrame du bao vao cache: {cache_pred_file}")
    except Exception as e:
        print(f"  [Cache] Khong the luu cache du bao: {e}")

    return df, selected


# ==============================================================================
# 3. TINH TOAN KHOI LUONG TUNG NGAY CHO 1 ROUTE
# ==============================================================================
def compute_daily_series(df: pd.DataFrame, route_name: str,
                          selected_month: str,
                          start_hour: int = 22, end_hour: int = 6) -> pd.DataFrame:
    """
    Moi ngay trong thang:
      - Cat khung gio tu start_hour den end_hour sang hom sau
      - Tap Actual  A = {bill | arrival_time in [ws, we)}
      - Tap Predict P = {bill | predicted_arrival_time in [ws, we)}
      - Tap Giao    I = A cap P
      - Tong khoi luong (kg) duoc tinh tren tap hop bill_code duy nhat
    """
    year, month_num = map(int, selected_month.split("-"))
    num_days = pd.Period(selected_month, freq="M").days_in_month
    is_all   = (route_name == "ALL_ROUTES")

    # Map bill_code -> actual_weight
    bill_weight_map = df.drop_duplicates("bill_code").set_index("bill_code")["actual_weight"].fillna(0.0).to_dict()

    rows = []
    for day in range(1, num_days + 1):
        ws = pd.Timestamp(year, month_num, day, start_hour, 0, 0)
        we = ws + pd.Timedelta(hours=(24 - start_hour + end_hour))

        # 1. Tap Actual (A): Cac bill thuc te cap ben kho trong ca dem [ws, we)
        mask_a = (df["arrival_time"] >= ws) & (df["arrival_time"] < we)
        if not is_all:
            mask_a = mask_a & (df["actual_pair"] == route_name)
        bills_a = set(df.loc[mask_a, "bill_code"].unique())

        # 2. Tap Predict (P) voi Cut-off Time nghiep vu tai moc ws:
        # Lay snapshot moi nhat cua cac bill tinh den moc ws va dang luon chuyen (arrival_time >= ws)
        sub_cut = df[df["snapshot_time"] <= ws]
        if not sub_cut.empty:
            latest = sub_cut.drop_duplicates("bill_code", keep="last")
            in_transit = latest[latest["arrival_time"] >= ws]
            mask_p = (in_transit["predicted_arrival_time"] >= ws) & (in_transit["predicted_arrival_time"] < we)
            if not is_all:
                mask_p = mask_p & (in_transit["pred_pair"] == route_name)
            bills_p = set(in_transit.loc[mask_p, "bill_code"].unique())
        else:
            bills_p = set()

        # 3. Tap Giao (Intersection): I = A cap P
        bills_i = bills_a & bills_p

        w_actual  = float(sum(bill_weight_map.get(b, 0.0) for b in bills_a))
        w_predict = float(sum(bill_weight_map.get(b, 0.0) for b in bills_p))
        w_inter   = float(sum(bill_weight_map.get(b, 0.0) for b in bills_i))

        rows.append({
            "day":        day,
            "date_str":   f"{day:02d}/{month_num:02d}",
            "ws":         ws,
            "we":         we,
            "w_actual":   w_actual,
            "w_predict":  w_predict,
            "w_inter":    w_inter,
            "n_actual":   len(bills_a),
            "n_predict":  len(bills_p),
            "n_inter":    len(bills_i),
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 4. TAO BIEU DO DUONG KHOI LUONG (3 DUONG)
# ==============================================================================
def make_line_chart(daily: pd.DataFrame, route_label: str,
                    start_hour: int = 22, end_hour: int = 6) -> go.Figure:
    """
    Bieu do duong ket hop 3 duong khoi luong (kg):
      - Duong Thuc Te  (Actual)       : xanh navy, net lien
      - Duong Du Doan (Predict)       : xanh coban, net dut
      - Duong Giao    (Intersection)  : xanh lam sang, net cham
    Moi diem tren truc X = 1 ngay trong thang.
    Truc Y = Tong khoi luong hang hoa (kg).
    """
    x     = daily["date_str"].tolist()
    w_act = daily["w_actual"].tolist()
    w_pre = daily["w_predict"].tolist()
    w_int = daily["w_inter"].tolist()

    fig = go.Figure()

    # -- Duong Thuc Te --------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=x, y=w_act,
        mode="lines+markers",
        name="Thực Tế (A)",
        line=dict(color=BLUE_DARK, width=2.5),
        marker=dict(size=7, color=BLUE_DARK, symbol="circle"),
        hovertemplate=(
            "<b>Ngày %{x}</b><br>"
            + f"Ca đêm {start_hour:02d}:00 → {end_hour:02d}:00<br>"
            + "Khối lượng TT: <b>%{y:,.1f} kg</b><extra></extra>"
        ),
    ))

    # -- Duong Du Doan --------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=x, y=w_pre,
        mode="lines+markers",
        name="Dự Đoán (P)",
        line=dict(color=BLUE_PRIMARY, width=2.5, dash="dash"),
        marker=dict(size=7, color=BLUE_PRIMARY, symbol="diamond"),
        hovertemplate=(
            "<b>Ngày %{x}</b><br>"
            + f"Ca đêm {start_hour:02d}:00 → {end_hour:02d}:00<br>"
            + "Khối lượng DĐ: <b>%{y:,.1f} kg</b><extra></extra>"
        ),
    ))

    # -- Duong Giao -----------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=x, y=w_int,
        mode="lines+markers",
        name="Giao (A ∩ P)",
        line=dict(color=BLUE_LIGHT, width=2.0, dash="dot"),
        marker=dict(size=6, color=BLUE_LIGHT, symbol="square"),
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.08)",
        hovertemplate=(
            "<b>Ngày %{x}</b><br>"
            + f"Ca đêm {start_hour:02d}:00 → {end_hour:02d}:00<br>"
            + "Khối lượng Giao: <b>%{y:,.1f} kg</b><extra></extra>"
        ),
    ))

    # Vung phu giua Actual va Giao (phan bo sot)
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=w_act + w_int[::-1],
        fill="toself",
        fillcolor="rgba(217, 119, 6, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name=r"Bỏ Sót (A \ P)",
    ))

    # Layout
    y_max = max(max(w_act or [0]), max(w_pre or [0])) * 1.18 + 10

    fig.update_layout(
        title=dict(
            text=f"<b>{route_label}</b><br>"
                 f"<span style='font-size:12.5px;color:{TEXT_MUTED}'>Khối lượng hàng hoá trong ca đêm "
                 f"{start_hour:02d}h00 – {end_hour:02d}h00 (kg / ngày)</span>",
            font=dict(size=15, color=TEXT_DARK, family="Inter, Arial, sans-serif"),
            x=0, xanchor="left",
            y=0.98, yanchor="top",
        ),
        xaxis=dict(
            title=dict(text="Ngày trong tháng", font=dict(size=12, color=TEXT_MUTED)),
            tickangle=-45,
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=BORDER,
            linewidth=1,
        ),
        yaxis=dict(
            title=dict(text="Khối lượng (kg)", font=dict(size=12, color=TEXT_MUTED)),
            range=[0, y_max],
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            zeroline=True,
            zerolinecolor=BORDER,
            zerolinewidth=1,
            showline=True,
            linecolor=BORDER,
            linewidth=1,
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11.5),
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=65, r=30, t=85, b=65),
        height=430,
        hovermode="x unified",
    )

    return fig


# ==============================================================================
# 5. XAY DUNG BAO CAO HTML
# ==============================================================================
def build_html_report(
    all_results: dict,
    selected_month: str,
    output_path: str,
    start_hour: int = 22,
    end_hour: int = 6,
):
    route_keys = list(all_results.keys())

    # Dropdown options
    options_html = ""
    for idx, rk in enumerate(route_keys):
        label = "Toàn Mạng Lưới" if rk == "ALL_ROUTES" else rk
        options_html += f'<option value="panel_{idx}">{label}</option>\n'

    # Route panels
    panels_html = ""
    for idx, rk in enumerate(route_keys):
        info     = all_results[rk]
        daily    = info["daily"]
        fig_html = info["fig_html"]
        label    = "Toàn Mạng Lưới" if rk == "ALL_ROUTES" else rk

        total_a = float(daily["w_actual"].sum())
        total_p = float(daily["w_predict"].sum())
        total_i = float(daily["w_inter"].sum())
        total_bias = ((total_p - total_a) / total_a * 100) if total_a > 0 else 0.0
        total_bias_str = f"{'+' if total_bias > 0 else ''}{total_bias:.1f}%"
        recall_pct  = (total_i / total_a * 100) if total_a > 0 else 0.0
        prec_pct    = (total_i / total_p * 100) if total_p > 0 else 0.0
        jaccard_pct = (total_i / (total_a + total_p - total_i) * 100) if (total_a + total_p - total_i) > 0 else 0.0

        display = "display:block;" if idx == 0 else "display:none;"
        panels_html += f"""
  <div id="panel_{idx}" class="route-panel" style="{display}">

    <!-- Summary strip -->
    <div class="summary-strip">
      <div class="s-item">
        <div class="s-label">Khối Lượng Thực Tế</div>
        <div class="s-value" style="color:{BLUE_DARK};">{total_a:,.1f} <span class="unit">kg</span></div>
      </div>
      <div class="s-item">
        <div class="s-label">Khối Lượng Dự Đoán</div>
        <div class="s-value" style="color:{BLUE_PRIMARY};">{total_p:,.1f} <span class="unit">kg</span></div>
      </div>
      <div class="s-item">
        <div class="s-label">Độ Lệch Tải (Bias)</div>
        <div class="s-value">{total_bias_str}</div>
      </div>
      <div class="s-item">
        <div class="s-label">Khối Lượng Giao (A ∩ P)</div>
        <div class="s-value" style="color:{BLUE_LIGHT};">{total_i:,.1f} <span class="unit">kg</span></div>
      </div>
      <div class="s-item">
        <div class="s-label">Recall Ca Đêm</div>
        <div class="s-value">{recall_pct:.1f}%</div>
      </div>
      <div class="s-item">
        <div class="s-label">Precision Ca Đêm</div>
        <div class="s-value">{prec_pct:.1f}%</div>
      </div>
      <div class="s-item">
        <div class="s-label">Jaccard (Khối Lượng)</div>
        <div class="s-value">{jaccard_pct:.1f}%</div>
      </div>
    </div>

    <!-- Line chart -->
    <div class="chart-card">
      {fig_html}
    </div>

    <!-- Table Legend Card: Chu thich cac tap hop & cac do do (Tren bang) -->
    <div class="table-legend-card">
      <div class="tl-header">
        <span class="tl-badge">Hướng Dẫn Đối Chiếu Nghiệp Vụ</span>
        <span class="tl-title">Ý Nghĩa Các Tập Hợp Dữ Liệu & Độ Đo Trong Bảng</span>
      </div>
      <div class="tl-grid">
        <!-- Cot 1: Chu thich cac tap hop -->
        <div class="tl-col">
          <div class="tl-col-header">1. Quy Ước Các Tập Hợp Vận Đơn (Sets)</div>
          <div class="tl-list">
            <div class="tl-item">
              <span class="dot" style="background:{BLUE_DARK};"></span>
              <div><b>Tập Thực Tế (A):</b> Toàn bộ mã vận đơn thực tế cập bến kho đích trong ca đêm (22h00 &rarr; 06h00).</div>
            </div>
            <div class="tl-item">
              <span class="dot" style="background:transparent;border:2px dashed {BLUE_PRIMARY};"></span>
              <div><b>Tập Dự Đoán (P):</b> Các mã vận đơn mô hình dự báo sẽ về trong ca đêm theo mốc chốt số Cut-off Time.</div>
            </div>
            <div class="tl-item">
              <span class="dot" style="background:{BLUE_LIGHT};"></span>
              <div><b>Tập Giao (A &cap; P):</b> Đơn hàng xuất hiện trong CẢ HAI tập — dự đoán chính xác cả tuyến kho lẫn ca đến.</div>
            </div>
            <div class="tl-item">
              <span class="dot" style="background:{AMBER};opacity:0.6;"></span>
              <div><b>Tập Bỏ Sót (A \\ P):</b> Đơn thực tế có về trong ca đêm nhưng mô hình dự đoán trệch ca (False Negatives).</div>
            </div>
            <div class="tl-item">
              <span class="dot" style="background:#94a3b8;"></span>
              <div><b>Tập Dự Đoán Ảo (P \\ A):</b> Đơn mô hình dự báo về ca đêm nhưng thực tế không về ca này (False Positives).</div>
            </div>
          </div>
        </div>

        <!-- Cot 2: Chu thich cac do do -->
        <div class="tl-col">
          <div class="tl-col-header">2. Giải Thích Các Chỉ Số & Độ Đo (Metrics)</div>
          <div class="tl-list">
            <div class="tl-item">
              <div><b>Khối Lượng TT / DĐ / Giao (kg):</b> Tổng khối lượng tính trên các mã vận đơn duy nhất của từng tập tương ứng.</div>
            </div>
            <div class="tl-item">
              <div><b>Độ Lệch Tải (% Bias):</b> Chênh lệch tương đối giữa tổng khối lượng DĐ và TT: <code style="font-size:11.5px;background:#e2e8f0;padding:1px 4px;border-radius:3px;">[(DĐ - TT) / TT] &times; 100%</code>. Dấu <b>(+)</b> là thừa tải, <b>(-)</b> là thiếu tải. Ngày có sản lượng TT &lt; 100kg ghi nhận <i>N/A</i>.</div>
            </div>
            <div class="tl-item">
              <div><b>Bỏ Sót / DĐ Ảo (kg):</b> Khối lượng hàng thực tế bị bỏ sót <code style="font-size:11.5px;background:#e2e8f0;padding:1px 4px;border-radius:3px;">max(0, TT - Giao)</code> hoặc dự đoán ảo <code style="font-size:11.5px;background:#e2e8f0;padding:1px 4px;border-radius:3px;">max(0, DĐ - Giao)</code>.</div>
            </div>
            <div class="tl-item">
              <div><b>Recall (%) &bull; Precision (%):</b> Recall = <code style="font-size:11.5px;background:#e2e8f0;padding:1px 4px;border-radius:3px;">(Giao / TT) &times; 100%</code> (độ bao phủ hàng thực tế). Precision = <code style="font-size:11.5px;background:#e2e8f0;padding:1px 4px;border-radius:3px;">(Giao / DĐ) &times; 100%</code> (độ chuẩn xác của tải dự báo).</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Daily detail table -->
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ngày</th>
            <th>Khung Giờ Ca Đêm</th>
            <th class="text-right">Khối Lượng TT (kg)</th>
            <th class="text-right">Khối Lượng DĐ (kg)</th>
            <th class="text-right">Độ Lệch Tải (%)</th>
            <th class="text-right">Khối Lượng Giao (kg)</th>
            <th class="text-right">Bỏ Sót (kg)</th>
            <th class="text-right">DĐ Ảo (kg)</th>
            <th class="text-right">Recall (%)</th>
            <th class="text-right">Precision (%)</th>
          </tr>
        </thead>
        <tbody>
"""
        for _, row in daily.iterrows():
            w_a, w_p, w_i = float(row["w_actual"]), float(row["w_predict"]), float(row["w_inter"])
            w_miss = max(0.0, w_a - w_i)
            w_over = max(0.0, w_p - w_i)
            rec  = (w_i / w_a * 100) if w_a > 0 else 0.0
            prec = (w_i / w_p * 100) if w_p > 0 else 0.0
            
            # Tinh Do Lech Tai (%) - Phuong an 1 (co dau +/-, khong to mau)
            if w_a >= 100.0:
                bias_val = ((w_p - w_a) / w_a) * 100.0
                sign = "+" if bias_val > 0 else ""
                bias_str = f"{sign}{bias_val:.1f}%"
            elif w_a > 0:
                bias_str = "N/A (<100kg)"
            else:
                bias_str = "N/A"

            ws_str = pd.Timestamp(row["ws"]).strftime("%d/%m %H:%M")
            we_str = pd.Timestamp(row["we"]).strftime("%d/%m %H:%M")
            miss_color = f"color:{AMBER};" if w_miss > 0 else ""
            panels_html += f"""          <tr>
            <td><b>{row['date_str']}</b></td>
            <td class="text-muted">{ws_str} &rarr; {we_str}</td>
            <td class="text-right">{w_a:,.1f}</td>
            <td class="text-right">{w_p:,.1f}</td>
            <td class="text-right">{bias_str}</td>
            <td class="text-right" style="color:{BLUE_PRIMARY};">{w_i:,.1f}</td>
            <td class="text-right" style="{miss_color}">{w_miss:,.1f}</td>
            <td class="text-right">{w_over:,.1f}</td>
            <td class="text-right">{rec:.1f}%</td>
            <td class="text-right">{prec:.1f}%</td>
          </tr>\n"""

        panels_html += f"""        </tbody>
      </table>
    </div>

  </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Đánh Giá Khối Lượng Ca Đêm (22h – 06h) &bull; Tháng {selected_month}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --blue-dark:    {BLUE_DARK};
      --blue-primary: {BLUE_PRIMARY};
      --blue-light:   {BLUE_LIGHT};
      --blue-pale:    {BLUE_PALE};
      --text-dark:    {TEXT_DARK};
      --text-muted:   {TEXT_MUTED};
      --border:       {BORDER};
      --amber:        {AMBER};
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', Arial, sans-serif;
      background: #f8fafc;
      color: var(--text-dark);
      font-size: 14px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}

    /* MAIN CONTAINER */
    .main {{ max-width: 1380px; margin: 0 auto; padding: 24px 32px; }}

    /* TOOLBAR COMPACT TITLE & SELECTOR */
    .toolbar {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 20px;
      margin-bottom: 20px;
      display: flex; align-items: center; justify-content: space-between;
      gap: 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,.03);
    }}
    .toolbar-left {{ display: flex; flex-direction: column; gap: 2px; }}
    .tb-title {{ font-size: 16px; font-weight: 700; color: var(--text-dark); letter-spacing: -.01em; }}
    .tb-subtitle {{ font-size: 12px; color: var(--text-muted); }}
    .toolbar-right {{ display: flex; align-items: center; gap: 10px; }}
    .toolbar-right label {{ font-weight: 600; font-size: 13px; white-space: nowrap; color: var(--text-dark); }}
    .route-select {{
      min-width: 340px;
      padding: 8px 12px;
      font-size: 13px; font-weight: 500;
      border: 1px solid var(--border); border-radius: 6px;
      background: #fff; color: var(--text-dark);
      cursor: pointer; outline: none;
    }}
    .route-select:focus {{
      border-color: var(--blue-primary);
      box-shadow: 0 0 0 2px rgba(29,78,216,.15);
    }}

    /* SUMMARY STRIP */
    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .s-item {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
    }}
    .s-label {{ font-size: 11px; font-weight: 600; color: var(--text-muted);
                text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }}
    .s-value {{ font-size: 22px; font-weight: 700; letter-spacing: -.02em; color: var(--text-dark); }}
    .unit {{ font-size: 13px; font-weight: 500; color: var(--text-muted); }}

    /* CHART CARD */
    .chart-card {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,.02);
    }}

    /* TABLE LEGEND CARD (ABOVE TABLE) */
    .table-legend-card {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,.02);
    }}
    .tl-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid #e2e8f0;
    }}
    .tl-badge {{
      background: var(--blue-pale);
      color: var(--blue-primary);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .03em;
      padding: 3px 8px;
      border-radius: 4px;
    }}
    .tl-title {{
      font-size: 13.5px;
      font-weight: 700;
      color: var(--text-dark);
    }}
    .tl-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 20px;
      font-size: 12.5px;
      line-height: 1.6;
    }}
    .tl-col {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .tl-col-header {{
      font-weight: 600;
      color: var(--blue-dark);
      font-size: 13px;
      margin-bottom: 2px;
    }}
    .tl-list {{
      display: flex;
      flex-direction: column;
      gap: 7px;
    }}
    .tl-item {{
      display: flex;
      align-items: flex-start;
      gap: 8px;
      color: #334155;
    }}
    .tl-item .dot {{
      margin-top: 4px;
    }}

    /* TABLE */
    .table-wrap {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow-x: auto;
      margin-bottom: 20px;
      box-shadow: 0 1px 2px rgba(0,0,0,.02);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    th {{
      background: #f8fafc; color: var(--text-dark); font-weight: 600;
      padding: 10px 14px; border-bottom: 1px solid var(--border);
      white-space: nowrap; text-align: left;
    }}
    td {{
      padding: 9px 14px;
      border-bottom: 1px solid #f1f5f9;
      color: #334155;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f8fafc; }}
    .text-right {{ text-align: right; }}
    .text-muted  {{ color: var(--text-muted); font-size: 12px; }}

    /* LEGEND BOX */
    .legend-box {{
      background: #eff6ff;
      border-left: 4px solid var(--blue-primary);
      border-radius: 0 6px 6px 0;
      padding: 14px 18px;
      margin-bottom: 28px;
      font-size: 12.5px;
      color: #1e3a8a;
      line-height: 1.7;
      display: flex; flex-direction: column; gap: 4px;
    }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; }}
    .dot {{
      display: inline-block; width: 12px; height: 12px;
      border-radius: 50%; flex-shrink: 0;
    }}

    /* FOOTER */
    footer {{
      text-align: center; padding: 18px 40px;
      border-top: 1px solid var(--border);
      font-size: 11.5px; color: var(--text-muted);
      background: #fff;
    }}
  </style>
</head>
<body>

<main class="main">

  <!-- Compact Toolbar (no gigantic header) -->
  <div class="toolbar">
    <div class="toolbar-left">
      <div class="tb-title">Đánh Giá Vận Hành Ca Đêm (22h00 – 06h00) &bull; Khối Lượng (kg)</div>
      <div class="tb-subtitle">Tháng đánh giá: <b>{selected_month}</b> &bull; Đối chiếu 3 đường: Thực Tế vs Dự Đoán vs Giao trùng khớp hàng ngày</div>
    </div>
    <div class="toolbar-right">
      <label for="routeSel">Chọn Cặp Kho Tuyến:</label>
      <select id="routeSel" class="route-select" onchange="switchPanel(this.value)">
        {options_html}
      </select>
    </div>
  </div>

  <!-- Route panels -->
  {panels_html}

</main>

<footer>
  Hệ Thống Tối Ưu Hóa Logistics &bull; Đánh Giá Khối Lượng Ca Đêm (22h00 – 06h00) Theo Cặp Kho
</footer>

<script>
  function switchPanel(targetId) {{
    document.querySelectorAll('.route-panel').forEach(p => p.style.display = 'none');
    const el = document.getElementById(targetId);
    if (el) {{
      el.style.display = 'block';
      window.dispatchEvent(new Event('resize'));
    }}
  }}
</script>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [HTML] Da luu: {output_path}")


# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"BIEU DO DUONG KHOI LUONG: ACTUAL vs PREDICT vs GIAO — CA DEM (22h – 06h)")
    print(f"  Model    : {args.model_path}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Output   : {args.output_dir}")
    print(f"  Thang    : {args.month}")
    print(f"  Khung gio: {args.window_start_hour:02d}h – {args.window_end_hour:02d}h")
    print("=" * 70)

    # 1. Nap data & du bao
    df, selected_month = load_and_predict(args)

    # 2. Xac dinh cac route
    route_counts = df["actual_pair"].value_counts()
    top_routes   = route_counts.head(args.top_pairs).index.tolist()
    routes       = ["ALL_ROUTES"] + top_routes

    print(f"\n[Routes] {len(top_routes)} cap kho lon nhat trong thang {selected_month}:")
    for r in top_routes:
        print(f"  - {r}: {route_counts[r]:,} ban ghi")

    # 3. Tinh toan & tao bieu do cho tung route
    all_results = {}
    for rk in routes:
        label = "Toàn Mạng Lưới" if rk == "ALL_ROUTES" else rk
        print(f"  -> Xu ly: {label}")
        daily = compute_daily_series(
            df, rk, selected_month,
            args.window_start_hour,
            args.window_end_hour,
        )
        fig     = make_line_chart(daily, label, args.window_start_hour, args.window_end_hour)
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False)

        all_results[rk] = {"daily": daily, "fig_html": fig_html}

    # 4. Xuat HTML
    output_path = os.path.join(args.output_dir, args.output_file)
    build_html_report(
        all_results,
        selected_month,
        output_path,
        args.window_start_hour,
        args.window_end_hour,
    )

    print("\n" + "=" * 70)
    print("HOAN TAT!")
    print(f"  --> {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
