import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go

def _parse_pair_info(pair_str, flow_name):
    """Trích tên kho + tỉnh 2 đầu từ pair string dạng 'KhoA (TinhA) → KhoB (TinhB)'
    và xác định bên nào là Kho 1A trung chuyển dựa vào flow_name, để có thể nhóm
    theo tỉnh đối tác (kho gửi/nhận thật) và theo kho 1A một cách nhất quán."""
    m = re.match(r'^(.*?)\s*\((.*?)\)\s*→\s*(.*?)\s*\((.*?)\)$', pair_str.strip())
    if not m:
        return {"kho_1a": "N/A", "prov_1a": "N/A", "kho_partner": pair_str, "prov_partner": "N/A"}
    kho_a, prov_a, kho_b, prov_b = [g.strip() for g in m.groups()]
    if "Kho 1A đích" in flow_name:
        # Luồng 2: Kho 1A đích → Kho nhận  => bên trái (A) là Kho 1A
        return {"kho_1a": kho_a, "prov_1a": prov_a, "kho_partner": kho_b, "prov_partner": prov_b}
    # Luồng 1: Kho gửi → Kho 1A nguồn  => bên phải (B) là Kho 1A
    return {"kho_1a": kho_b, "prov_1a": prov_b, "kho_partner": kho_a, "prov_partner": prov_a}


def _weighted_cluster_std(cluster_stats):
    """Std trung bình có trọng số theo số bill/cụm - đại diện mức biến động TỰ
    NHIÊN của y_offset tại cặp kho này (khác với sai số của model)."""
    total_n, s = 0, 0.0
    for cs in (cluster_stats or []):
        try:
            std_val = float(str(cs.get('std', '')).replace('h', ''))
        except (ValueError, TypeError):
            continue
        n = cs.get('n', 0) or 0
        s += std_val * n
        total_n += n
    return s / total_n if total_n > 0 else np.nan


def _weighted_avg(df, value_col, weight_col):
    """Trung bình có trọng số theo n_bill (ở đây dùng n_test - đúng số quan sát
    mà MAE/MAPE của từng cặp được tính trên đó), thay vì coi mọi cặp kho ngang
    nhau như trung bình đơn giản."""
    d = df[[value_col, weight_col]].dropna()
    if d.empty or d[weight_col].sum() <= 0:
        return np.nan
    return float((d[value_col] * d[weight_col]).sum() / d[weight_col].sum())


def _volume_tier(n_bills):
    if n_bills < 500:
        return "Nhỏ (<500 bill)"
    elif n_bills < 2000:
        return "Vừa (500-2000 bill)"
    return "Lớn (>2000 bill)"


def _build_pair_level_df(results_by_flow):
    """1 dòng / cặp kho, CHỈ lấy best_combo (model thực sự được chọn để triển
    khai) - dùng làm nền cho Tổng quan (weighted) + Phân nhóm theo tỉnh/kho 1A,
    tránh trộn lẫn với các combo thua trong lúc tuning."""
    rows = []
    for flow_name, pair_reports in results_by_flow:
        for rep in pair_reports:
            bc  = rep.get("best_combo", {}) or {}
            tm  = rep.get("test_metrics", {}) or {}
            trm = rep.get("train_metrics", {}) or {}
            info = _parse_pair_info(rep["pair"], flow_name)
            n_test = rep.get("n_test", 0) or 0
            n_tv   = rep.get("n_samples", 0) or 0
            n_clusters = rep.get("n_clusters", 0) or 0
            rows.append({
                "flow": flow_name, "pair": rep["pair"],
                "kho_1a": info["kho_1a"], "prov_1a": info["prov_1a"],
                "kho_partner": info["kho_partner"], "prov_partner": info["prov_partner"],
                "n_bills": n_tv + n_test, "n_test": n_test,
                "n_clusters": n_clusters, "is_single_group": n_clusters < 2,
                "volume_tier": _volume_tier(n_tv + n_test),
                "cluster_selection": bc.get("cs_name", "N/A"),
                "feature_option":    bc.get("fo_name", "N/A"),
                "regression_model":  bc.get("reg_name", "N/A"),
                "test_MAE": tm.get("MAE"), "test_RMSE": tm.get("RMSE"),
                "test_R2":  tm.get("R2"),  "test_MAPE": tm.get("MAPE"),
                "train_MAE": trm.get("MAE"),
                "data_std": _weighted_cluster_std(rep.get("cluster_stats", [])),
            })
    return pd.DataFrame(rows)


def build_html_report(results_by_flow, summary_df):
    html = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Multi-Strategy Regression Pipeline</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f1f5f9; color: #1e293b; padding: 24px; }
  h1 { text-align: center; font-size: 24px; margin-bottom: 8px; }
  .subtitle { text-align: center; color: #64748b; font-size: 14px; margin-bottom: 24px; }
  .flow-header { background: linear-gradient(135deg, #1e40af, #3b82f6); color: white;
                   padding: 14px 24px; border-radius: 10px; font-size: 18px; font-weight: 700;
                   margin: 32px 0 16px; box-shadow: 0 2px 8px rgba(30,64,175,0.3); }
  .summary-card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 24px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .summary-card h2 { font-size: 16px; margin-bottom: 12px; color: #334155; }
  .note-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px;
                 margin-bottom: 24px; font-size: 13px; line-height: 1.6; color: #334155; }
  .note-card h2 { font-size: 16px; margin-bottom: 12px; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
  .note-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .note-section { margin-bottom: 12px; }
  .note-section b { color: #1e293b; display: block; margin-bottom: 4px; font-weight: 600; }
  .avg-card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 24px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .avg-card h2 { font-size: 16px; margin-bottom: 12px; color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
  .avg-grid { display: flex; flex-direction: column; gap: 32px; }
  .avg-row-container { display: flex; flex-direction: column; gap: 16px; 
                       background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; }
  .avg-table-wrap { overflow-x: auto; background: white; border-radius: 8px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
  .avg-chart-wrap { width: 100%; min-height: 320px; background: white; border-radius: 8px; 
                    border: 1px solid #e2e8f0; padding: 12px; }
  .avg-table-wrap h3 { font-size: 14px; font-weight: 700; color: #1e293b; margin: 0; padding: 12px 16px;
                       background: #f1f5f9; border-bottom: 1px solid #e2e8f0; border-radius: 8px 8px 0 0; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #ffffff; padding: 12px 16px; text-align: left; font-weight: 600;
       border-bottom: 2px solid #e2e8f0; color: #475569; user-select: none; white-space: nowrap; }
  th.sortable:hover { background: #f1f5f9; cursor: pointer; }
  td { padding: 10px 16px; border-bottom: 1px solid #f1f5f9; }
  tr:nth-child(even) td { background: #fafafa; }
  tr:hover td { background: #eff6ff; }
  .pair-card { background: white; border-radius: 8px; margin-bottom: 24px; overflow: hidden;
                 border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
  .pair-header { background: #f8fafc; padding: 16px 20px; border-bottom: 1px solid #e2e8f0;
                   display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
  .pair-title { font-size: 16px; font-weight: 700; color: #334155; }
  .pair-meta { display: flex; gap: 10px; flex-wrap: wrap; }
  .meta-badge { background: #ffffff; color: #64748b; padding: 4px 10px; border-radius: 6px;
                  font-size: 12px; font-weight: 500; border: 1px solid #e2e8f0; }
  .meta-badge.strong { color: #0f172a; font-weight: 600; background: #f1f5f9; }
  .pair-body { padding: 20px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-container { min-height: 400px; }
  .section-title { font-size: 14px; font-weight: 700; color: #334155; margin: 16px 0 12px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
  .test-box { margin-bottom: 24px; padding: 16px 20px; background: #f8fafc; border: 1px solid #e2e8f0;
                border-left: 4px solid #3b82f6; border-radius: 6px; }
  .test-box h3 { margin-bottom: 12px; color: #1e293b; font-size: 15px; font-weight: 700; }
  tr.best-row td { font-weight: 600; background: #f0fdf4; color: #166534; }
  tr.best-row td:first-child { border-left: 3px solid #22c55e; }
  .kpi-grid { display: flex; gap: 16px; flex-wrap: wrap; }
  .kpi-card { flex: 1; min-width: 170px; background: #f8fafc; border: 1px solid #e2e8f0;
               border-radius: 8px; padding: 14px 16px; }
  .kpi-card .kpi-label { font-size: 12px; color: #64748b; margin-bottom: 4px; }
  .kpi-card .kpi-value { font-size: 22px; font-weight: 700; color: #0f172a; }
  .kpi-card .kpi-sub { font-size: 11px; color: #94a3b8; margin-top: 3px; line-height: 1.4; }
  .group-grid-2x2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 900px) { .group-grid-2x2 { grid-template-columns: 1fr; } }
</style>
</head><body>
<h1>Multi-Strategy Regression Pipeline</h1>
<p class="subtitle">Train → Val selection (3 chiến lược chọn cụm × 2 feature options × 3 mô hình hồi quy) → Retrain trainval → Test ONCE</p>

<div class="note-card">
  <h2>Hướng dẫn đọc báo cáo</h2>
  <div class="note-grid">
    <div>
      <div class="note-section"><b>Pipeline (mỗi cặp kho)</b>
        Bước 1: Tách Test 20% TRƯỚC (không đụng tới cho đến bước cuối) →
        Bước 2: DBSCAN phân cụm y_offset CHỈ trên trainval (80%) →
        Bước 3: Val Selection bằng K-Fold CV trên toàn bộ trainval (mỗi điểm dữ liệu
        được dùng làm validation đúng 1 lần, out-of-fold) → chọn best combo theo MAE gộp thấp nhất →
        Bước 4: Retrain final models trên toàn bộ trainval với best combo →
        Bước 5: Test ONCE – chỉ chạy đúng 1 lần trên test với best combo
      </div>
      <div class="note-section"><b>Chiến lược chọn cụm</b>
        <u>Cách 1 (Dominant Cluster)</u>: Với mỗi khung giờ, dùng cụm chiếm đa số trong training – đơn giản, ổn định<br>
        <u>Cách 2 (ML Classifier)</u>: RandomForest tự phân loại cụm từ full features – linh hoạt hơn<br>
        <u>Cách 3 (Hybrid)</u>: Dùng ML nếu xác suất ≥ ngưỡng (mặc định 60%), còn lại dùng Cách 1<br>
        <u>Cách 0 (Không phân cụm)</u>: Áp dụng khi DBSCAN không tách được ≥2 cụm cho cặp kho đó
        (dữ liệu đồng nhất/ít mẫu) – vẫn dự báo bình thường, chỉ bỏ qua layer chọn cụm/classifier,
        không phải bị loại khỏi báo cáo
      </div>
      <div class="note-section"><b>Lưu ý quan trọng</b>
        Test hoàn toàn không tham gia DBSCAN và không có nhãn cụm thật, nên
        không tồn tại eval scope "Dominant (Correct)" mang tính oracle trên test.
        Cụm cho test được suy ra bằng <u>pick_cluster_by_hour</u> (inference-only).
        "Train (diagnostic)" chỉ để so sánh phát hiện overfit, không dùng để chọn model.
      </div>
    </div>
    <div>
      <div class="note-section"><b>Feature Options</b>
        <u>Option 1 (Hour)</u>: Chỉ dùng giờ xuất phát (departure_hour_numeric)<br>
        <u>Option 2 (Full)</u>: Giờ + loại VD + dịch vụ + ngày trong tuần + giờ tạo đơn + tỉnh + trọng lượng
      </div>
      <div class="note-section"><b>Metrics</b>
        <u>MAE</u>: Sai số trung bình (giờ) – nhỏ hơn = tốt hơn<br>
        <u>RMSE</u>: Nhạy với outlier (giờ) – nhỏ hơn = tốt hơn<br>
        <u>R²</u>: Mức độ giải thích biến động – càng gần 1 càng tốt (âm = tệ hơn đoán trung bình)<br>
        <u>MAPE</u>: Sai số phần trăm – nhỏ hơn = tốt hơn
      </div>
    </div>
  </div>
</div>
"""

    # ══ Tổng quan (Weighted) + Phân nhóm (Pattern) + Chẩn đoán nguyên nhân ══
    pair_df = _build_pair_level_df(results_by_flow)
    cs_pair_counts = {}

    if not pair_df.empty:
        valid = pair_df[pair_df['test_MAE'].notna()].copy()

        # Đếm số cặp kho theo chiến lược chọn cụm (best_combo) - để phân biệt
        # "Cách 0 (Không phân cụm)" chỉ là số ít so với Cách 1/2/3
        for flow_val, g in pair_df.groupby('flow'):
            n_single = int(g['is_single_group'].sum())
            n_multi  = int((~g['is_single_group']).sum())
            cs_pair_counts[flow_val] = {
                "Cách 0 (Không phân cụm)":     n_single,
                "Cách 1 (Dominant Cluster)":   n_multi,
                "Cách 2 (ML Classifier)":      n_multi,
                "Cách 3 (Hybrid)":             n_multi,
            }

        # ── 1. TỔNG QUAN HỆ THỐNG (tách theo từng luồng) ──
        # "Weighted" = Micro-average (gộp mọi bill lại rồi tính 1 MAE chung - đúng
        # bản chất khi triển khai thật). "Simple" = Macro-average (mỗi cặp kho vai
        # trò ngang nhau, bất kể ít/nhiều bill) - đặt tên theo đúng khái niệm
        # micro/macro-average quen thuộc (như micro/macro F1) để dễ theo dõi.
        flow_overview_cards = ""
        for flow_val in sorted(pair_df['flow'].unique()):
            pdf = pair_df[pair_df['flow'] == flow_val]
            vdf = valid[valid['flow'] == flow_val] if not valid.empty else valid
            total_bills_f   = int(pdf['n_bills'].sum())
            total_pairs_f   = len(pdf)
            n_single_f      = int(pdf['is_single_group'].sum())
            micro_mae_f     = _weighted_avg(vdf, 'test_MAE', 'n_test')
            micro_mape_f    = _weighted_avg(vdf, 'test_MAPE', 'n_test')
            macro_mae_f     = vdf['test_MAE'].mean()  if not vdf.empty else np.nan
            macro_mape_f    = vdf['test_MAPE'].mean() if not vdf.empty else np.nan
            gap_f = abs(micro_mae_f - macro_mae_f) if pd.notna(micro_mae_f) and pd.notna(macro_mae_f) else np.nan
            gap_note_f = "(có cặp kho outlier kéo lệch)" if pd.notna(gap_f) and gap_f > 0.3 else "(2 số gần nhau, ít outlier)"

            flow_overview_cards += f"""
      <div style="flex:1; min-width:450px;">
        <h3 style="font-size:14px; font-weight:700; color:#1e293b; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #e2e8f0;">{flow_val}</h3>
        <div class="kpi-grid" style="flex-wrap:wrap;">
          <div class="kpi-card"><div class="kpi-label">MAE (Micro - Weighted)</div><div class="kpi-value">{micro_mae_f:.2f}h</div>
            <div class="kpi-sub">Gộp theo N test mỗi cặp - đúng số khi triển khai thật</div></div>
          <div class="kpi-card"><div class="kpi-label">MAE (Macro - đơn giản)</div><div class="kpi-value">{macro_mae_f:.2f}h</div>
            <div class="kpi-sub">Mỗi cặp kho vai trò ngang nhau. Chênh với Micro: {gap_f:.2f}h {gap_note_f}</div></div>
          <div class="kpi-card"><div class="kpi-label">MAPE (Macro - đơn giản)</div><div class="kpi-value">{macro_mape_f:.1f}%</div>
            <div class="kpi-sub">Mỗi cặp kho vai trò ngang nhau</div></div>
          <div class="kpi-card"><div class="kpi-label">MAPE (Micro - Weighted)</div><div class="kpi-value">{micro_mape_f:.1f}%</div>
            <div class="kpi-sub">Gộp theo N test mỗi cặp</div></div>
          <div class="kpi-card"><div class="kpi-label">Tổng số bill</div><div class="kpi-value">{total_bills_f:,}</div>
            <div class="kpi-sub">{total_pairs_f} cặp kho</div></div>
          <div class="kpi-card"><div class="kpi-label">Cặp Single Group</div><div class="kpi-value">{n_single_f}/{total_pairs_f}</div>
            <div class="kpi-sub">Không tách được cụm bằng DBSCAN</div></div>
        </div>
      </div>"""

        html += f"""
<div class="summary-card">
  <h2>1. Tổng quan hệ thống <span style="font-size:11px;font-weight:400;color:#64748b;">(dựa trên Best Combo mỗi cặp kho, Test set)</span></h2>
  <div style="display:flex; gap:24px; flex-wrap:wrap;">{flow_overview_cards}
  </div>
</div>
"""

        # ── 2. PHÂN NHÓM (Pattern): theo Tỉnh đối tác / Kho 1A / Volume-tier / Cluster-mode ──
        def _group_table(group_col, label, top_n=None):
            if valid.empty or group_col not in valid.columns:
                return ""
            grp = valid.groupby(group_col).apply(lambda g: pd.Series({
                "n_pairs":  g['pair'].nunique(),
                "n_bills":  int(g['n_bills'].sum()),
                "w_mae":    _weighted_avg(g, 'test_MAE', 'n_test'),
                "w_mape":   _weighted_avg(g, 'test_MAPE', 'n_test'),
            })).reset_index().sort_values("w_mae", ascending=False)
            if top_n:
                grp = grp.head(top_n)
            rows = ""
            for _, r in grp.iterrows():
                rows += (f"<tr><td><b>{r[group_col]}</b></td><td>{int(r['n_pairs'])}</td>"
                          f"<td>{int(r['n_bills']):,}</td>"
                          f"<td><b>{r['w_mae']:.2f}h</b></td><td>{r['w_mape']:.1f}%</td></tr>")
            return f"""
        <div class="avg-table-wrap">
          <h4 style="margin:5px 0;">{label} <span style="font-size:11px;color:#94a3b8;">(sắp theo MAE Micro, tệ nhất trên đầu)</span></h4>
          <table>
            <thead><tr><th>{label}</th><th>Số cặp kho</th><th>Tổng bill</th><th>MAE (Micro)</th><th>MAPE (Micro)</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

        group_tables = "".join([
            _group_table("prov_partner",       "Tỉnh đối tác (gửi/nhận)"),
            _group_table("kho_1a",             "Kho 1A trung chuyển"),
            _group_table("volume_tier",        "Nhóm theo lưu lượng"),
            _group_table("cluster_selection",  "Chiến lược chọn cụm (Best Combo)"),
        ])

        html += f"""
<div class="avg-card">
  <h2>2. Phân nhóm theo Pattern <span style="font-size:11px;font-weight:400;color:#64748b;">(chỗ nào tốt/tệ, dựa trên Best Combo)</span></h2>
  <div class="avg-row-container">
    <div class="group-grid-2x2">{group_tables}</div>
  </div>
</div>
"""

        # ── 3. CHI TIẾT THEO CẶP KHO (tách riêng bảng theo từng luồng) ──
        detail_tables_html = ""
        for flow_val in sorted(pair_df['flow'].unique()):
            df_flow = pair_df[pair_df['flow'] == flow_val].sort_values('n_bills', ascending=False)
            rows_html = ""
            for _, r in df_flow.iterrows():
                mae_disp  = "N/A" if pd.isna(r['test_MAE'])  else f"{r['test_MAE']:.2f}h"
                mape_disp = "N/A" if pd.isna(r['test_MAPE']) else f"{r['test_MAPE']:.1f}%"
                best_combo_info = f"{r.get('cluster_selection', 'N/A')} | {r.get('feature_option', 'N/A')} | {r.get('regression_model', 'N/A')}"
                rows_html += f"""<tr>
                    <td><b>{r['pair']}</b></td>
                    <td>{int(r['n_bills']):,}</td>
                    <td>{best_combo_info}</td>
                    <td>{mae_disp}</td><td>{mape_disp}</td>
                </tr>"""
            detail_tables_html += f"""
    <div class="avg-table-wrap" style="flex: 1; min-width: 450px;">
      <h4 style="margin: 5px 0;">{flow_val}</h4>
      <table>
        <thead><tr><th>Cặp kho</th><th>Tổng bill</th><th>Best Combo</th><th>Test MAE</th><th>Test MAPE</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

        html += f"""
<div class="summary-card">
  <h2>3. Chi tiết theo cặp kho (Best Combo, sắp theo lưu lượng)</h2>
  <div class="avg-row-container">
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">{detail_tables_html}</div>
  </div>
</div>
"""

    # ── Summary Table + Avg Tables ──
    if not summary_df.empty:
        # Lọc bỏ outlier cực đoan trước khi tính avg (|MAE| > 1000 coi là artifact)
        df_valid = summary_df[summary_df['MAE'].abs() < 1000].copy()

        def _avg_table(df, group_col, label, idx, cs_pair_counts=None):
            """Tạo HTML gồm 2 bảng (1 cho mỗi luồng) trung bình metrics."""
            cols_to_group = []
            if group_col != "data_split" and "data_split" in df.columns:
                cols_to_group.append("data_split")
            if group_col not in cols_to_group:
                cols_to_group.append(group_col)
                
            header_map = {
                "data_split": "Split",
                "cluster_selection": "Chọn Cụm", 
                "feature_option": "Features", 
                "regression_model": "Model",
                "flow": "Luồng"
            }
            
            flows = sorted(df['flow'].unique())
            tables_html = ""
            for flow_val in flows:
                df_flow = df[df['flow'] == flow_val]
                if df_flow.empty: continue
                
                grp = df_flow.groupby(cols_to_group)[['MAE', 'RMSE', 'R2', 'MAPE']].mean().reset_index()
                # Sort để dữ liệu nhóm liền kề nhau dễ nhìn
                grp = grp.sort_values(cols_to_group).reset_index(drop=True)
                
                th_html = "".join([f"<th>{header_map.get(col, col)}</th>" for col in cols_to_group])
                rows = ""
                def _fmt(v, fmt=".3f", suffix=""):
                    return f"{v:{fmt}}{suffix}" if pd.notna(v) else "N/A"

                for row_idx, r in grp.iterrows():
                    row_num = row_idx + 1
                    td_html = ""
                    for col in cols_to_group:
                        val = r[col]
                        if col == group_col:
                            extra = ""
                            if group_col == "cluster_selection" and cs_pair_counts is not None:
                                cnt = cs_pair_counts.get(flow_val, {}).get(val)
                                if cnt is not None:
                                    extra = f' <span style="color:#94a3b8;font-weight:400;">(n={cnt} cặp)</span>'
                            td_html += f"<td><b>{val}</b>{extra}</td>"
                        else:
                            td_html += f"<td>{val}</td>"
                    rows += (f"<tr><td style='color:#64748b;font-weight:600;'>#{row_num}</td>{td_html}"
                             f"<td>{_fmt(r['MAE'])}</td><td>{_fmt(r['RMSE'])}</td>"
                             f"<td>{_fmt(r['R2'], '.4f')}</td><td>{_fmt(r['MAPE'], '.1f', '%')}</td></tr>")

                tables_html += (
                    '<div class="avg-table-wrap" style="flex: 1; min-width: 300px;">'
                    f'<h4 style="margin: 5px 0;">{flow_val}</h4>'
                    '<table>'
                    f'<thead><tr><th>#</th>{th_html}<th>Avg MAE</th><th>Avg RMSE</th><th>Avg R²</th><th>Avg MAPE</th></tr></thead>'
                    f'<tbody>{rows}</tbody>'
                    '</table></div>'
                )

            return f'<div class="avg-row-container" style="margin-bottom: 20px;"><div style="width: 100%;"><h3>{label}</h3><div style="display: flex; gap: 20px; flex-wrap: wrap;">{tables_html}</div></div></div>'

        avg_html = ""
        for i, (col, lbl) in enumerate([
            ("flow",               "Theo Luồng"),
            ("cluster_selection",  "Theo Chiến lược chọn cụm"),
            ("feature_option",     "Theo Feature Option"),
            ("regression_model",   "Theo Regression Model"),
        ]):
            if col in df_valid.columns:
                _counts_arg = cs_pair_counts if col == "cluster_selection" else None
                avg_html += _avg_table(df_valid, col, lbl, i, cs_pair_counts=_counts_arg)

        html += f"""
<div class="avg-card">
  <h2>Trung bình hiệu suất theo từng nhóm
    <span style="font-size:11px;font-weight:400;color:#64748b;"> (loại bỏ outlier MAE &gt; 1000h)</span>
  </h2>
  <div class="avg-grid">{avg_html}</div>
</div>
"""

        # ── Bảng tổng hợp tổ hợp chiến lược ──
        combo_cols_no_flow = ['data_split', 'cluster_selection', 'feature_option', 'regression_model']
        combo_cols_no_flow = [c for c in combo_cols_no_flow if c in df_valid.columns]
        
        flows = sorted(df_valid['flow'].unique())
        combo_tables_html = ""
        for flow_val in flows:
            df_flow = df_valid[df_valid['flow'] == flow_val]
            if df_flow.empty: continue
            
            combo_grp = df_flow.groupby(combo_cols_no_flow)[['MAE', 'RMSE', 'R2', 'MAPE']].mean().reset_index()
            combo_grp = combo_grp.sort_values(combo_cols_no_flow)
            
            combo_rows = ""
            def _fmt_combo(v, fmt=".3f", suffix=""):
                return f"{v:{fmt}}{suffix}" if pd.notna(v) else "N/A"
            for _, r in combo_grp.iterrows():
                split_td = f"<td>{r['data_split']}</td>" if 'data_split' in combo_cols_no_flow else ""
                cs_val = r['cluster_selection']
                cs_extra = ""
                _cnt = cs_pair_counts.get(flow_val, {}).get(cs_val)
                if _cnt is not None:
                    cs_extra = f' <span style="color:#94a3b8;font-weight:400;">(n={_cnt})</span>'
                combo_rows += (f"<tr>{split_td}<td>{cs_val}{cs_extra}</td>"
                               f"<td>{r['feature_option']}</td><td>{r['regression_model']}</td>"
                               f"<td><b>{_fmt_combo(r['MAE'])}</b></td>"
                               f"<td>{_fmt_combo(r['RMSE'])}</td><td>{_fmt_combo(r['R2'], '.4f')}</td>"
                               f"<td>{_fmt_combo(r['MAPE'], '.1f', '%')}</td></tr>")

            _has_split_combo = 'data_split' in combo_cols_no_flow
            _j = (lambda base: base + 1 if _has_split_combo else base)
            tid = f"comboSummaryTable_{''.join(c for c in str(flow_val) if c.isalnum())}"
            split_th_combo = f'<th class="sortable" onclick="sortTable(0, \'{tid}\')">Split ↕</th>' if _has_split_combo else ''

            combo_tables_html += f"""
    <div class="avg-table-wrap" style="flex: 1; min-width: 450px;">
      <h4 style="margin: 5px 0;">{flow_val}</h4>
      <table id="{tid}">
        <thead><tr>
          {split_th_combo}
          <th class="sortable" onclick="sortTable({_j(0)}, '{tid}')">Chọn Cụm ↕</th>
          <th class="sortable" onclick="sortTable({_j(1)}, '{tid}')">Features ↕</th>
          <th class="sortable" onclick="sortTable({_j(2)}, '{tid}')">Regression ↕</th>
          <th class="sortable" onclick="sortTable({_j(3)}, '{tid}')">Avg MAE ↕</th>
          <th class="sortable" onclick="sortTable({_j(4)}, '{tid}')">Avg RMSE ↕</th>
          <th class="sortable" onclick="sortTable({_j(5)}, '{tid}')">Avg R² ↕</th>
          <th class="sortable" onclick="sortTable({_j(6)}, '{tid}')">Avg MAPE ↕</th>
        </tr></thead>
        <tbody>{combo_rows}</tbody>
      </table>
    </div>
"""

        html += f"""
<div class="summary-card">
  <h2>Trung bình toàn tổ hợp (Split → Cụm → Feature → Model)</h2>
  <div class="avg-row-container">
    <div style="display: flex; gap: 20px; flex-wrap: wrap; width: 100%;">
      {combo_tables_html}
    </div>
  </div>
</div>
"""

        summary_rows = ""
        for _, row in summary_df.sort_values("MAE", ascending=True).iterrows():
            mae_str  = f"{row['MAE']:.2f}"  if not pd.isna(row['MAE'])  else "N/A"
            rmse_str = f"{row['RMSE']:.2f}" if not pd.isna(row['RMSE']) else "N/A"
            r2_str   = f"{row['R2']:.4f}"   if not pd.isna(row['R2'])   else "N/A"
            mape_str = f"{row['MAPE']:.1f}%" if not pd.isna(row['MAPE']) else "N/A"
            split_td = f"<td>{row['data_split']}</td>" if 'data_split' in row.index else ""
            n_val = int(row['n_samples']) if 'n_samples' in row.index else int(row.get('n_samples_test', 0))
            summary_rows += f"""<tr>
                <td>{row['flow']}</td>
                {split_td}
                <td><b>{row['pair']}</b></td>
                <td>{row['cluster_selection']}</td>
                <td>{row['feature_option']}</td>
                <td>{row['regression_model']}</td>
                <td>{n_val:,}</td>
                <td><b>{mae_str}</b></td>
                <td>{rmse_str}</td>
                <td>{r2_str}</td>
                <td>{mape_str}</td>
            </tr>"""

        _has_split = 'data_split' in summary_df.columns
        _i = (lambda base: base + 1 if _has_split else base)

        split_th_summary = '<th class="sortable" onclick="sortTable(1, \'summaryTable\')">Split ↕</th>' if _has_split else ''

        html += f"""
<div class="summary-card">
  <h2>Bảng tổng hợp so sánh tất cả tổ hợp chiến lược</h2>
  <table id="summaryTable">
    <thead><tr>
      <th class="sortable" onclick="sortTable(0, 'summaryTable')">Luồng ↕</th>
      {split_th_summary}
      <th class="sortable" onclick="sortTable({_i(1)}, 'summaryTable')">Cặp Kho ↕</th>
      <th class="sortable" onclick="sortTable({_i(2)}, 'summaryTable')">Chọn Cụm ↕</th>
      <th class="sortable" onclick="sortTable({_i(3)}, 'summaryTable')">Features ↕</th>
      <th class="sortable" onclick="sortTable({_i(4)}, 'summaryTable')">Regression ↕</th>
      <th class="sortable" onclick="sortTable({_i(5)}, 'summaryTable')">N Test ↕</th>
      <th class="sortable" onclick="sortTable({_i(6)}, 'summaryTable')">MAE (h) ↕</th>
      <th class="sortable" onclick="sortTable({_i(7)}, 'summaryTable')">RMSE (h) ↕</th>
      <th class="sortable" onclick="sortTable({_i(8)}, 'summaryTable')">R² ↕</th>
      <th class="sortable" onclick="sortTable({_i(9)}, 'summaryTable')">MAPE ↕</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>
"""

    chart_idx = 0

    # ── Per-pair reports ──
    for flow_name, pair_reports in results_by_flow:
        html += f'<div class="flow-header"> {flow_name}</div>\n'

        for rep in pair_reports:
            pair = rep["pair"]
            val_scores = rep.get("val_scores", [])
            best_combo = rep.get("best_combo", {})
            c_stats = rep.get("cluster_stats", [])

            if not val_scores:
                continue

            # ── Cluster Stats Table ──
            cluster_rows = ""
            for cs in c_stats:
                cluster_rows += f"<tr><td><b>Cụm {cs['cluster']}</b></td><td>{cs['n']:,}</td><td>{cs['mean']}</td><td>{cs['median']}</td><td>{cs['std']}</td></tr>"

            # ── Combo Detail Table ──
            combo_rows = ""
            def _f(v, fmt=".2f"): return format(v, fmt) if not (isinstance(v, float) and (v != v)) else "N/A"
            for vs in val_scores:
                is_best = (vs['cs_name'] == best_combo.get('cs_name') and 
                           vs['fo_name'] == best_combo.get('fo_name') and 
                           vs['reg_name'] == best_combo.get('reg_name'))
                row_cls = "best-row" if is_best else ""
                combo_rows += f"""<tr class='{row_cls}'>
                    <td>{vs['cs_name']}{' ★' if is_best else ''}</td>
                    <td>{vs['fo_name']}</td>
                    <td>{vs['reg_name']}</td>
                    <td>{_f(vs['val_mae'])}</td><td>{_f(vs['val_rmse'])}</td>
                    <td style='color:#64748b;font-size:11px'>{vs['n_val']:,}</td>
                </tr>"""

            # ── Bar Chart: MAE comparison on Val ──
            colors_map = {"LR": "#cbd5e1", "RF": "#94a3b8", "LGBM": "#3b82f6"}
            fig = go.Figure()
            for reg_name in ["LR", "RF", "LGBM"]:
                x_vals = []
                y_vals = []
                for vs in val_scores:
                    if vs['reg_name'] == reg_name:
                        label = f"{vs['cs_name']}<br>{vs['fo_name']}"
                        x_vals.append(label)
                        y_vals.append(vs['val_mae'])
                if x_vals:
                    fig.add_trace(go.Bar(
                        x=x_vals, y=y_vals,
                        name=reg_name,
                        marker_color=colors_map.get(reg_name, "#888"),
                        hovertemplate=f"{reg_name}<br>%{{x}}<br>Val MAE: %{{y:.2f}}h<extra></extra>"
                    ))

            fig.update_layout(
                title="MAE trên tập Validation (để chọn Best Combo)",
                barmode="group",
                height=450, width=1100,
                plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
                font=dict(family="Segoe UI, Arial", size=10),
                margin=dict(t=60, b=120, l=50, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                yaxis_title="Val MAE (giờ)"
            )

            id_chart = f"chart_{chart_idx}"
            slot_info = " | ".join([f"{k}: {v}" for k, v in rep['slot_cluster_map'].items()])
            fig_json = fig.to_json()

            # ── Single Group (DBSCAN không tách được cụm) → hiển thị khác đi ──
            # Không show "Clf Acc"/"Hybrid ML" (không có classifier để đo) và không
            # gán nhãn "(Cách 1)" cứng cho dòng khung giờ, vì cặp kho này không đi
            # qua layer chọn cụm/classifier.
            is_single_group = rep.get('n_clusters', 0) < 2
            if is_single_group:
                clf_badges = '<span class="meta-badge">Không phân cụm (single-group)</span>'
                slot_line = ('<p class="section-title"> DBSCAN không tách được cụm cho cặp kho này '
                              '→ toàn bộ trainval được xử lý như 1 nhóm dữ liệu duy nhất</p>')
            else:
                clf_badges = (f'<span class="meta-badge">Clf Acc={rep.get("clf_acc", 0):.3f}</span>'
                              f'<span class="meta-badge">Hybrid ML: {rep.get("hybrid_ml_rate", 0)*100:.1f}%</span>')
                slot_label = best_combo.get('cs_name', 'Cách 1 (Dominant Cluster)')
                slot_line = f'<p class="section-title"> Khung giờ → Cụm ({slot_label}): {slot_info}</p>'

            # ── Final Test Metrics display ──
            tm = rep.get('test_metrics', {})
            trm = rep.get('train_metrics', {})
            
            html += f"""
<div class="pair-card">
  <div class="pair-header">
    <span class="pair-title">{pair}</span>
    <div class="pair-meta">
      <span class="meta-badge">{rep.get('n_samples', 0):,} trainval samples</span>
      <span class="meta-badge">{rep.get('n_test', 0):,} test samples</span>
      <span class="meta-badge">{rep.get('n_clusters', 0)} cụm</span>
      <span class="meta-badge strong">Dominant: Cụm {rep.get('dominant_cluster', -1)}</span>
      {clf_badges}
    </div>
  </div>
  <div class="pair-body">
    <div class="test-box">
      <h3>Kết quả Final Test (chỉ chạy 1 lần) trên {rep.get('n_test', 0):,} bills</h3>
      <p><b>Best Combo:</b> {best_combo.get('cs_name')} | {best_combo.get('fo_name')} | {best_combo.get('reg_name')}</p>
      <div style="display: flex; gap: 40px; margin-top: 10px;">
        <div>
          <b>Test Metrics:</b> MAE={_f(tm.get('MAE'))}h, RMSE={_f(tm.get('RMSE'))}h, R²={_f(tm.get('R2'), '.4f')}, MAPE={_f(tm.get('MAPE'), '.1f')}%
        </div>
        <div>
          <b>Train(Diag) Metrics:</b> MAE={_f(trm.get('MAE'))}h, RMSE={_f(trm.get('RMSE'))}h, R²={_f(trm.get('R2'), '.4f')}, MAPE={_f(trm.get('MAPE'), '.1f')}%
        </div>
      </div>
    </div>
    
    {slot_line}

    <div class="grid-2">
      <div>
        <p class="section-title">Thống kê Cụm</p>
        <table>
          <thead><tr><th>Cụm</th><th>Số bill</th><th>Mean</th><th>Median</th><th>Std</th></tr></thead>
          <tbody>{cluster_rows}</tbody>
        </table>
      </div>
      <div id="{id_chart}" class="chart-container"></div>
    </div>

    <p class="section-title"> Chi tiết Val Selection (Tuning)</p>
    <table>
      <thead><tr>
        <th>Chọn Cụm</th><th>Features</th><th>Regression</th>
        <th>Val MAE</th><th>Val RMSE</th>
        <th>N Val</th>
      </tr></thead>
      <tbody>{combo_rows}</tbody>
    </table>
  </div>
</div>
<script>
var d_{chart_idx} = {fig_json};
Plotly.newPlot('{id_chart}', d_{chart_idx}.data, d_{chart_idx}.layout, {{responsive: true}});
</script>
"""
            chart_idx += 1

    # Sort table script
    html += """
<script>
function sortTable(n, tableId) {
  if (!tableId) tableId = "summaryTable";
  var table = document.getElementById(tableId);
  if (!table) return;
  var tbody = table.querySelector("tbody");
  var rows = Array.from(tbody.querySelectorAll("tr"));

  // Toggle direction: nếu click cùng cột thì đảo chiều
  var prevCol = parseInt(table.dataset.sortCol ?? "-1");
  var prevDir = table.dataset.sortDir || "asc";
  var dir = (prevCol === n && prevDir === "asc") ? "desc" : "asc";
  table.dataset.sortDir = dir;
  table.dataset.sortCol = n;

  // Cập nhật icon trên header
  var ths = table.querySelectorAll("thead th");
  ths.forEach(function(th, idx) {
    var txt = th.textContent.replace(/ [\u2191\u2193\u2195]/g, '').trim();
    th.textContent = txt + (idx === n ? (dir === "asc" ? " \u2191" : " \u2193") : " \u2195");
  });

  rows.sort(function(rowA, rowB) {
    var cellA = rowA.querySelectorAll("td")[n];
    var cellB = rowB.querySelectorAll("td")[n];
    if (!cellA || !cellB) return 0;
    var a = (cellA.innerText || cellA.textContent).replace(/,/g, '').replace(/%/g, '').trim();
    var b = (cellB.innerText || cellB.textContent).replace(/,/g, '').replace(/%/g, '').trim();
    var aNum = parseFloat(a);
    var bNum = parseFloat(b);
    var cmp = (!isNaN(aNum) && !isNaN(bNum)) ? (aNum - bNum) : a.toLowerCase().localeCompare(b.toLowerCase());
    return dir === "asc" ? cmp : -cmp;
  });

  rows.forEach(function(row) { tbody.appendChild(row); });
}
</script>
</body></html>"""
    return html