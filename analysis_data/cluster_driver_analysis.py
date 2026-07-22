"""
Phương pháp:
  - Chi-squared + Cramér's V cho biến phân loại
  - Kruskal-Wallis + η² cho biến liên tục

Output:
  - Console: Bảng tổng hợp cho từng cặp kho
  - CSV:     output_plot/cluster_drivers_summary.csv
  - HTML:    output_plot/cluster_drivers.html (stacked bar charts)
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Cấu hình ────────────────────────────────────────────────────────────
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_all_traces")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_plot")
N_TOP = 10
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 150

EXCLUDED_KHO = [
    "Kho TGDĐ Bảo Hành", "Kho Hà Tĩnh", "Kho Kiến An", "Kho Thường Tín",
    "Kho Amway", "Kho Digiworld", "Kho Thái Nguyên", "Kho Việt Trì",
    "Kho Ninh Hòa", "Kho An Giang",
    "Bưu cục Dự Án Vgreen Sóng Thần", "Bưu cục Dự Án Vgreen Văn Giang"
]


# ── Helpers ──────────────────────────────────────────────────────────────
def find_optimal_shift(time_series):
    """Tính điểm cắt (thung lũng) để shift trục giờ tránh đứt đoạn"""
    if time_series.empty:
        return 0
    hours = pd.to_datetime(time_series).dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0)
    extended = pd.concat([counts.iloc[-1:], counts, counts.iloc[:1]])
    rolling = extended.rolling(3, center=True).mean().iloc[1:-1]
    return int(rolling.idxmin())


def cramers_v(contingency_table):
    """Tính Cramér's V từ bảng contingency"""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def kruskal_eta_squared(groups):
    """Chạy Kruskal-Wallis và tính η² (effect size)"""
    if len(groups) < 2:
        return np.nan, np.nan
    # Lọc bỏ các nhóm có < 5 mẫu
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) < 2:
        return np.nan, np.nan
    stat, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = (stat - k + 1) / (n - k) if n > k else 0.0
    return p, max(0, eta2)


def interpret_effect(v):
    """Diễn giải mức độ ảnh hưởng"""
    if pd.isna(v):
        return "N/A"
    if v < 0.1:
        return "Rất yếu"
    if v < 0.3:
        return "Yếu"
    if v < 0.5:
        return "Trung bình"
    return "Mạnh"


# ── Hàm chính ───────────────────────────────────────────────────────────
def get_top_pairs(df, col_a, col_b, sort_by, wt_col, is_dest_flow):
    """Lấy top N cặp kho"""
    df = df.copy()
    if is_dest_flow:
        df = df[~df[col_b].isin(EXCLUDED_KHO)]
    else:
        df = df[~df[col_a].isin(EXCLUDED_KHO)]
    df["pair"] = df[col_a].astype(str) + " → " + df[col_b].astype(str)

    check_col = col_b if is_dest_flow else col_a
    key = "so_bill" if sort_by == "bill_count" else "tong_kg"

    agg = (df.groupby(["pair", check_col])
             .agg(so_bill=("bill_code", "nunique"), tong_kg=(wt_col, "sum"))
             .reset_index())
    top = agg.nlargest(N_TOP, key)
    return df, top["pair"].tolist()



def analyze_flow(df, pairs, t_col_1, t_col_2, flow_name,
                 cat_features, cont_features):

    all_results = []
    flow_pairs = []

    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"  CẶP KHO: {pair}")
        print(f"{'='*70}")

        sub = df.loc[df["pair"] == pair, :].copy()

        sub = sub.dropna(subset=[t_col_1, t_col_2]).copy()

        # Lọc time ngoại lai
        if "time" in sub.columns:
            sub = sub[(sub["time"] >= 0) & (sub["time"] <= 100)].copy()
            if not sub.empty:
                q1 = sub["time"].quantile(0.01)
                q3 = sub["time"].quantile(0.99)
                sub = sub[(sub["time"] >= q1) & (sub["time"] <= q3)].copy()

        if sub.empty or len(sub) < 200:
            print(f"  Không đủ dữ liệu ({len(sub)} bills). Bỏ qua.")
            continue

        # Tính Y (offset liên tục)
        dt1 = pd.to_datetime(sub[t_col_1])
        dt2 = pd.to_datetime(sub[t_col_2])
        departure_date = dt1.dt.normalize()
        sub["y_offset"] = (dt2 - departure_date).dt.total_seconds() / 3600.0
        sub['departure_hour'] = dt1.dt.hour.fillna(-1).astype(int).astype(str) + "h"
        sub.loc[sub['departure_hour'] == "-1h", 'departure_hour'] = "N/A"
        sub = sub[(sub["y_offset"] >= 0) & (sub["y_offset"] <= 48)].copy()

        if sub.empty or len(sub) < 200:
            print(f"  Không đủ dữ liệu sau lọc ({len(sub)} bills). Bỏ qua.")
            continue

        if len(sub) > 20000:
            sub = sub.sample(20000, random_state=42)

        # ══ Chạy DBSCAN (1 LẦN DUY NHẤT) ══
        y_vals = sub["y_offset"].values.reshape(-1, 1)
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        sub["cluster"] = dbscan.fit_predict(y_vals)

        # Sắp xếp lại tên cụm theo y_offset tăng dần
        valid_mask = sub["cluster"] != -1
        if valid_mask.any():
            cluster_means = sub.loc[valid_mask].groupby("cluster")["y_offset"].mean()
            sorted_clusters = cluster_means.sort_values().index.tolist()
            cluster_mapping = {old_c: new_c for new_c, old_c in enumerate(sorted_clusters)}
            cluster_mapping[-1] = -1
            sub["cluster"] = sub["cluster"].map(cluster_mapping)

        # Loại bỏ nhiễu
        sub_clean = sub[sub["cluster"] != -1].copy()
        n_clusters = sub_clean["cluster"].nunique()
        n_noise = (sub["cluster"] == -1).sum()

        print(f"  Tổng bills: {len(sub):,}")
        print(f"  Số cụm: {n_clusters} | Nhiễu: {n_noise:,} ({n_noise/len(sub)*100:.1f}%)")

        if n_clusters < 2:
            print(f" Chỉ có {n_clusters} cụm, không đủ để kiểm định. Bỏ qua.")
            continue

        # ── Thống kê cụm ──
        cluster_stats = []
        print(f"\n  Thống kê giờ đến (Y offset) theo cụm:")
        for c in sorted(sub_clean["cluster"].unique()):
            c_data = sub_clean[sub_clean["cluster"] == c]["y_offset"]
            print(f"    Cụm {c}: n={len(c_data):,}, "
                  f"mean={c_data.mean():.1f}h, "
                  f"median={c_data.median():.1f}h, "
                  f"std={c_data.std():.1f}h")
            cluster_stats.append({
                "cluster": c, "n": len(c_data),
                "mean": f"{c_data.mean():.1f}h",
                "median": f"{c_data.median():.1f}h",
                "std": f"{c_data.std():.1f}h"
            })

        # ── Kiểm định từng yếu tố (1 LẦN DUY NHẤT) ──
        print(f"\n  {'─'*50}")
        print(f"  KẾT QUẢ KIỂM ĐỊNH THỐNG KÊ")
        print(f"  {'─'*50}")
        print(f"  {'Yếu tố':<25} {'Phương pháp':<18} {'p-value':<12} {'Effect Size':<12} {'Mức độ':<12}")
        print(f"  {'─'*50}")

        test_results = []

        for feat in cat_features:
            if feat not in sub_clean.columns or sub_clean[feat].isna().all():
                continue
            ct = pd.crosstab(sub_clean[feat], sub_clean["cluster"])
            # Loại bỏ các hàng có tổng < 5
            ct = ct[ct.sum(axis=1) >= 5]
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            cv = cramers_v(ct)
            effect = interpret_effect(cv)
            print(f"  {feat:<25} {'Chi-squared':<18} {p:<12.2e} {cv:<12.4f} {effect:<12}")

            # Push vào cả 2 list
            all_results.append({
                "flow": flow_name, "pair": pair, "feature": feat,
                "test": "Chi-squared", "p_value": p, "effect_size": cv,
                "interpretation": effect, "n_clusters": n_clusters,
                "n_samples": len(sub_clean)
            })
            test_results.append({
                "feature": feat, "test": "Chi-squared",
                "p": p, "effect": cv, "level": effect
            })

        for feat in cont_features:
            if feat not in sub_clean.columns or sub_clean[feat].isna().all():
                continue
            groups = [
                sub_clean.loc[sub_clean["cluster"] == c, feat].dropna().values
                for c in sorted(sub_clean["cluster"].unique())
            ]
            p, eta2 = kruskal_eta_squared(groups)
            effect = interpret_effect(eta2)
            print(f"  {feat:<25} {'Kruskal-Wallis':<18} {p:<12.2e} {eta2:<12.4f} {effect:<12}")

            all_results.append({
                "flow": flow_name, "pair": pair, "feature": feat,
                "test": "Kruskal-Wallis", "p_value": p, "effect_size": eta2,
                "interpretation": effect, "n_clusters": n_clusters,
                "n_samples": len(sub_clean)
            })
            test_results.append({
                "feature": feat, "test": "Kruskal-Wallis",
                "p": p, "effect": eta2, "level": effect
            })

        # ── Phân tích chi tiết: Tỷ lệ phân bố theo cụm ──
        for feat in cat_features:
            if feat not in sub_clean.columns or sub_clean[feat].isna().all():
                continue
            ct = pd.crosstab(sub_clean["cluster"], sub_clean[feat], normalize="index") * 100
            if ct.shape[1] > 0:
                print(f"\n  Tỷ lệ {feat} trong mỗi cụm (%):")
                top_cats = sub_clean[feat].value_counts().head(5).index.tolist()
                display_cols = [c for c in top_cats if c in ct.columns]
                if display_cols:
                    ct_display = ct[display_cols].round(1)
                    for c_idx in ct_display.index:
                        vals = " | ".join([f"{col}: {ct_display.loc[c_idx, col]:.1f}%" for col in display_cols])
                        print(f"    Cụm {c_idx}: {vals}")

        # ══ Tạo figure Plotly (dùng CHUNG sub_clean từ DBSCAN ở trên) ══
        subplot_titles = ["Phân cụm dữ liệu (DBSCAN)",
                          "VD_type (Loại đơn)", "Service (Dịch vụ)",
                          "Day of Week (Ngày trong tuần)", "Creation Hour (Giờ tạo đơn)",
                          "Origin Province (Tỉnh gửi)", "Destination Province (Tỉnh nhận)",
                          "Weight (Khối lượng)"]
        fig = make_subplots(
            rows=5, cols=2, subplot_titles=subplot_titles,
            row_heights=[0.32, 0.17, 0.17, 0.17, 0.17],
            horizontal_spacing=0.12, vertical_spacing=0.08,
            specs=[[{"type": "xy"}, None],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "box"}, None]]
        )

        colors = [
            "#2563EB", "#E11D48", "#10B981", "#F59E0B", "#8B5CF6",
            "#06B6D4", "#F97316", "#EC4899", "#14B8A6", "#84CC16",
            "#6366F1", "#EF4444", "#A855F7", "#0EA5E9", "#D946EF"
        ]
        cluster_labels = [f"Cụm {c}" for c in sorted(sub_clean["cluster"].unique())]

        # --- THÊM SCATTER PLOT (DBSCAN) VÀO HÀNG 1 ---
        hour_shift = find_optimal_shift(sub_clean[t_col_1])
        dt1_clean = pd.to_datetime(sub_clean[t_col_1])
        raw_hours = dt1_clean.dt.hour + dt1_clean.dt.minute / 60.0 + dt1_clean.dt.second / 3600.0
        x_vals = (raw_hours - hour_shift) % 24
        
        cluster_colors = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6"]
        
        for clus in sorted(sub_clean["cluster"].unique()):
            mask = sub_clean["cluster"] == clus
            c_color = cluster_colors[clus % len(cluster_colors)]
            c_name = f"Cụm {clus}"
            
            fig.add_trace(go.Scatter(
                x=x_vals[mask].tolist(),
                y=sub_clean.loc[mask, "y_offset"].tolist(),
                customdata=raw_hours[mask].tolist(),
                mode='markers',
                marker=dict(size=4, color=c_color, opacity=0.6),
                name=c_name,
                showlegend=True,
                hovertemplate=(
                    f"Giờ xuất phát: %{{customdata:.2f}}h<br>"
                    f"Giờ đến: %{{y:.1f}}h<br>"
                    f"Nhóm: {c_name}<extra></extra>"
                )
            ), row=1, col=1)
            
        tv_x2 = list(range(0, 25, 2))
        tt_x2 = [f"{int((v + hour_shift) % 24)}h" for v in tv_x2]
        fig.update_xaxes(title_text="Giờ xuất phát", range=[0, 24], tickvals=tv_x2, ticktext=tt_x2, showgrid=True, gridcolor="#E5E7EB", row=1, col=1)
        fig.update_yaxes(title_text="Giờ đến (Offset từ ngày gửi)", showgrid=True, gridcolor="#E5E7EB", row=1, col=1)
        # ---------------------------------------------

        # Helper: vẽ stacked bar cho 1 biến phân loại
        def add_stacked_bar(feat, row, col, legend_group_prefix):
            if feat not in sub_clean.columns or sub_clean[feat].isna().all():
                return
            ct = pd.crosstab(sub_clean["cluster"], sub_clean[feat], normalize="index") * 100
            top_cats = sub_clean[feat].value_counts().head(8).index.tolist()
            other_cols = [c for c in ct.columns if c not in top_cats]
            if other_cols:
                ct["Khác"] = ct[other_cols].sum(axis=1)
            display_cats = [c for c in top_cats if c in ct.columns] + (["Khác"] if other_cols else [])

            for cat_idx, cat in enumerate(display_cats):
                if cat not in ct.columns:
                    continue
                fig.add_trace(go.Bar(
                    x=cluster_labels,
                    y=ct[cat].values.tolist(),
                    name=str(cat),
                    marker_color=colors[cat_idx % len(colors)],
                    showlegend=False,
                    legendgroup=f"{legend_group_prefix}_{cat}",
                    hovertemplate=f"<b>{cat}</b>: %{{y:.1f}}%<extra></extra>"
                ), row=row, col=col)
            fig.update_yaxes(title_text="Tỷ lệ (%)", range=[0, 100], row=row, col=col)

        # Vẽ 6 stacked bar
        add_stacked_bar("VD_type", 2, 1, "vd")
        add_stacked_bar("service", 2, 2, "sv")
        add_stacked_bar("day_of_week", 3, 1, "dw")
        add_stacked_bar("creation_hour", 3, 2, "ch")
        add_stacked_bar("origin_province", 4, 1, "op")
        add_stacked_bar("destination_province", 4, 2, "dp")

        # Vẽ boxplot cho weight
        if "actual_weight" in sub_clean.columns:
            box_colors = ["#2563EB", "#E11D48", "#10B981", "#F59E0B", "#8B5CF6",
                          "#06B6D4", "#F97316", "#EC4899"]
            for c_idx, c in enumerate(sorted(sub_clean["cluster"].unique())):
                w_data = sub_clean.loc[sub_clean["cluster"] == c, "actual_weight"].dropna()
                # Cap tại percentile 95 để boxplot không bị nén
                cap = w_data.quantile(0.95)
                w_data = w_data[w_data <= cap]
                fig.add_trace(go.Box(
                    y=w_data.tolist(),
                    name=f"Cụm {c}",
                    marker_color=box_colors[c_idx % len(box_colors)],
                    boxmean=True,
                    showlegend=False,
                    hovertemplate=f"Cụm {c}<br>Weight: %{{y:.1f}} kg<extra></extra>"
                ), row=5, col=1)
            fig.update_yaxes(title_text="Khối lượng (kg)", row=5, col=1)

        fig.update_layout(
            barmode="stack",
            height=1400, width=1100,
            plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
            font=dict(family="Segoe UI, Arial", size=10),
            margin=dict(t=80, b=30, l=50, r=20),
            showlegend=True,
            legend=dict(
                title=dict(text="Cụm dữ liệu", font=dict(size=12)),
                font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#D1D5DB", borderwidth=1,
                orientation="v", yanchor="top", y=1.0, xanchor="left", x=0.6
            )
        )

        flow_pairs.append({
            "pair": pair, "fig": fig, "cluster_stats": cluster_stats,
            "test_results": test_results, "n_total": len(sub),
            "n_noise": n_noise, "n_clusters": n_clusters
        })

    return all_results, flow_pairs

def build_full_html(results_by_flow, all_test_results):
    html = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Cluster Driver Analysis</title>
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
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #f8fafc; padding: 10px 12px; text-align: left; font-weight: 600;
       border-bottom: 2px solid #e2e8f0; color: #475569; }
  td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
  tr:hover td { background: #f8fafc; }
  .pair-card { background: white; border-radius: 10px; margin-bottom: 20px; overflow: hidden;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .pair-header { background: #f8fafc; padding: 14px 20px; border-bottom: 1px solid #e2e8f0;
                   display: flex; justify-content: space-between; align-items: center; }
  .pair-title { font-size: 15px; font-weight: 700; color: #1e293b; }
  .pair-meta { display: flex; gap: 16px; }
  .meta-badge { background: #eff6ff; color: #1d4ed8; padding: 4px 10px; border-radius: 6px;
                  font-size: 12px; font-weight: 600; }
  .meta-badge.warn { background: #fef3c7; color: #92400e; }
  .pair-body { padding: 16px 20px; }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .cluster-table { width: 100%; }
  .cluster-table th { font-size: 12px; }
  .cluster-table td { font-size: 12px; }
  .test-table { width: 100%; }
  .test-table th { font-size: 12px; }
  .test-table td { font-size: 12px; }
  .effect-bar { height: 14px; border-radius: 3px; display: inline-block; vertical-align: middle; }
  .chart-container { margin-top: 12px; }
  .legend-hint { font-size: 11px; color: #94a3b8; margin-top: 4px; text-align: center; }
</style>
</head><body>
<h1> Phân tích Yếu tố Hình thành Phân cụm</h1>
<p class="subtitle">Kiểm định thống kê xem yếu tố nào quyết định bill rơi vào cụm giờ đến nào</p>
"""
    import pandas as pd
    if all_test_results:
        df_summary = pd.DataFrame(all_test_results)
        flows = df_summary["flow"].unique()
        for flow in flows:
            df_flow = df_summary[df_summary["flow"] == flow]
            agg = df_flow.groupby("feature")["effect_size"].agg(["mean", "median", "max", "count"])
            agg = agg.sort_values("mean", ascending=False)
            summary_rows = ""
            for feat, row in agg.iterrows():
                level = interpret_effect(row["mean"])
                bar_width = min(row["mean"] * 300, 100)
                bar_color = "#EF4444" if row["mean"] >= 0.3 else "#F59E0B" if row["mean"] >= 0.1 else "#94A3B8"
                summary_rows += f"""<tr>
                    <td style="font-weight:600">{feat}</td>
                    <td>{row['mean']:.4f}</td>
                    <td>{row['median']:.4f}</td>
                    <td>{row['max']:.4f}</td>
                    <td>{int(row['count'])}</td>
                    <td>{level}</td>
                    <td><div style="background:{bar_color};height:16px;width:{bar_width}%;border-radius:3px"></div></td>
                </tr>"""

            html += f"""
<div class="summary-card">
  <h2> Xếp hạng Yếu tố Ảnh hưởng: {flow}</h2>
  <table>
    <thead><tr>
      <th>Yếu tố</th><th>Mean</th><th>Median</th><th>Max</th><th>Số cặp</th><th>Mức độ</th><th>Trực quan</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>
"""
    else:
        html += "<div class='summary-card'><p>Không có dữ liệu tổng hợp.</p></div>"

    chart_idx = 0
    for flow_name, pair_results in results_by_flow:
        html += f'<div class="flow-header"> {flow_name}</div>\n'

        for pr in pair_results:
            pair = pr["pair"]
            fig = pr["fig"]
            c_stats = pr["cluster_stats"]
            t_results = pr["test_results"]
            n_total = pr["n_total"]
            n_noise = pr["n_noise"]
            n_clusters = pr["n_clusters"]

            # Cluster stats table
            cluster_rows = ""
            for cs in c_stats:
                cluster_rows += f"<tr><td><b>Cụm {cs['cluster']}</b></td><td>{cs['n']:,}</td><td>{cs['mean']}</td><td>{cs['median']}</td><td>{cs['std']}</td></tr>"

            # Test results table
            test_rows = ""
            for tr in t_results:
                p_str = f"{tr['p']:.2e}" if not pd.isna(tr['p']) else "N/A"
                e_str = f"{tr['effect']:.4f}" if not pd.isna(tr['effect']) else "N/A"
                test_rows += f"""<tr>
                    <td><b>{tr['feature']}</b></td><td>{tr.get('test', 'N/A')}</td><td>{p_str}</td><td>{e_str}</td><td>{tr['level']}</td>
                </tr>"""

            div_id = f"chart_{chart_idx}"
            chart_idx += 1

            html += f"""
<div class="pair-card">
  <div class="pair-header">
    <span class="pair-title">{pair}</span>
    <div class="pair-meta">
      <span class="meta-badge">{n_total:,} bills</span>
      <span class="meta-badge">{n_clusters} cụm</span>
      <span class="meta-badge warn">{n_noise:,} nhiễu ({n_noise/max(n_total,1)*100:.1f}%)</span>
    </div>
  </div>
  <div class="pair-body">
    <div class="stats-grid">
      <div>
        <table class="cluster-table">
          <thead><tr><th>Cụm</th><th>Số bill</th><th>Trung bình</th><th>Trung vị</th><th>Độ lệch</th></tr></thead>
          <tbody>{cluster_rows}</tbody>
        </table>
      </div>
      <div>
        <table class="test-table">
          <thead><tr><th>Yếu tố</th><th>Phương pháp</th><th>p-value</th><th>Effect Size</th><th>Mức độ</th></tr></thead>
          <tbody>{test_rows}</tbody>
        </table>
      </div>
    </div>
    <div class="chart-container" id="{div_id}"></div>
    <p class="legend-hint">Hover vào cột để xem chi tiết tỷ lệ % · Biểu đồ Weight đã cap ở P95 để dễ nhìn</p>
  </div>
</div>
"""
            fig_json = fig.to_json()
            html += f"""<script>
var d_{chart_idx} = {fig_json};
Plotly.newPlot('{div_id}', d_{chart_idx}.data, d_{chart_idx}.layout, {{responsive: true}});
</script>\n"""

    html += "</body></html>"
    return html

# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Đọc dữ liệu ──
    print("Đọc dữ liệu...")
    bill_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bill.csv")
    bill_df = pd.read_csv(bill_path, usecols=["bill_code", "VD_type", "service",
                                                "receiving_date", "actual_weight",
                                                "origin_province", "destination_province",
                                                "bill_creation_date"])
    print(f"  bill.csv: {len(bill_df):,} dòng")

    # Thêm ngày trong tuần
    bill_df["receiving_date"] = pd.to_datetime(bill_df["receiving_date"], errors="coerce")
    day_map = {0: "T2", 1: "T3", 2: "T4", 3: "T5", 4: "T6", 5: "T7", 6: "CN"}
    bill_df["day_of_week"] = bill_df["receiving_date"].dt.dayofweek.map(day_map)

    bill_df["bill_creation_date"] = pd.to_datetime(bill_df["bill_creation_date"], errors="coerce")
    bill_df["creation_hour"] = bill_df["bill_creation_date"].dt.hour.fillna(-1).astype(int).astype(str) + "h"
    bill_df.loc[bill_df["creation_hour"] == "-1h", "creation_hour"] = "N/A"

    df_o = pd.read_csv(os.path.join(INPUT_DIR, "origin_head.csv"))
    df_d = pd.read_csv(os.path.join(INPUT_DIR, "destination_tail.csv"))
    print(f"  origin_head: {len(df_o):,} bills")
    print(f"  dest_tail:   {len(df_d):,} bills")

    # Merge thông tin bill
    merge_cols = ["bill_code", "VD_type", "service", "day_of_week", "actual_weight",
                  "origin_province", "destination_province", "creation_hour"]
    bill_merge = bill_df[merge_cols].copy()

    df_o = df_o.drop(columns=["actual_weight"], errors="ignore")
    df_o = df_o.merge(bill_merge, on="bill_code", how="left")

    df_d = df_d.drop(columns=["actual_weight"], errors="ignore")
    df_d = df_d.merge(bill_merge, on="bill_code", how="left")

    print(f"  Merge xong: origin VD_type matched {df_o['VD_type'].notna().sum():,}/{len(df_o):,}")
    print(f"              dest   VD_type matched {df_d['VD_type'].notna().sum():,}/{len(df_d):,}")

    # ── Cấu hình features ──
    cat_features = ["VD_type", "service", "day_of_week", "origin_province", "destination_province", "creation_hour", "departure_hour"]
    cont_features = ["actual_weight"]

    all_test_results = []
    results_by_flow = []

    # ── Luồng 1: Origin → 1A ──
    print("\n" + "█" * 70)
    print("  LUỒNG 1: KHO GỬI → KHO 1A NGUỒN")
    print("█" * 70)

    df_o_proc, pairs_o = get_top_pairs(df_o, "kho_o", "kho_o1a", "bill_count",
                                        "actual_weight", False)
    print(f"  Top {N_TOP} cặp kho: {len(pairs_o)} cặp")

    results_o, flow_pairs_o = analyze_flow(
        df_o_proc, pairs_o, "time_o", "time_o1a",
        "Kho gửi → Kho 1A", cat_features, cont_features
    )
    all_test_results.extend(results_o)
    results_by_flow.append(("Luồng 1: Kho gửi → Kho 1A nguồn", flow_pairs_o))

    # ── Luồng 2: 1A → Dest ──
    print("\n" + "█" * 70)
    print("  LUỒNG 2: KHO 1A ĐÍCH → KHO NHẬN")
    print("█" * 70)

    df_d_proc, pairs_d = get_top_pairs(df_d, "kho_d1a", "kho_d", "bill_count",
                                        "actual_weight", True)
    print(f"  Top {N_TOP} cặp kho: {len(pairs_d)} cặp")

    results_d, flow_pairs_d = analyze_flow(
        df_d_proc, pairs_d, "time_d1a", "time_d",
        "Kho 1A → Kho nhận", cat_features, cont_features
    )
    all_test_results.extend(results_d)
    results_by_flow.append(("Luồng 2: Kho 1A đích → Kho nhận", flow_pairs_d))

    # ── Xuất CSV tổng hợp ──
    if all_test_results:
        df_results = pd.DataFrame(all_test_results)
        csv_path = os.path.join(OUTPUT_DIR, "cluster_drivers_summary.csv")
        df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n Bảng tổng hợp: {csv_path}")

        # Tổng hợp trung bình effect size theo feature
        print(f"\n{'═'*70}")
        print(f"  TỔNG HỢP: XẾP HẠNG YẾU TỐ ẢNH HƯỞNG (Trung bình Effect Size)")
        print(f"{'═'*70}")
        summary = df_results.groupby("feature")["effect_size"].agg(["mean", "median", "max", "count"])
        summary = summary.sort_values("mean", ascending=False)
        print(f"  {'Yếu tố':<25} {'Mean':<10} {'Median':<10} {'Max':<10} {'Số cặp':<8}")
        print(f"  {'─'*60}")
        for feat, row in summary.iterrows():
            print(f"  {feat:<25} {row['mean']:<10.4f} {row['median']:<10.4f} {row['max']:<10.4f} {int(row['count']):<8}")

    # ── Xuất HTML ──
    print(f"\nVẽ biểu đồ...")
    html_content = build_full_html(results_by_flow, all_test_results)
    html_path = os.path.join(OUTPUT_DIR, "cluster_drivers.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" Biểu đồ: {html_path}")

    print(f"\n Hoàn tất phân tích!")
