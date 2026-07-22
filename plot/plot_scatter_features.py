import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_all_traces")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_plot")
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_TOP = 10

EXCLUDED_KHO = [
    "Kho TGDĐ Bảo Hành", "Kho Hà Tĩnh", "Kho Kiến An", "Kho Thường Tín",
    "Kho Amway", "Kho Digiworld", "Kho Thái Nguyên", "Kho Việt Trì",
    "Kho Ninh Hòa", "Kho An Giang",
    "Bưu cục Dự Án Vgreen Sóng Thần", "Bưu cục Dự Án Vgreen Văn Giang"
]

CATEGORY_COLORS = [
    "#2563EB", "#E11D48", "#10B981", "#F59E0B", "#8B5CF6",
    "#06B6D4", "#F97316", "#EC4899", "#14B8A6", "#84CC16",
    "#6366F1", "#EF4444", "#A855F7", "#0EA5E9", "#D946EF",
    "#22C55E", "#FBBF24", "#64748B", "#78716C", "#FB923C"
]

def find_optimal_shift(time_series):
    if time_series.empty: return 0
    hours = pd.to_datetime(time_series).dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0)
    extended = pd.concat([counts.iloc[-1:], counts, counts.iloc[:1]])
    rolling = extended.rolling(3, center=True).mean().iloc[1:-1]
    return int(rolling.idxmin())

def get_top_pairs(df, col_a, col_b, sort_by, wt_col, is_dest_flow, buu_cuc_set):
    df = df.copy()
    if is_dest_flow:
        df = df[~df[col_b].isin(EXCLUDED_KHO)]
    else:
        df = df[~df[col_a].isin(EXCLUDED_KHO)]
    df["pair"] = df[col_a].astype(str) + " → " + df[col_b].astype(str)
    
    check_col = col_b if is_dest_flow else col_a
    key = "so_bill" if sort_by == "bill_count" else "tong_kg"
    
    if buu_cuc_set is not None:
        is_bc = df[check_col].fillna("").astype(str).str.strip().isin(buu_cuc_set)
        df.loc[is_bc, "pair"] = df.loc[is_bc, "pair"] + " [Bưu Cục]"
        df.loc[~is_bc, "pair"] = df.loc[~is_bc, "pair"] + " [Thường]"
        df = df[is_bc].copy()
        
    agg = (df.groupby(["pair", check_col])
             .agg(so_bill=("bill_code", "nunique"), tong_kg=(wt_col, "sum"))
             .reset_index())
             
    top = agg.nlargest(N_TOP, key)
    return df, top["pair"].tolist()

def generate_scatter_figs(df, pairs, time_cols, time_labels, title_prefix):
    t_col_1, t_col_2 = time_cols[0], time_cols[1]
    
    # Chúng ta sẽ vẽ mỗi pair thành 1 block có subplots
    # Để tránh html quá to và lag, ta sẽ gom HTML string
    
    pair_blocks = []
    
    for idx, pair in enumerate(pairs):
        sub = df.loc[df["pair"] == pair].copy()
        sub = sub.dropna(subset=[t_col_1, t_col_2, "time"]).copy()
        
        # Lọc time ngoại lai
        sub = sub[(sub["time"] <= 100) & (sub["time"] >= 0)].copy()
        if not sub.empty:
            q1_time = sub["time"].quantile(0.01)
            q3_time = sub["time"].quantile(0.99)
            sub = sub[(sub["time"] >= q1_time) & (sub["time"] <= q3_time)].copy()
            
        if sub.empty: continue
        
        hour_shift = find_optimal_shift(sub[t_col_1])
        dt1 = pd.to_datetime(sub[t_col_1])
        dt2 = pd.to_datetime(sub[t_col_2])
        
        departure_date = dt1.dt.normalize()
        y2_vals_raw = (dt2 - departure_date).dt.total_seconds() / 3600.0
        raw_hours2 = dt1.dt.hour + dt1.dt.minute / 60.0 + dt1.dt.second / 3600.0
        x2_vals_raw = (raw_hours2 - hour_shift) % 24
        
        sub["x2"] = x2_vals_raw
        sub["y2"] = y2_vals_raw
        
        sub = sub[(sub["y2"] >= 0) & (sub["y2"] <= 48)].copy()
        if sub.empty: continue
        
        sub["_h_bin"] = pd.to_datetime(sub[t_col_1]).dt.hour
        total_bills = len(sub)
        hour_counts = sub["_h_bin"].value_counts()
        valid_hours = hour_counts[hour_counts >= 0.01 * total_bills].index
        sub = sub[sub["_h_bin"].isin(valid_hours)].copy()
        
        if sub.empty: continue
        
        # Sampling nếu quá nhiều điểm để biểu đồ nhanh hơn
        if len(sub) > 10000:
            sub = sub.sample(10000, random_state=42)
            
        x2_plot = sub["x2"]
        y2_plot = sub["y2"]
        real_hours_plot2 = (x2_plot + hour_shift) % 24
        sub["_real_h"] = real_hours_plot2
        
        # Tính correlation
        corr_pearson, p_p = stats.pearsonr(real_hours_plot2, y2_plot)
        corr_spearman, p_s = stats.spearmanr(real_hours_plot2, y2_plot)
        
        # Chạy DBSCAN
        y_for_cluster = y2_plot.values.reshape(-1, 1)
        dbscan = DBSCAN(eps=0.5, min_samples=100)
        clusters = dbscan.fit_predict(y_for_cluster)
        sub["_cluster"] = clusters
        
        titles = [
            f"1) Phân cụm (DBSCAN) | Corr: {corr_spearman:.2f}",
            "2) Tô màu theo VD_type",
            "3) Tô màu theo Service",
            "4) Tô màu theo Day of Week",
            "5) Tô màu theo Creation Hour",
            "6) Tô màu theo Actual Weight"
        ]
        
        fig = make_subplots(rows=6, cols=1, subplot_titles=titles, vertical_spacing=0.05)
        
        # Common axes settings function
        def update_axes(f, row, col):
            tv_x2 = list(range(0, 25, 2))
            tt_x2 = [f"{int((v + hour_shift) % 24)}h" for v in tv_x2]
            tv_y2 = list(range(0, 49, 3))
            tt_y2 = [f"{v % 24}h" + ("" if v < 24 else f" (+{v//24}d)") for v in tv_y2]
            max_y2 = min(48, y2_plot.max() + 2) if not y2_plot.empty else 48
            
            f.update_xaxes(title_text=f"Giờ xuất phát ({time_labels[0]})", range=[0, 24], tickvals=tv_x2, ticktext=tt_x2, showgrid=True, gridcolor="#E5E7EB", row=row, col=col)
            f.update_yaxes(title_text=f"Giờ đến ({time_labels[1]})", range=[0, max_y2], tickvals=tv_y2, ticktext=tt_y2, showgrid=True, gridcolor="#E5E7EB", row=row, col=col)

        # Plot 1: DBSCAN
        cluster_colors = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6"]
        unique_clusters = sorted(sub["_cluster"].unique())
        for clus in unique_clusters:
            mask = sub["_cluster"] == clus
            if clus == -1:
                c_color = "rgba(156, 163, 175, 0.4)"
                c_name = "Nhiễu (Noise)"
                opacity_val = 0.2
            else:
                c_color = cluster_colors[clus % len(cluster_colors)]
                c_name = f"Cụm {clus+1}"
                opacity_val = 0.6
                
            fig.add_trace(go.Scatter(
                x=sub.loc[mask, "x2"].tolist(), y=sub.loc[mask, "y2"].tolist(),
                customdata=sub.loc[mask, "_real_h"].tolist(),
                mode='markers', marker=dict(size=4, color=c_color, opacity=opacity_val),
                name=c_name, showlegend=True, legend="legend", legendgroup="Cụm",
                hovertemplate="Xuất phát: %{customdata:.2f}h<br>Đến: %{y:.1f}h<extra></extra>"
            ), row=1, col=1)
        update_axes(fig, 1, 1)
        
        # Helper for categorical plots
        def plot_categorical(feat, r, c, legend_name):
            if feat not in sub.columns: return
            sub[feat] = sub[feat].fillna("N/A")
            cats = sub[feat].value_counts().index.tolist()
            for i, cat in enumerate(cats):
                mask = sub[feat] == cat
                color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
                fig.add_trace(go.Scatter(
                    x=sub.loc[mask, "x2"].tolist(), y=sub.loc[mask, "y2"].tolist(),
                    customdata=sub.loc[mask, "_real_h"].tolist(),
                    mode='markers', marker=dict(size=4, color=color, opacity=0.6),
                    name=str(cat), showlegend=True, legend=legend_name, legendgroup=cat,
                    hovertemplate=f"Xuất phát: %{{customdata:.2f}}h<br>Đến: %{{y:.1f}}h<br>{feat}: {cat}<extra></extra>"
                ), row=r, col=c)
            update_axes(fig, r, c)
            
        # Plot 2: VD_type
        plot_categorical("VD_type", 2, 1, "legend2")
        
        # Plot 3: service
        plot_categorical("service", 3, 1, "legend3")
        
        # Plot 4: day_of_week
        plot_categorical("day_of_week", 4, 1, "legend4")
        
        # Plot 5: creation_hour
        plot_categorical("creation_hour", 5, 1, "legend5")
        
        # Plot 6: actual_weight
        if "actual_weight" in sub.columns:
            w_vals = sub["actual_weight"].fillna(0)
            # Cap weight at 95th percentile to avoid extreme outliers skewing the color/size
            cap_w = w_vals.quantile(0.95)
            if cap_w == 0: cap_w = 1
            w_vals_clipped = w_vals.clip(upper=cap_w)
            
            # Normalize for size (e.g. size from 3 to 10)
            sizes = 3 + (w_vals_clipped / cap_w) * 7
            
            fig.add_trace(go.Scatter(
                x=sub["x2"].tolist(), y=sub["y2"].tolist(),
                customdata=np.stack((sub["_real_h"], sub["actual_weight"]), axis=-1),
                mode='markers', 
                marker=dict(
                    size=sizes.tolist(),
                    color=w_vals_clipped.tolist(),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Weight(kg)", len=0.15, y=0.083, yanchor="middle", x=1.02),
                    opacity=0.7
                ),
                name="Weight", showlegend=False,
                hovertemplate="Xuất phát: %{customdata[0]:.2f}h<br>Đến: %{y:.1f}h<br>Weight: %{customdata[1]:.1f} kg<extra></extra>"
            ), row=6, col=1)
            update_axes(fig, 6, 1)
            
        fig.update_layout(
            height=3600, width=1200,
            title_text=f"<b>{pair}</b><br>Tổng bills: {len(sub):,} | {title_prefix}",
            plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
            font=dict(family="Segoe UI, Arial", size=11),
            margin=dict(t=100, b=30, l=50, r=200),
            legend=dict(title=dict(text="Cụm DBSCAN"), y=0.916, yanchor="middle", x=1.02),
            legend2=dict(title=dict(text="VD_type"), y=0.750, yanchor="middle", x=1.02),
            legend3=dict(title=dict(text="Service"), y=0.583, yanchor="middle", x=1.02),
            legend4=dict(title=dict(text="Day of Week"), y=0.416, yanchor="middle", x=1.02),
            legend5=dict(title=dict(text="Creation Hour"), y=0.250, yanchor="middle", x=1.02)
        )
        
        pair_blocks.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
        
    return pair_blocks


if __name__ == "__main__":
    print("Đọc dữ liệu kho...")
    wh_path = os.path.join(INPUT_DIR, "..", "warehouse.csv")
    wh_df = pd.read_csv(wh_path)
    buu_cuc_set = set(wh_df[wh_df['Bưu Cục'] == 'Y']['name'].dropna().str.strip())
    
    bill_path = os.path.join(INPUT_DIR, "..", "bill.csv")
    print(f"Đọc bill.csv ...")
    bill_df = pd.read_csv(bill_path, usecols=["bill_code", "VD_type", "service", "actual_weight", "receiving_date", "bill_creation_date"])
    
    # Tính day_of_week
    bill_df["receiving_date"] = pd.to_datetime(bill_df["receiving_date"], errors="coerce")
    day_map = {0: "T2", 1: "T3", 2: "T4", 3: "T5", 4: "T6", 5: "T7", 6: "CN"}
    bill_df["day_of_week"] = bill_df["receiving_date"].dt.dayofweek.map(day_map)
    
    # Tính creation_hour
    bill_df["bill_creation_date"] = pd.to_datetime(bill_df["bill_creation_date"], errors="coerce")
    bill_df["creation_hour"] = bill_df["bill_creation_date"].dt.hour.fillna(-1).astype(int).astype(str) + "h"
    bill_df.loc[bill_df["creation_hour"] == "-1h", "creation_hour"] = "N/A"
    
    print("Đọc dữ liệu traces...")
    df_o = pd.read_csv(os.path.join(INPUT_DIR, "origin_head.csv"))
    df_d = pd.read_csv(os.path.join(INPUT_DIR, "destination_tail.csv"))
    
    df_o = df_o.drop(columns=["actual_weight"], errors="ignore")
    df_d = df_d.drop(columns=["actual_weight"], errors="ignore")
    
    df_o = df_o.merge(bill_df, on="bill_code", how="left")
    df_d = df_d.merge(bill_df, on="bill_code", how="left")
    
    print("\nXử lý Luồng Origin -> 1A...")
    df_o_proc, pairs_o = get_top_pairs(df_o, "kho_o", "kho_o1a", "bill_count", "actual_weight", False, buu_cuc_set)
    html_blocks_o = generate_scatter_figs(df_o_proc, pairs_o, ["time_o", "time_o1a"], ["Giờ đi Kho đầu", "Giờ đến Kho 1A"], "Luồng gửi")
    
    print("\nXử lý Luồng 1A -> Dest...")
    df_d_proc, pairs_d = get_top_pairs(df_d, "kho_d1a", "kho_d", "bill_count", "actual_weight", True, buu_cuc_set)
    html_blocks_d = generate_scatter_figs(df_d_proc, pairs_d, ["time_d1a", "time_d"], ["Giờ đi Kho 1A", "Giờ đến Kho nhận"], "Luồng nhận")
    
    # Ghi file Luồng Gửi
    html_out_o = os.path.join(OUTPUT_DIR, "scatter_features_origin.html")
    print(f"Ghi ra file {html_out_o} ...")
    html_content_o = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Phân tích Scatter Plot Đặc Trưng - Luồng Gửi</title>
    <script src="https://cdn.plot.ly/plotly-2.34.0.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f1f5f9; padding: 20px; }}
        h1 {{ text-align: center; color: #1e293b; }}
        .pair-container {{ margin-bottom: 40px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); overflow-x: auto; }}
        hr {{ border: 1px solid #e2e8f0; margin: 40px 0; }}
        .flow-title {{ color: #2563EB; border-bottom: 3px solid #2563EB; padding-bottom: 10px; margin-top: 50px; }}
    </style>
</head>
<body>
    <h1>Phân tích Scatter Plot theo Đặc Trưng - Luồng Gửi</h1>
    <h2 class="flow-title">Kho gửi → Kho 1A nguồn</h2>
"""
    for block in html_blocks_o:
        html_content_o += f'<div class="pair-container">{block}</div>'
    html_content_o += "</body></html>"
    with open(html_out_o, "w", encoding="utf-8") as f:
        f.write(html_content_o)

    # Ghi file Luồng Nhận
    html_out_d = os.path.join(OUTPUT_DIR, "scatter_features_dest.html")
    print(f"Ghi ra file {html_out_d} ...")
    html_content_d = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Phân tích Scatter Plot Đặc Trưng - Luồng Nhận</title>
    <script src="https://cdn.plot.ly/plotly-2.34.0.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f1f5f9; padding: 20px; }}
        h1 {{ text-align: center; color: #1e293b; }}
        .pair-container {{ margin-bottom: 40px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); overflow-x: auto; }}
        hr {{ border: 1px solid #e2e8f0; margin: 40px 0; }}
        .flow-title {{ color: #2563EB; border-bottom: 3px solid #2563EB; padding-bottom: 10px; margin-top: 50px; }}
    </style>
</head>
<body>
    <h1>Phân tích Scatter Plot theo Đặc Trưng - Luồng Nhận</h1>
    <h2 class="flow-title">Kho 1A đích → Kho nhận</h2>
"""
    for block in html_blocks_d:
        html_content_d += f'<div class="pair-container">{block}</div>'
    html_content_d += "</body></html>"
    with open(html_out_d, "w", encoding="utf-8") as f:
        f.write(html_content_d)
        
    print("Xong!")
