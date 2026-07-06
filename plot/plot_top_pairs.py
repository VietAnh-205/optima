import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_all_traces")
OUTPUT_DIR = 'output_plot'
BINS   = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36, 48, 72, 120, float("inf")]
LABELS = ["0-1h","1-2h","2-3h","3-4h","4-5h","5-6h","6-8h","8-10h","10-12h",
          "12-18h","18-24h","24-36h","36-48h","48-72h","72-120h",">120h"]

BAR_COLOR, VIOLIN_COLOR = "#2563EB", "#1D4ED8"
N_TOP     = 10
FIG_WIDTH = 1400 
BAR_ROW_H = 1000

# Violin: mỗi cặp kho chiếm 1 khối 24 hàng x 1 cột = 24 ô (mỗi ô 1 giờ)
V_COLS, V_ROWS_PER_PAIR = 1, 24
V_ROW_H = 1000  # Chiều cao cho mỗi subplot (1 giờ) của violin


# ── Helpers dùng chung ────────────────────────────────────────────────
def make_bins(series):
    s = pd.cut(series.dropna(), bins=BINS, labels=LABELS, right=False)
    cnt = pd.Series(s).value_counts().reindex(LABELS, fill_value=0)
    pct = (cnt / max(cnt.sum(), 1) * 100).round(2)
    return cnt, pct


def median_label(med_val):
    for lo, hi, lab in zip(BINS[:-1], BINS[1:], LABELS):
        if lo <= med_val < hi:
            return lab
    return LABELS[-1]


def validate_columns(df, required, name):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Thiếu cột: {sorted(missing)}. Hiện có: {list(df.columns)}")


def top_pairs(df, col_a, col_b, sort_by, wt_col):
    df = df.copy()
    df["pair"] = df[col_a].astype(str) + " → " + df[col_b].astype(str)
    agg = (df.groupby("pair")
             .agg(so_bill=("bill_code", "nunique"), tong_kg=(wt_col, "sum"))
             .reset_index())
    key = "so_bill" if sort_by == "bill_count" else "tong_kg"
    return df, agg.nlargest(N_TOP, key)


def vspacing(rows, desired):
    return 0.0 if rows <= 1 else min(desired, 0.9 / (rows - 1))


def axref(ax, axis="x", suffix=""):
    return (axis if ax == 1 else f"{axis}{ax}") + suffix


# ── Figure 1: Bar chart (mỗi cặp kho = 1 hàng) ──────────────────────────
def build_bar_fig(df, pairs, top, sort_label, title_prefix):
    rows = len(pairs)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=pairs,
                         vertical_spacing=vspacing(rows, 0.01))

    for idx, pair in enumerate(pairs):
        r = ax = idx + 1
        subset = df.loc[df["pair"] == pair, "time"].dropna()
        cnt, pct = make_bins(subset)

        n_bill   = int(top.loc[top["pair"] == pair, "so_bill"].iloc[0])
        total_kg = top.loc[top["pair"] == pair, "tong_kg"].iloc[0]
        med  = float(subset.median()) if len(subset) else 0
        mean = float(subset.mean())   if len(subset) else 0

        fig.add_trace(go.Bar(
            x=LABELS, y=cnt.values.tolist(), marker_color=BAR_COLOR,
            text=[f"{int(v):,}<br>({p:.0f}%)" for v, p in zip(cnt.values, pct.values)],
            textposition="outside", textfont=dict(size=7), cliponaxis=False,
            hovertemplate=f"<b>%{{x}}</b><br>%{{y:,}} bill<br>Cặp: {pair}<extra></extra>",
        ), row=r, col=1)

        if len(subset):
            lab = median_label(med)
            fig.add_shape(type="line", xref=axref(ax), yref=axref(ax, "y", " domain"),
                          x0=lab, x1=lab, y0=0, y1=1,
                          line=dict(dash="dash", color="#E65100", width=1.5))
            fig.add_annotation(xref=axref(ax), yref=axref(ax, "y", " domain"),
                                x=lab, y=0.97, xanchor="left", yanchor="top",
                                text=f"Med={med:.0f}h", showarrow=False,
                                font=dict(size=8, color="#E65100"))

        fig.add_annotation(
            text=f"N={n_bill:,} | {total_kg/1000:.1f}T<br>Med={med:.0f}h  Mean={mean:.0f}h",
            xref=axref(ax, "x", " domain"), yref=axref(ax, "y", " domain"),
            xanchor="left", yanchor="top", x=0.01, y=0.99, showarrow=False,
            font=dict(size=9, color="#374151"), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#D1D5DB", borderpad=3)

        fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=r, col=1)
        y_ceiling = (cnt.max() or 1) * 1.45
        fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", rangemode="tozero",
                          range=[0, y_ceiling], title_text="Số bill", row=r, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} cặp kho theo {sort_label} — Phân phối thời gian vận chuyển</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=BAR_ROW_H * rows, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=90, b=30, l=70, r=40))
    return fig


def build_violin_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label):
    rows = len(pairs)
    cols = 2
    
    gap_px = 60
    # Chiều cao cho mỗi biểu đồ: 1500px x 2 = 3000px
    V_PAIR_H = 3000
    total_plot_height = rows * V_PAIR_H + (rows - 1) * gap_px

    grid_rows = rows * 2
    specs = []
    subplot_titles = []
    
    for pair in pairs:
        specs.append([{"type": "xy"}, {"type": "xy"}])
        specs.append([{"type": "xy"}, {"type": "xy"}])
        
        subplot_titles.append(f"{pair}<br>({time_labels[0]} vs T.gian VC)")
        subplot_titles.append(f"{pair}<br>({time_labels[1]} vs T.gian VC)")
        subplot_titles.append(f"{pair}<br>Tương quan: {time_labels[0]} theo {time_labels[1]}")
        subplot_titles.append(f"{pair}<br>Tương quan: {time_labels[1]} theo {time_labels[0]}")

    fig = make_subplots(rows=grid_rows, cols=cols, subplot_titles=subplot_titles,
                         vertical_spacing=0.015, horizontal_spacing=0.1, specs=specs)
    
    VIOLIN_COLORS = ["#10B981", "#1D4ED8"]
    VIOLIN_FILLS  = ["rgba(16,185,129,0.15)", "rgba(29,78,216,0.15)"]

    for p, pair in enumerate(pairs):
        grid_r_start = p * 2 + 1
        
        # ==========================================
        # Row 1: The two side-by-side Violins
        # ==========================================
        for c, t_col in enumerate(time_cols):
            col_idx = c + 1
            
            sub_raw = df.loc[df["pair"] == pair, [t_col, "time"]].dropna()
            if sub_raw.empty:
                continue

            sub = sub_raw[(sub_raw["time"] <= 100) & (sub_raw['time'] >= 0)].copy()
            sub_raw["hour"] = pd.to_datetime(sub_raw[t_col]).dt.hour
            sub["hour"] = pd.to_datetime(sub[t_col]).dt.hour
            
            total_bills = len(sub_raw)
            y_cats = []
            stats = {}
            for h in range(24):
                count_h = len(sub_raw[sub_raw["hour"] == h])
                pct_h = (count_h / total_bills * 100) if total_bills > 0 else 0
                label = f"{h}-{h+1}h<br>{count_h:,} ({pct_h:.1f}%)"
                y_cats.append(label)
                stats[h] = (label, count_h, pct_h)

            for h in range(24):
                x_vals = sub.loc[sub["hour"] == h, "time"].values
                label, count_h, pct_h = stats[h]
                
                if len(x_vals) < 2:
                    # Add dummy trace to preserve the Y category label
                    fig.add_trace(go.Violin(x=[None], y=[label], name=label, showlegend=False, hoverinfo='skip'), row=grid_r_start, col=col_idx)
                    continue
                
                fig.add_trace(go.Violin(
                    x=x_vals.tolist(),
                    y=[label] * len(x_vals),
                    name=label, line_color=VIOLIN_COLORS[c],
                    fillcolor=VIOLIN_FILLS[c], box_visible=True, meanline_visible=True,
                    points='outliers', showlegend=False, orientation='h', spanmode='hard',
                    hovertemplate=(
                        f"<b>{h}h–{h+1}h</b><br>"
                        f"Mốc: {time_labels[c]}<br>"
                        f"Số bill: {count_h:,} ({pct_h:.1f}%)<br>"
                        f"Thời gian vc: %{{x:.0f}}h<br>"
                        f"Cặp: {pair}<extra></extra>"
                    ),
                ), row=grid_r_start, col=col_idx)

            fig.update_xaxes(title_text="Thời gian vc (h)", showgrid=True, gridcolor="#E5E7EB", row=grid_r_start, col=col_idx)
            fig.update_yaxes(categoryorder="array", categoryarray=y_cats[::-1], showgrid=True, gridcolor="#E5E7EB", row=grid_r_start, col=col_idx)

        # ==========================================
        # Row 2: Inter-hour Violin (Left & Right)
        # ==========================================
        t_col_1, t_col_2 = time_cols[0], time_cols[1]
        sub2_raw = df.loc[df["pair"] == pair, [t_col_1, t_col_2]].dropna()
        if not sub2_raw.empty:
            dt1 = pd.to_datetime(sub2_raw[t_col_1])
            dt2 = pd.to_datetime(sub2_raw[t_col_2])
            
            sub2_raw["hour_1_cont"] = dt1.dt.hour + dt1.dt.minute / 60.0 + dt1.dt.second / 3600.0
            sub2_raw["hour_2_cont"] = dt2.dt.hour + dt2.dt.minute / 60.0 + dt2.dt.second / 3600.0
            sub2_raw["hour_1"] = dt1.dt.hour
            sub2_raw["hour_2"] = dt2.dt.hour
            
            total_bills_2 = len(sub2_raw)
            
            # --- LEFT PLOT ---
            y_cats_left = []
            stats_left = {}
            for h in range(24):
                count_h = len(sub2_raw[sub2_raw["hour_2"] == h])
                pct_h = (count_h / total_bills_2 * 100) if total_bills_2 > 0 else 0
                label = f"{h}-{h+1}h<br>{count_h:,} ({pct_h:.1f}%)"
                y_cats_left.append(label)
                stats_left[h] = (label, count_h, pct_h)
                
            for h in range(24):
                x_vals = sub2_raw.loc[sub2_raw["hour_2"] == h, "hour_1_cont"].values
                label = stats_left[h][0]
                
                if len(x_vals) < 2:
                    fig.add_trace(go.Violin(x=[None], y=[label], name=label, showlegend=False, hoverinfo='skip'), row=grid_r_start + 1, col=1)
                    continue
                
                fig.add_trace(go.Violin(
                    x=x_vals.tolist(), y=[label] * len(x_vals),
                    name=label, line_color="#8B5CF6", fillcolor="rgba(139,92,246,0.15)",
                    box_visible=True, meanline_visible=True, points='outliers',
                    showlegend=False, orientation='h', spanmode='hard',
                    hovertemplate=(f"<b>{h}h–{h+1}h tại {time_labels[1]}</b><br>Giờ tại {time_labels[0]}: %{{x:.1f}}h<extra></extra>")
                ), row=grid_r_start + 1, col=1)
                
            fig.update_xaxes(title_text=f"Giờ tại {time_labels[0]} (0-24h)", range=[0, 24], dtick=2, showgrid=True, gridcolor="#E5E7EB", row=grid_r_start + 1, col=1)
            fig.update_yaxes(categoryorder="array", categoryarray=y_cats_left[::-1], title_text=f"Khung giờ {time_labels[1]}", showgrid=True, gridcolor="#E5E7EB", row=grid_r_start + 1, col=1)

            # --- RIGHT PLOT ---
            y_cats_right = []
            stats_right = {}
            for h in range(24):
                count_h = len(sub2_raw[sub2_raw["hour_1"] == h])
                pct_h = (count_h / total_bills_2 * 100) if total_bills_2 > 0 else 0
                label = f"{h}-{h+1}h<br>{count_h:,} ({pct_h:.1f}%)"
                y_cats_right.append(label)
                stats_right[h] = (label, count_h, pct_h)
                
            for h in range(24):
                x_vals = sub2_raw.loc[sub2_raw["hour_1"] == h, "hour_2_cont"].values
                label = stats_right[h][0]
                
                if len(x_vals) < 2:
                    fig.add_trace(go.Violin(x=[None], y=[label], name=label, showlegend=False, hoverinfo='skip'), row=grid_r_start + 1, col=2)
                    continue
                
                fig.add_trace(go.Violin(
                    x=x_vals.tolist(), y=[label] * len(x_vals),
                    name=label, line_color="#F59E0B", fillcolor="rgba(245,158,11,0.15)", # Amber
                    box_visible=True, meanline_visible=True, points='outliers',
                    showlegend=False, orientation='h', spanmode='hard',
                    hovertemplate=(f"<b>{h}h–{h+1}h tại {time_labels[0]}</b><br>Giờ tại {time_labels[1]}: %{{x:.1f}}h<extra></extra>")
                ), row=grid_r_start + 1, col=2)
                
            fig.update_xaxes(title_text=f"Giờ tại {time_labels[1]} (0-24h)", range=[0, 24], dtick=2, showgrid=True, gridcolor="#E5E7EB", row=grid_r_start + 1, col=2)
            fig.update_yaxes(categoryorder="array", categoryarray=y_cats_right[::-1], title_text=f"Khung giờ {time_labels[0]}", showgrid=True, gridcolor="#E5E7EB", row=grid_r_start + 1, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} theo {sort_label} — Violin: phân phối theo giờ tại 2 mốc vs thời gian vận chuyển</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=total_plot_height + 120, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=110, b=30, l=70, r=40), violingap=0)
    return fig


def build_scatter_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label):
    rows = len(pairs)
    cols = 1
    
    gap_px = 60
    S_PAIR_H = 800
    total_plot_height = rows * S_PAIR_H + (rows - 1) * gap_px

    subplot_titles = [f"{pair}<br>({time_labels[0]} vs {time_labels[1]})" for pair in pairs]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                         vertical_spacing=vspacing(rows, 0.05))

    for p, pair in enumerate(pairs):
        r = p + 1
        t_col_1 = time_cols[0]
        
        sub = df.loc[df["pair"] == pair, [t_col_1, "time"]].dropna()
        sub = sub[(sub["time"] <= 100) & (sub['time'] >= 0)].copy()

        if sub.empty:
            continue
            
        dt1 = pd.to_datetime(sub[t_col_1])
        
        # Convert to continuous hours (0-24)
        x_vals = dt1.dt.hour + dt1.dt.minute / 60.0 + dt1.dt.second / 3600.0
        y_vals = sub["time"]
        
        fig.add_trace(go.Scattergl(
            x=x_vals.tolist(), y=y_vals.tolist(),
            mode='markers',
            marker=dict(size=4, color="#1B4EF5", opacity=0.4), # amber color
            name="Bill",
            hovertemplate=f"Xuất phát ({time_labels[0]}): %{{x:.2f}}h<br>T.gian VC: %{{y:.1f}}h<br>Cặp: {pair}<extra></extra>"
        ), row=r, col=1)

        fig.update_xaxes(title_text=f"Giờ xuất phát ({time_labels[0]})", range=[0, 24], dtick=2, showgrid=True, gridcolor="#E5E7EB", row=r, col=1)
        fig.update_yaxes(title_text=f"Thời gian VC (h)", rangemode="tozero", showgrid=True, gridcolor="#E5E7EB", row=r, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} theo {sort_label} — Scatter: Giờ xuất phát vs Thời gian vận chuyển</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=total_plot_height + 120, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=110, b=30, l=70, r=40))
    return fig


def write_combined_html(fig_bar, fig_violin, fig_scatter, out_file, page_title):
    # os.makedirs(os.path.dirname(out_file), exist_ok=True)
    bar_div    = pio.to_html(fig_bar, full_html=False, include_plotlyjs=False)
    violin_div = pio.to_html(fig_violin, full_html=False, include_plotlyjs=False)
    scatter_div = pio.to_html(fig_scatter, full_html=False, include_plotlyjs=False)
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"><title>{page_title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#f4f6f9; margin:0; padding:20px; }}
  h2   {{ color:#1e3a5f; margin: 0 0 16px; font-weight: 600; }}
  
  .tabs {{
    display: flex;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 20px;
    background: #fff;
    border-radius: 8px 8px 0 0;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .tab-btn {{
    background: inherit;
    border: none;
    outline: none;
    cursor: pointer;
    padding: 16px 32px;
    font-size: 16px;
    font-weight: 600;
    color: #64748b;
    transition: all 0.2s ease;
    border-bottom: 3px solid transparent;
  }}
  .tab-btn:hover {{ background-color: #f8fafc; color: #334155; }}
  .tab-btn.active {{ color: #2563EB; border-bottom: 3px solid #2563EB; background: #fff; }}
  
  .tabcontent {{
    display: none;
    background: #fff;
    padding: 20px;
    border-radius: 0 0 8px 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    animation: fadeEffect 0.4s;
  }}
  @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
</style>
</head>
<body>
  <h2>{page_title}</h2>
  
  <div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'BarChart')">Phân phối (Bar Chart)</button>
    <button class="tab-btn" onclick="openTab(event, 'ViolinPlot')">Tương quan (Violin Plot)</button>
    <button class="tab-btn" onclick="openTab(event, 'ScatterPlot')">Phân tán (Scatter Plot)</button>
  </div>

  <div id="BarChart" class="tabcontent" style="display:block;">
    {bar_div}
  </div>

  <div id="ViolinPlot" class="tabcontent">
    {violin_div}
  </div>

  <div id="ScatterPlot" class="tabcontent">
    {scatter_div}
  </div>

  <script>
  function openTab(evt, tabName) {{
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
    
    tablinks = document.getElementsByClassName("tab-btn");
    for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
    
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
    
    // Resize plotly to fix hidden rendering issue
    window.dispatchEvent(new Event('resize'));
  }}
  </script>
</body>
</html>"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ==> {out_file}")


def build_all(df, col_a, col_b, sort_by, title_prefix, out_file, wt_col="actual_weight", time_cols=None, time_labels=None):
    required = {col_a, col_b, "bill_code", wt_col, "time"}
    if time_cols:
        required.update(time_cols)
    validate_columns(df, required, out_file)

    df, top = top_pairs(df, col_a, col_b, sort_by, wt_col)
    sort_label = "Số bill" if sort_by == "bill_count" else "Tổng kg"
    pairs = top["pair"].tolist()
    if not pairs:
        print(f"  !! Không có dữ liệu cho {out_file}, bỏ qua.")
        return

    fig_bar = build_bar_fig(df, pairs, top, sort_label, title_prefix)
    if time_cols:
        fig_violin = build_violin_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label)
        fig_scatter = build_scatter_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label)
        write_combined_html(fig_bar, fig_violin, fig_scatter, out_file, f"{title_prefix} – Top {N_TOP} {sort_label}")
    else:
        fig_bar.write_html(out_file, include_plotlyjs="cdn")
        print(f"  ==> {out_file}")


if __name__ == "__main__":
    print("Đọc dữ liệu...")
    df_o = pd.read_csv(os.path.join(INPUT_DIR, "origin_to_1A.csv"))
    df_d = pd.read_csv(os.path.join(INPUT_DIR, "1A_to_destination.csv"))
    print(f"  origin_to_1A : {len(df_o):,} bill")
    print(f"  1A_to_dest   : {len(df_d):,} bill")

    print("\nVẽ top 10 cặp kho NHIỀU BILL nhất...")
    build_all(df_o, "kho_o", "kho_o1a", "bill_count", "Kho gửi → Kho 1A nguồn",
              os.path.join(OUTPUT_DIR, "top10_bill_origin.html"), 
              time_cols=["time_o", "time_o1a"], time_labels=["Giờ đến Kho đầu", "Giờ đến Kho 1A"])
    build_all(df_d, "kho_d1a", "kho_d", "bill_count", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, "top10_bill_dest.html"), 
              time_cols=["time_d1a", "time_d"], time_labels=["Giờ đến Kho 1A", "Giờ đến Kho đích"])

    print("\nVẽ top 10 cặp kho NHIỀU KG nhất...")
    build_all(df_o, "kho_o", "kho_o1a", "total_kg", "Kho gửi → Kho 1A nguồn",
              os.path.join(OUTPUT_DIR, "top10_kg_origin.html"), 
              time_cols=["time_o", "time_o1a"], time_labels=["Giờ đến Kho đầu", "Giờ đến Kho 1A"])
    build_all(df_d, "kho_d1a", "kho_d", "total_kg", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, "top10_kg_dest.html"), 
              time_cols=["time_d1a", "time_d"], time_labels=["Giờ đến Kho 1A", "Giờ đến Kho đích"])

    print("\nXong! 4 file HTML (bar + violin) đã được lưu.")