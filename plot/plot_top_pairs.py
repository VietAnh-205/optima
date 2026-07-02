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
FIG_WIDTH = 1500
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


def build_violin_fig(df, pairs, time_col, title_prefix, sort_label):
    rows = len(pairs)
    
    gap_px = 40
    # Chiều cao cho 1 CẶP KHO (chứa 24 giờ bên trong)
    V_PAIR_H = 2500
    total_plot_height = rows * V_PAIR_H + (rows - 1) * gap_px
    v_space = gap_px / total_plot_height if rows > 1 else 0.0

    fig = make_subplots(rows=rows, cols=1, subplot_titles=pairs,
                         vertical_spacing=0.01)

    for p, pair in enumerate(pairs):
        r = ax = p + 1
        sub_raw = df.loc[df["pair"] == pair, [time_col, "time"]].dropna()
        if sub_raw.empty:
            fig.add_annotation(text="Không có dữ liệu", xref=axref(ax, "x", " domain"),
                                yref=axref(ax, "y", " domain"), x=0.5, y=0.5,
                                showarrow=False, font=dict(size=11, color="#9CA3AF"))
            continue

        # Giới hạn Y ≤ P99 để tránh outlier
        # cap = sub_raw["time"].quantile(0.99)
        # sub = sub_raw[sub_raw["time"] <= cap].copy()
        sub = sub_raw.copy()
        sub_raw["hour"] = pd.to_datetime(sub_raw[time_col]).dt.hour
        sub["hour"] = pd.to_datetime(sub[time_col]).dt.hour
        
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
            if len(x_vals) < 2:
                continue
            
            label, count_h, pct_h = stats[h]
            fig.add_trace(go.Violin(
                x=x_vals.tolist(),
                y=[label] * len(x_vals),
                name=label, line_color=VIOLIN_COLOR,
                fillcolor="rgba(29,78,216,0.15)", box_visible=True, meanline_visible=True,
                points='outliers', showlegend=False, orientation='h',
                hovertemplate=(
                    f"<b>{h}h–{h+1}h</b><br>"
                    f"Số bill: {count_h:,} ({pct_h:.1f}%)<br>"
                    f"Thời gian vc: %{{x:.0f}}h<br>"
                    f"Cặp: {pair}<extra></extra>"
                ),
            ), row=r, col=1)

        fig.update_xaxes(title_text="Thời gian vc (h)", showgrid=True, gridcolor="#E5E7EB", row=r, col=1)
        # Đảo ngược mảng để 0-1h ở trên cùng, 23-24h ở dưới cùng
        fig.update_yaxes(categoryorder="array", categoryarray=y_cats[::-1], showgrid=True, gridcolor="#E5E7EB", row=r, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} theo {sort_label} — Violin: từng giờ tại kho 1A vs thời gian vận chuyển</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=total_plot_height + 90 + 30, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=90, b=30, l=70, r=40), violingap=0)
    return fig


def write_combined_html(fig_bar, fig_violin, out_file, page_title):
    bar_div    = pio.to_html(fig_bar, full_html=False, include_plotlyjs=False)
    violin_div = pio.to_html(fig_violin, full_html=False, include_plotlyjs=False)
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"><title>{page_title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#fff; margin:0; padding:16px; }}
  h2   {{ color:#1e3a5f; border-bottom:2px solid #2563EB; padding-bottom:6px; margin:0 0 4px; }}
  hr   {{ border:none; border-top:1px solid #E5E7EB; margin:8px 0; }}
</style>
</head>
<body>
  <h2>Phân phối thời gian vận chuyển (Bar chart)</h2>
  {bar_div}
  <hr>
  <h2>Giờ tại kho 1A vs Thời gian vận chuyển (Violin plot)</h2>
  {violin_div}
</body>
</html>"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ==> {out_file}")


def build_all(df, col_a, col_b, sort_by, title_prefix, out_file, wt_col="actual_weight", time_col=None):
    required = {col_a, col_b, "bill_code", wt_col, "time"}
    if time_col:
        required.add(time_col)
    validate_columns(df, required, out_file)

    df, top = top_pairs(df, col_a, col_b, sort_by, wt_col)
    sort_label = "Số bill" if sort_by == "bill_count" else "Tổng kg"
    pairs = top["pair"].tolist()
    if not pairs:
        print(f"  !! Không có dữ liệu cho {out_file}, bỏ qua.")
        return

    fig_bar = build_bar_fig(df, pairs, top, sort_label, title_prefix)
    if time_col:
        fig_violin = build_violin_fig(df, pairs, time_col, title_prefix, sort_label)
        write_combined_html(fig_bar, fig_violin, out_file, f"{title_prefix} – Top {N_TOP} {sort_label}")
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
              os.path.join(OUTPUT_DIR, "top10_bill_origin.html"), time_col="time_o1a")
    build_all(df_d, "kho_d1a", "kho_d", "bill_count", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, "top10_bill_dest.html"), time_col="time_d1a")

    print("\nVẽ top 10 cặp kho NHIỀU KG nhất...")
    build_all(df_o, "kho_o", "kho_o1a", "total_kg", "Kho gửi → Kho 1A nguồn",
              os.path.join(OUTPUT_DIR, "top10_kg_origin.html"), time_col="time_o1a")
    build_all(df_d, "kho_d1a", "kho_d", "total_kg", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, "top10_kg_dest.html"), time_col="time_d1a")

    print("\nXong! 4 file HTML (bar + violin) đã được lưu.")