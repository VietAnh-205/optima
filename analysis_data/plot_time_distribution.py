

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INPUT_DIR  = "output_all_traces"
OUTPUT_HTML = os.path.join(INPUT_DIR, "time_distribution.html")

# ── 1. Đọc dữ liệu ────────────────────────────────────────────────────────────
df_o = pd.read_csv(os.path.join(INPUT_DIR, "origin_to_1A.csv"),       usecols=["time"])
df_d = pd.read_csv(os.path.join(INPUT_DIR, "1A_to_destination.csv"),  usecols=["time"])

df_o = df_o.dropna().rename(columns={"time": "gio"})
df_d = df_d.dropna().rename(columns={"time": "gio"})


BINS   = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36, 48, 72, 120, float("inf")]
LABELS = [
    "0-1h", "1-2h", "2-3h", "3-4h", "4-5h", "5-6h",
    "6-8h", "8-10h", "10-12h",
    "12-18h", "18-24h",
    "24-36h", "36-48h", "48-72h", "72-120h", ">120h",
]
COLOR_SINGLE = "#2563EB"   # 1 màu xanh dương duy nhất cho tất cả cột
COLORS_O = [COLOR_SINGLE] * len(LABELS)
COLORS_D = [COLOR_SINGLE] * len(LABELS)


def make_bins(df, bins, labels):
    df = df.copy()
    df["bucket"] = pd.cut(df["gio"], bins=bins, labels=labels, right=False)
    cnt = df["bucket"].value_counts().reindex(labels, fill_value=0)
    pct = (cnt / cnt.sum() * 100).round(2)
    return cnt, pct

cnt_o, pct_o = make_bins(df_o, BINS, LABELS)
cnt_d, pct_d = make_bins(df_d, BINS, LABELS)

# ── 3. Thống kê mô tả ─────────────────────────────────────────────────────────
def stats(df):
    s = df["gio"]
    return {
        "count": f"{len(s):,.0f}",
        "mean":  f"{s.mean():.1f}h",
        "median":f"{s.median():.1f}h",
        "p25":   f"{s.quantile(0.25):.1f}h",
        "p75":   f"{s.quantile(0.75):.1f}h",
    }

so = stats(df_o)
sd = stats(df_d)

# ── 4. Vẽ: 1 hàng 2 cột, text = "N bill (X%)" ────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Kho gửi → Kho 1A nguồn",
        "Kho 1A đích → Kho nhận",
    ),
    horizontal_spacing=0.08,
)

# Text label: "N (X%)"
text_o = [f"{int(c):,}<br>({p:.1f}%)" for c, p in zip(cnt_o.values, pct_o.values)]
text_d = [f"{int(c):,}<br>({p:.1f}%)" for c, p in zip(cnt_d.values, pct_d.values)]

fig.add_trace(go.Bar(
    x=LABELS, y=cnt_o.values,
    marker_color=COLORS_O,
    text=text_o,
    textposition="outside",
    textfont=dict(size=10),
    name="Nguồn→1A",
    hovertemplate="<b>%{x}</b><br>%{y:,} bill<br>%{text}<extra></extra>",
    cliponaxis=False,
), row=1, col=1)

fig.add_trace(go.Bar(
    x=LABELS, y=cnt_d.values,
    marker_color=COLORS_D,
    text=text_d,
    textposition="outside",
    textfont=dict(size=10),
    name="1A→Đích",
    hovertemplate="<b>%{x}</b><br>%{y:,} bill<br>%{text}<extra></extra>",
    cliponaxis=False,
), row=1, col=2)

# Đường median — trục X là categorical, dùng index số nguyên (0-based) của label
def get_median_label_idx(med_val, bins, labels):
    """Trả về index (0-based) của bin chứa giá trị median."""
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= med_val < hi:
            return i
    return len(labels) - 1

for col_idx, (df_part, lbl, color) in enumerate([
    (df_o, "Median nguồn→1A", "#E65100"),
    (df_d, "Median 1A→đích",  "#E65100"),
], start=1):
    med     = df_part["gio"].median()
    med_idx = get_median_label_idx(med, BINS, LABELS)
    fig.add_vline(
        x=med_idx, line_dash="dash", line_color=color, line_width=2,
        annotation_text=f"{lbl}: {med:.0f}h (bin: {LABELS[med_idx]})",
        annotation_font_size=11,
        annotation_font_color=color,
        row=1, col=col_idx,
    )

# ── 5. Layout ──────────────────────────────────────────────────────────────────
fig.update_layout(
    title={
        "text": (
            "<b>Phân phối thời gian vận chuyển</b><br>"
            "<sup>Kho gửi → Kho 1A nguồn &nbsp;|&nbsp; Kho 1A đích → Kho nhận</sup>"
        ),
        "x": 0.5, "xanchor": "center",
        "font": {"size": 20},
    },
    height=580,
    width=1400,
    showlegend=False,
    plot_bgcolor="#F8F9FA",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Segoe UI, Arial", size=12),
    margin=dict(t=110, b=200, l=60, r=40),
    uniformtext=dict(mode="hide", minsize=8),
)

for c in [1, 2]:
    fig.update_yaxes(
        title_text="Số bill",
        showgrid=True, gridcolor="#E0E0E0",
        zeroline=True, zerolinecolor="#BDBDBD",
        row=1, col=c,
    )
    fig.update_xaxes(tickangle=-30, row=1, col=c)

# ── 6. Annotation thống kê ────────────────────────────────────────────────────
def stat_text(s, label):
    return (
        f"<b>{label}</b><br>"
        f"N = {s['count']}<br>"
        f"Mean = {s['mean']}  |  Median = {s['median']}<br>"
        f"Q1 = {s['p25']}  |  Q3 = {s['p75']}"
    )

fig.add_annotation(
    text=stat_text(so, "Kho gửi → Kho 1A nguồn"),
    xref="paper", yref="paper", x=0.0, y=-0.38,
    showarrow=False, align="left",
    font=dict(size=11, color="#424242"),
    bgcolor="#EFF6FF", bordercolor="#93C5FD", borderpad=7,
)
fig.add_annotation(
    text=stat_text(sd, "Kho 1A đích → Kho nhận"),
    xref="paper", yref="paper", x=0.55, y=-0.38,
    showarrow=False, align="left",
    font=dict(size=11, color="#424242"),
    bgcolor="#F0FDF4", bordercolor="#86EFAC", borderpad=7,
)

# ── 7. Lưu ────────────────────────────────────────────────────────────────────
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
print(f": {OUTPUT_HTML}")


