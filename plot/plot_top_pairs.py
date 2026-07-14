import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import DBSCAN

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_all_traces")
OUTPUT_DIR = 'output_plot'
BINS   = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36, 48, 72, 120, float("inf")]
LABELS = ["0-1h","1-2h","2-3h","3-4h","4-5h","5-6h","6-8h","8-10h","10-12h",
          "12-18h","18-24h","24-36h","36-48h","48-72h","72-120h",">120h"]

BAR_COLOR, VIOLIN_COLOR = "#2563EB", "#1D4ED8"
N_TOP     = 10

# Bảng màu cho phân cụm scatter theo nhãn (tối đa ~20 loại)
CATEGORY_COLORS = [
    "#2563EB",  # xanh dương
    "#E11D48",  # đỏ hồng
    "#10B981",  # xanh lá
    "#F59E0B",  # vàng cam
    "#8B5CF6",  # tím
    "#06B6D4",  # cyan
    "#F97316",  # cam
    "#EC4899",  # hồng
    "#14B8A6",  # teal
    "#84CC16",  # lime
    "#6366F1",  # indigo
    "#EF4444",  # đỏ
    "#A855F7",  # purple
    "#0EA5E9",  # sky
    "#D946EF",  # fuchsia
    "#22C55E",  # green
    "#FBBF24",  # amber
    "#64748B",  # slate
    "#78716C",  # stone
    "#FB923C",  # orange light
]

FIG_WIDTH = 1400 
BAR_ROW_H = 1000

# Violin: mỗi cặp kho chỉ còn 1 hàng x 2 cột (hình 1 + hình 4), giảm resolution
V_COLS, V_ROWS_PER_PAIR = 2, 1
V_ROW_H = 600  # Chiều cao cho mỗi subplot (giảm so với trước)

EXCLUDED_KHO = [
    "Kho TGDĐ Bảo Hành",
    "Kho Hà Tĩnh",
    "Kho Kiến An",
    "Kho Thường Tín",
    "Kho Amway",
    "Kho Digiworld",
    "Kho Thái Nguyên",
    "Kho Việt Trì",
    "Kho Ninh Hòa",
    "Kho An Giang",
    "Bưu cục Dự Án Vgreen Sóng Thần",
    "Bưu cục Dự Án Vgreen Văn Giang"
]

# ── Helpers dùng chung ────────────────────────────────────────────────
def find_optimal_shift(time_series):
    """Tính điểm cắt (thung lũng) để shift trục giờ tránh đứt đoạn"""
    if time_series.empty: return 0
    hours = pd.to_datetime(time_series).dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0)
    # Dùng rolling 3 để tìm thung lũng ổn định, nối đuôi-đầu để xử lý tính tuần hoàn
    extended = pd.concat([counts.iloc[-1:], counts, counts.iloc[:1]])
    rolling = extended.rolling(3, center=True).mean().iloc[1:-1]
    return int(rolling.idxmin())


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


def top_pairs(df, col_a, col_b, sort_by, wt_col, is_dest_flow=False, buu_cuc_set=None):
    df = df.copy()
    
    # # Lọc bỏ các kho trong danh sách bị loại trừ
    # df = df[~df[col_a].isin(EXCLUDED_KHO) & ~df[col_b].isin(EXCLUDED_KHO)].copy()
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

        # Chỉ giữ lại các bill có kho đầu (origin) / kho cuối (dest) là Bưu Cục
        df = df[is_bc].copy()

        agg = (df.groupby(["pair", check_col])
                 .agg(so_bill=("bill_code", "nunique"), tong_kg=(wt_col, "sum"))
                 .reset_index())

        top_bc = agg.nlargest(N_TOP, key)
        return df, top_bc.reset_index(drop=True)
        
    agg = (df.groupby(["pair", check_col])
             .agg(so_bill=("bill_code", "nunique"), tong_kg=(wt_col, "sum"))
             .reset_index())
             
    return df, agg.nlargest(N_TOP, key)


def vspacing(rows, desired):
    return 0.0 if rows <= 1 else min(desired, 0.9 / (rows - 1))


def axref(ax, axis="x", suffix=""):
    return (axis if ax == 1 else f"{axis}{ax}") + suffix


# ── Figure 1: Bar chart (mỗi cặp kho = 1 hàng) ──────────────────────────
def build_bar_fig(df, pairs, top, sort_label, title_prefix):
    rows = len(pairs)
    gap_px = 60
    total_plot_height = BAR_ROW_H * rows + (rows - 1) * gap_px
    fig = make_subplots(rows=rows, cols=1, subplot_titles=pairs,
                         vertical_spacing=gap_px / max(total_plot_height, 1))

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
                                text=f"Med={med:.1f}h", showarrow=False,
                                font=dict(size=8, color="#E65100"))

        fig.add_annotation(
            text=f"N={n_bill:,} | {total_kg/1000:.1f}T<br>Med={med:.1f}h  Mean={mean:.1f}h",
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
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} kho bưu cục theo {sort_label} — Phân phối thời gian vận chuyển</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=BAR_ROW_H * rows, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=90, b=30, l=70, r=40))
    return fig


def build_violin_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label, pair_shifts):
    """
    Mỗi cặp kho chỉ còn 2 hình đặt cạnh nhau trên 1 hàng (thay vì lưới 2x2):
      - Hình 1 (trái): phân phối Thời gian vận chuyển theo từng khung giờ của time_labels[0]
      - Hình 4 (phải): tương quan — với mỗi khung giờ của time_labels[0], phân phối giờ đến
        (time_labels[1]) tương ứng
    """
    rows = len(pairs)
    cols = 2

    V_ROW_H = 600
    gap_px = 80
    total_plot_height = rows * V_ROW_H + (rows - 1) * gap_px
    specs = [[{"type": "xy"}, {"type": "xy"}] for _ in pairs]

    subplot_titles = []
    for pair in pairs:
        subplot_titles.append(f"{pair}<br>({time_labels[0]} vs T.gian VC)")
        subplot_titles.append(f"{pair}<br>Tương quan: {time_labels[1]} theo {time_labels[0]}")

    # Create subplots
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                         vertical_spacing=gap_px / max(total_plot_height, 1),
                         horizontal_spacing=0.06, specs=specs)

    VIOLIN_COLORS = ["#10B981", "#1D4ED8"]
    VIOLIN_FILLS  = ["rgba(16,185,129,0.15)", "rgba(29,78,216,0.15)"]

    for p, pair in enumerate(pairs):
        r = p + 1

        # ==========================================
        # Cột 1 (hình 1): Violin time_labels[0] vs T.gian VC
        # ==========================================
        t_col = time_cols[0]
        sub_raw = df.loc[df["pair"] == pair, [t_col, "time"]].dropna().copy()
        if not sub_raw.empty:
            sub = sub_raw[(sub_raw["time"] <= 100) & (sub_raw['time'] >= 0)].copy()
            sub_raw["hour"] = pd.to_datetime(sub_raw[t_col]).dt.hour
            sub["hour"] = pd.to_datetime(sub[t_col]).dt.hour

            total_bills = len(sub_raw)
            y_cats = []
            stats_left = {}
            # Shifted order: cycle liền mạch cho từng cặp
            hour_shift = pair_shifts.get(pair, 0)
            shifted_order = [(i + hour_shift) % 24 for i in range(24)]
            for h in shifted_order:
                h_next = (h + 1) % 24
                count_h = len(sub_raw[sub_raw["hour"] == h])
                pct_h = (count_h / total_bills * 100) if total_bills > 0 else 0
                label = f"{h}-{h_next}h<br>{count_h:,} ({pct_h:.1f}%)"
                y_cats.append(label)
                stats_left[h] = (label, count_h, pct_h)

            for h in shifted_order:
                h_next = (h + 1) % 24
                x_vals = sub.loc[sub["hour"] == h, "time"].values
                if len(x_vals) > 500:
                    x_vals = np.random.choice(x_vals, 500, replace=False)
                label, count_h, pct_h = stats_left[h]

                if len(x_vals) < 2:
                    fig.add_trace(go.Violin(x=[None], y=[label], name=label, showlegend=False, hoverinfo='skip'), row=r, col=1)
                    continue

                fig.add_trace(go.Violin(
                    x=x_vals.tolist(),
                    y=[label] * len(x_vals),
                    name=label, line_color=VIOLIN_COLORS[0],
                    fillcolor=VIOLIN_FILLS[0], box_visible=True, meanline_visible=True,
                    points='outliers', showlegend=False, orientation='h', spanmode='hard',
                    hovertemplate=(
                        f"<b>{h}h–{h_next}h</b><br>"
                        f"Mốc: {time_labels[0]}<br>"
                        f"Số bill: {count_h:,} ({pct_h:.1f}%)<br>"
                        f"Thời gian vc: %{{x:.1f}}h<br>"
                        f"Cặp: {pair}<extra></extra>"
                    ),
                ), row=r, col=1)

            fig.update_xaxes(title_text="Thời gian vc (h)", showgrid=True, gridcolor="#E5E7EB", row=r, col=1)
            fig.update_yaxes(categoryorder="array", categoryarray=y_cats[::-1], showgrid=True, gridcolor="#E5E7EB", row=r, col=1)

        # ==========================================
        # Cột 2 (hình 4): Tương quan time_labels[1] theo time_labels[0]
        # ==========================================
        t_col_1, t_col_2 = time_cols[0], time_cols[1]
        sub2_raw = df.loc[df["pair"] == pair, [t_col_1, t_col_2]].dropna().copy()
        if not sub2_raw.empty:
            dt1 = pd.to_datetime(sub2_raw[t_col_1])
            dt2 = pd.to_datetime(sub2_raw[t_col_2])

            # Tính giờ đến dưới dạng offset từ 0h ngày xuất phát
            # VD: xuất phát 22h ngày 1, đến 3h ngày 2 → hour_2_cont = 27h
            departure_date = dt1.dt.normalize()  # 0h ngày xuất phát
            hour_2_offset = (dt2 - departure_date).dt.total_seconds() / 3600.0
            sub2_raw["hour_2_cont"] = hour_2_offset
            sub2_raw["hour_1"] = dt1.dt.hour

            # Lọc bỏ outlier: chỉ giữ 0 ≤ offset ≤ 36h
            sub2_raw = sub2_raw[(sub2_raw["hour_2_cont"] >= 0) & (sub2_raw["hour_2_cont"] <= 36)].copy()

            total_bills_2 = len(sub2_raw)

            y_cats_right = []
            stats_right = {}
            hour_shift = pair_shifts.get(pair, 0)
            shifted_order = [(i + hour_shift) % 24 for i in range(24)]
            for h in shifted_order:
                h_next = (h + 1) % 24
                count_h = len(sub2_raw[sub2_raw["hour_1"] == h])
                pct_h = (count_h / total_bills_2 * 100) if total_bills_2 > 0 else 0
                label = f"{h}-{h_next}h<br>{count_h:,} ({pct_h:.1f}%)"
                y_cats_right.append(label)
                stats_right[h] = (label, count_h, pct_h)

            for h in shifted_order:
                h_next = (h + 1) % 24
                x_vals = sub2_raw.loc[sub2_raw["hour_1"] == h, "hour_2_cont"].values
                if len(x_vals) > 500:
                    x_vals = np.random.choice(x_vals, 500, replace=False)
                label = stats_right[h][0]

                if len(x_vals) < 2:
                    fig.add_trace(go.Violin(x=[None], y=[label], name=label, showlegend=False, hoverinfo='skip'), row=r, col=2)
                    continue

                fig.add_trace(go.Violin(
                    x=x_vals.tolist(), y=[label] * len(x_vals),
                    name=label, line_color="#F59E0B", fillcolor="rgba(245,158,11,0.15)",
                    box_visible=True, meanline_visible=True, points='outliers',
                    showlegend=False, orientation='h', spanmode='hard',
                    hovertemplate=(f"<b>{h}h–{h_next}h tại {time_labels[0]}</b><br>Giờ đến {time_labels[1]}: %{{x:.1f}}h<extra></extra>")
                ), row=r, col=2)

            # Trục X: 0-36h, tick labels hiển thị giờ thực (mod 24)
            _tv = list(range(0, 37, 3))
            _tt = [f"{v % 24}h" + ("" if v < 24 else " (+1d)") for v in _tv]
            fig.update_xaxes(title_text=f"Giờ đến {time_labels[1]} (offset từ ngày xuất phát)", range=[0, 36], tickvals=_tv, ticktext=_tt, showgrid=True, gridcolor="#E5E7EB", row=r, col=2)
            fig.update_yaxes(categoryorder="array", categoryarray=y_cats_right[::-1], title_text=f"Khung giờ {time_labels[0]}", showgrid=True, gridcolor="#E5E7EB", row=r, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} kho bưu cục theo {sort_label} — Violin: phân phối T.gian VC & tương quan giờ đến</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=total_plot_height + 300, width=FIG_WIDTH , showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=110, b=30, l=70, r=40), violingap=0)
    return fig

def add_dt_regression_traces(fig, r, c, x_vals, y_vals, hour_shift, line_color='#EF4444', fill_color='rgba(239, 68, 68, 0.15)', name_prefix=""):
    """Tính toán và vẽ đường hồi quy Step Function bằng Decision Tree"""
    if len(x_vals) < 10:
        return
        
    x_arr = x_vals.values if isinstance(x_vals, pd.Series) else np.array(x_vals)
    y_arr = y_vals.values if isinstance(y_vals, pd.Series) else np.array(y_vals)
    
    try:
        # Giới hạn độ sâu để không bị nhiễu (Overfit). max_depth=4 ~ tối đa 16 bậc
        # min_samples_leaf=0.05 đảm bảo mỗi luồng chứa ít nhất 5% lượng bill
        dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05, random_state=42)
        X_train = x_arr.reshape(-1, 1)
        dt.fit(X_train, y_arr)
        
        # Sinh các điểm trên trục X để vẽ (tăng độ mịn để nét bậc thang sắc nét)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 300)
        X_test = x_line.reshape(-1, 1)
        y_line = dt.predict(X_test)
        
        real_x_line = (x_line + hour_shift) % 24
        
        # Tính độ lệch chuẩn (std) cho từng cụm (leaf node) để vẽ dải PI
        leaves_train = dt.apply(X_train)
        leaf_std = {}
        for leaf_idx in np.unique(leaves_train):
            std_val = np.std(y_arr[leaves_train == leaf_idx])
            leaf_std[leaf_idx] = max(std_val, 1e-6)
            
        leaves_test = dt.apply(X_test)
        std_line = np.array([leaf_std[idx] for idx in leaves_test])
        
        y_upper = y_line + 1.96 * std_line
        y_lower = y_line - 1.96 * std_line
        
        x_ci = np.concatenate([x_line, x_line[::-1]]).tolist()
        y_ci = np.concatenate([y_upper, y_lower[::-1]]).tolist()
        
        # Vẽ dải PI (Prediction Interval)
        fig.add_trace(go.Scatter(
            x=x_ci, y=y_ci, fill='toself', fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False, name=f"{name_prefix}PI"
        ), row=r, col=c)
        
        # Vẽ đường Step Function
        fig.add_trace(go.Scatter(
            x=x_line.tolist(), y=y_line.tolist(),
            customdata=real_x_line.tolist(),
            mode='lines', line=dict(color=line_color, width=3, shape='vh'),
            name=f"{name_prefix}Trend", showlegend=False,
            hovertemplate="Trục X (thực tế): %{customdata:.2f}h<br>Dự báo (Y): %{y:.1f}<extra></extra>"
        ), row=r, col=c)
                    
    except Exception as e:
        print(f"  !! Lỗi vẽ DT regression (row={r}, col={c}): {e}")

def build_scatter_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label, pair_shifts, pair_shifts2, color_col=None):
    """
    Scatter plot với tùy chọn tô màu theo nhãn phân loại (color_col).
    Nếu color_col được truyền vào (VD: 'VD_type'), mỗi giá trị sẽ có 1 màu riêng.
    """
    rows = len(pairs)
    cols = 2
    
    ROW_H = 600
    gap_px = 80
    total_plot_height = rows * ROW_H + (rows - 1) * gap_px

    subplot_titles = []
    for pair in pairs:
        subplot_titles.append(f"{pair}<br>({time_labels[1]} vs T.gian VC)")
        subplot_titles.append(f"{pair}<br>({time_labels[0]} vs Giờ đến)")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                         vertical_spacing=gap_px / max(total_plot_height, 1))

    # --- Xây dựng bảng màu cho color_col ---
    cat_color_map = {}
    if color_col and color_col in df.columns:
        # Lấy top categories theo tần suất, còn lại gộp vào "Khác"
        cat_counts = df[color_col].fillna("N/A").value_counts()
        max_cats = len(CATEGORY_COLORS) - 1  # dành 1 slot cho "Khác"
        top_cats = cat_counts.head(max_cats).index.tolist()
        for i, cat in enumerate(top_cats):
            cat_color_map[cat] = CATEGORY_COLORS[i]
        cat_color_map["Khác"] = CATEGORY_COLORS[max_cats]
    
    # Theo dõi legend đã hiện chưa (chỉ hiện 1 lần cho toàn bộ figure)
    legend_shown = set()

    for p, pair in enumerate(pairs):
        r = p + 1
        t_col_1 = time_cols[0]
        t_col_2 = time_cols[1]
        
        hour_shift = pair_shifts.get(pair, 0)
        hour_shift2 = pair_shifts2.get(pair, 0)

        # --- CỘT 1 (Trái): X = Giờ đến (shifted), Y = Thời gian vận chuyển ---
        needed_cols_3 = [t_col_2, "time"]
        if color_col and color_col in df.columns:
            needed_cols_3.append(color_col)
        sub3 = df.loc[df["pair"] == pair, needed_cols_3].dropna(subset=[t_col_2, "time"])
        sub3 = sub3[(sub3["time"] <= 100) & (sub3['time'] >= 0)].copy()

        if not sub3.empty:
            q1_3 = sub3["time"].quantile(0.1)
            q3_3 = sub3["time"].quantile(0.9)
            sub3 = sub3[(sub3["time"] >= q1_3) & (sub3["time"] <= q3_3)].copy()

            if not sub3.empty:
                dt3 = pd.to_datetime(sub3[t_col_2])
                sub3["_h_bin"] = dt3.dt.hour
                total_bills_3 = len(sub3)
                hour_counts_3 = sub3["_h_bin"].value_counts()
                valid_hours_3 = hour_counts_3[hour_counts_3 >= 0.01 * total_bills_3].index
                sub3 = sub3[sub3["_h_bin"].isin(valid_hours_3)].copy()
                
            if not sub3.empty:
                dt3 = pd.to_datetime(sub3[t_col_2])
                raw_hours3 = dt3.dt.hour + dt3.dt.minute / 60.0 + dt3.dt.second / 3600.0
                sub3["_x_shifted"] = (raw_hours3 - hour_shift2) % 24
                x3_vals = sub3["_x_shifted"]
                y3_vals = sub3["time"]
                
                n3 = len(x3_vals)
                # Vẽ Decision Tree Regression (Tạm ẩn theo yêu cầu)
                # add_dt_regression_traces(fig, r, 1, x3_vals, y3_vals, hour_shift2)

                # Sampling nếu quá nhiều điểm
                if n3 > 10000:
                    sample_idx3 = sub3.sample(10000, random_state=42).index
                    sub3_plot = sub3.loc[sample_idx3]
                else:
                    sub3_plot = sub3
                
                x3_plot = sub3_plot["_x_shifted"]
                y3_plot = sub3_plot["time"]
                real_hours_plot3 = (x3_plot + hour_shift2) % 24

                # --- Vẽ scatter 1 màu mặc định (Bỏ tô màu theo VD_type) ---
                fig.add_trace(go.Scatter(
                    x=x3_plot.tolist(), y=y3_plot.tolist(),
                    customdata=real_hours_plot3.tolist(),
                    mode='markers', marker=dict(size=4, color="#10B981", opacity=0.4),
                    name="Bill", showlegend=False,
                    hovertemplate=f"Đến ({time_labels[1]}): %{{customdata:.2f}}h<br>T.gian VC: %{{y:.1f}}h<br>Cặp: {pair}<extra></extra>"
                ), row=r, col=1)

                _tv_s3 = list(range(0, 25, 2))
                _tt_s3 = [f"{int((v + hour_shift2) % 24)}h" for v in _tv_s3]
                fig.update_xaxes(title_text=f"Giờ đến ({time_labels[1]})", range=[0, 24], tickvals=_tv_s3, ticktext=_tt_s3, showgrid=True, gridcolor="#E5E7EB", row=r, col=1)
                fig.update_yaxes(title_text=f"Thời gian VC (h)", rangemode="tozero", showgrid=True, gridcolor="#E5E7EB", row=r, col=1)

        # --- CỘT 2 (Phải): X = Giờ xuất phát, Y = Giờ đến (offset) ---
        needed_cols_2 = [t_col_1, t_col_2]
        if color_col and color_col in df.columns:
            needed_cols_2.append(color_col)
        sub2 = df.loc[df["pair"] == pair, needed_cols_2].dropna(subset=[t_col_1, t_col_2]).copy()
        if not sub2.empty:
            dt1 = pd.to_datetime(sub2[t_col_1])
            dt2 = pd.to_datetime(sub2[t_col_2])
            
            departure_date = dt1.dt.normalize()
            y2_vals_raw = (dt2 - departure_date).dt.total_seconds() / 3600.0
            
            raw_hours2 = dt1.dt.hour + dt1.dt.minute / 60.0 + dt1.dt.second / 3600.0
            x2_vals_raw = (raw_hours2 - hour_shift) % 24
            
            sub2["x2"] = x2_vals_raw
            sub2["y2"] = y2_vals_raw
            
            sub2 = sub2[(sub2["y2"] >= 0) & (sub2["y2"] <= 36)].copy()
            
            if not sub2.empty:
                q1_2 = sub2["y2"].quantile(0.05)
                q3_2 = sub2["y2"].quantile(0.95)
                sub2 = sub2[(sub2["y2"] >= q1_2) & (sub2["y2"] <= q3_2)].copy()

            if not sub2.empty:
                sub2["_h_bin"] = pd.to_datetime(sub2[t_col_1]).dt.hour
                total_bills_2 = len(sub2)
                hour_counts_2 = sub2["_h_bin"].value_counts()
                valid_hours_2 = hour_counts_2[hour_counts_2 >= 0.01 * total_bills_2].index
                sub2 = sub2[sub2["_h_bin"].isin(valid_hours_2)].copy()
            
            if not sub2.empty:
                x2_vals = sub2["x2"]
                y2_vals = sub2["y2"]
                n2 = len(x2_vals)
                
                # Vẽ Decision Tree Regression (Tạm ẩn theo yêu cầu)
                # add_dt_regression_traces(fig, r, 2, x2_vals, y2_vals, hour_shift)
                
                # Sampling nếu quá nhiều điểm
                if n2 > 10000:
                    sample_idx2 = sub2.sample(10000, random_state=42).index
                    sub2_plot = sub2.loc[sample_idx2]
                else:
                    sub2_plot = sub2
                
                x2_plot = sub2_plot["x2"]
                y2_plot = sub2_plot["y2"]
                real_hours_plot2 = (x2_plot + hour_shift) % 24

                # Chạy DBSCAN
                y_for_cluster = y2_plot.values.reshape(-1, 1)
                dbscan = DBSCAN(eps=0.5, min_samples=50)
                clusters = dbscan.fit_predict(y_for_cluster)
                sub2_plot = sub2_plot.copy()
                sub2_plot["_cluster"] = clusters
                sub2_plot["_real_h"] = real_hours_plot2
                
                # Bảng màu cho các cụm chính
                cluster_colors = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6"]
                
                def hex_to_rgba(hex_str, alpha):
                    h = hex_str.lstrip('#')
                    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
                
                unique_clusters = sorted(sub2_plot["_cluster"].unique())
                for clus in unique_clusters:
                    mask = sub2_plot["_cluster"] == clus
                    if mask.sum() == 0:
                        continue
                        
                    x_clus = sub2_plot.loc[mask, "x2"]
                    y_clus = sub2_plot.loc[mask, "y2"]
                        
                    if clus == -1:
                        c_color = "rgba(156, 163, 175, 0.4)" # Xám nhạt cho điểm nhiễu
                        c_name = "Nhiễu (Noise)"
                        opacity_val = 0.2
                    else:
                        c_color = cluster_colors[clus % len(cluster_colors)]
                        c_name = f"Cụm {clus+1}"
                        opacity_val = 0.6
                        
                        # Vẽ Decision Tree riêng cho cụm này
                        fill_c = hex_to_rgba(c_color, 0.15)
                        add_dt_regression_traces(fig, r, 2, x_clus, y_clus, hour_shift, line_color="#000000", fill_color=fill_c, name_prefix=f"{c_name} - ")
                        
                    show_leg = c_name not in legend_shown
                    legend_shown.add(c_name)
                    fig.add_trace(go.Scatter(
                        x=sub2_plot.loc[mask, "x2"].tolist(),
                        y=sub2_plot.loc[mask, "y2"].tolist(),
                        customdata=sub2_plot.loc[mask, "_real_h"].tolist(),
                        mode='markers',
                        marker=dict(size=4, color=c_color, opacity=opacity_val),
                        name=c_name, legendgroup=c_name, showlegend=show_leg,
                        hovertemplate=(
                            f"Xuất phát ({time_labels[0]}): %{{customdata:.2f}}h<br>"
                            f"Giờ đến: %{{y:.1f}}h<br>"
                            f"Nhóm: {c_name}<br>"
                            f"Cặp: {pair}<extra></extra>"
                        ),
                    ), row=r, col=2)

                
                tv_x2 = list(range(0, 25, 2))
                tt_x2 = [f"{int((v + hour_shift) % 24)}h" for v in tv_x2]
                
                tv_y2 = list(range(0, 37, 3))
                tt_y2 = [f"{v % 24}h" + ("" if v < 24 else " (+1d)") for v in tv_y2]
                
                fig.update_xaxes(title_text=f"Giờ xuất phát ({time_labels[0]})", range=[0, 24], tickvals=tv_x2, ticktext=tt_x2, showgrid=True, gridcolor="#E5E7EB", row=r, col=2)
                fig.update_yaxes(title_text=f"Giờ tại {time_labels[1]}", range=[0, 36], tickvals=tv_y2, ticktext=tt_y2, showgrid=True, gridcolor="#E5E7EB", row=r, col=2)

    color_label = f" — Phân cụm (DBSCAN)"
    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} kho bưu cục theo {sort_label} — Scatter: Tương quan Thời gian vận chuyển và Giờ đến{color_label}</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=total_plot_height + 120, width=FIG_WIDTH,
        showlegend=True,
        legend=dict(
            title=dict(text="Cụm dữ liệu", font=dict(size=12)),
            font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#D1D5DB", borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ) if cat_color_map else dict(),
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=140 if cat_color_map else 110, b=30, l=70, r=40))
    return fig


def write_combined_html(fig_bar, fig_violin, fig_scatter, out_file, page_title):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # Khôi phục render HTML cho 2 tab
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


def build_all(df, col_a, col_b, sort_by, title_prefix, out_file, wt_col="actual_weight", time_cols=None, time_labels=None, is_dest_flow=False, buu_cuc_set=None, color_col=None):
    required = {col_a, col_b, "bill_code", wt_col, "time"}
    if time_cols:
        required.update(time_cols)
    validate_columns(df, required, out_file)

    df, top = top_pairs(df, col_a, col_b, sort_by, wt_col, is_dest_flow, buu_cuc_set)
    sort_label = "Số bill" if sort_by == "bill_count" else "Tổng kg"
    pairs = top["pair"].tolist()
    if not pairs:
        print(f"  !! Không có dữ liệu cho {out_file}, bỏ qua.")
        return

    fig_bar = build_bar_fig(df, pairs, top, sort_label, title_prefix)
    if time_cols:
        t_col = time_cols[0]
        t_col_2 = time_cols[1]
        pair_shifts = {}
        pair_shifts2 = {}
        for pair in pairs:
            sub = df.loc[df["pair"] == pair, t_col].dropna()
            pair_shifts[pair] = find_optimal_shift(sub)
            sub2 = df.loc[df["pair"] == pair, t_col_2].dropna()
            pair_shifts2[pair] = find_optimal_shift(sub2)

        fig_violin = build_violin_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label, pair_shifts)
        fig_scatter = build_scatter_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label, pair_shifts, pair_shifts2, color_col=color_col)
        write_combined_html(fig_bar, fig_violin, fig_scatter, out_file, f"{title_prefix} – Top {N_TOP} {sort_label}")
    else:
        fig_bar.write_html(out_file, include_plotlyjs="cdn")
        print(f"  ==> {out_file}")


if __name__ == "__main__":
    print("Đọc dữ liệu kho...")
    wh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "warehouse.csv")
    wh_df = pd.read_csv(wh_path)
    buu_cuc_set = set(wh_df[wh_df['Bưu Cục'] == 'Y']['name'].dropna().str.strip())

    COLOR_COL = "VD_type"  
    bill_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bill.csv")
    print(f"Đọc bill.csv (lấy cột '{COLOR_COL}')...")
    bill_df = pd.read_csv(bill_path, usecols=["bill_code", COLOR_COL])
    print(f"  bill.csv: {len(bill_df):,} dòng, {bill_df[COLOR_COL].nunique()} loại {COLOR_COL}")

    print("Đọc dữ liệu...")
    df_o = pd.read_csv(os.path.join(INPUT_DIR, "origin_head.csv"))
    df_d = pd.read_csv(os.path.join(INPUT_DIR, "destination_tail.csv"))
    print(f"  origin_to_1A : {len(df_o):,} bill")
    print(f"  1A_to_dest   : {len(df_d):,} bill")

    df_o = df_o.merge(bill_df, on="bill_code", how="left")
    df_d = df_d.merge(bill_df, on="bill_code", how="left")
    print(f"  Merge {COLOR_COL}: origin matched {df_o[COLOR_COL].notna().sum():,}/{len(df_o):,}, "
          f"dest matched {df_d[COLOR_COL].notna().sum():,}/{len(df_d):,}")

    print(f"\nVẽ top {N_TOP} cặp kho BƯU CỤC NHIỀU BILL nhất...")
    build_all(df_o, "kho_o", "kho_o1a", "bill_count", "Kho gửi → Kho 1A nguồn",
              os.path.join(OUTPUT_DIR, f"top{N_TOP}_bill_origin.html"), 
              time_cols=["time_o", "time_o1a"], time_labels=["Giờ ra khỏi Kho đầu", "Giờ đến Kho 1A"],
              is_dest_flow=False, buu_cuc_set=buu_cuc_set, color_col=COLOR_COL)
    build_all(df_d, "kho_d1a", "kho_d", "bill_count", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, f"top{N_TOP}_bill_dest.html"), 
              time_cols=["time_d1a", "time_d"], time_labels=["Giờ ra khỏi Kho 1A", "Giờ đến Kho đích"],
              is_dest_flow=True, buu_cuc_set=buu_cuc_set, color_col=COLOR_COL)

    print(f"\nVẽ top {N_TOP} cặp kho BƯU CỤC NHIỀU KG nhất...")
    build_all(df_o, "kho_o", "kho_o1a", "total_kg", "Kho gửi → Kho 1A nguồn",
              os.path.join(OUTPUT_DIR, f"top{N_TOP}_kg_origin.html"), 
              time_cols=["time_o", "time_o1a"], time_labels=["Giờ ra khỏi Kho đầu", "Giờ đến Kho 1A"],
              is_dest_flow=False, buu_cuc_set=buu_cuc_set, color_col=COLOR_COL)
    build_all(df_d, "kho_d1a", "kho_d", "total_kg", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, f"top{N_TOP}_kg_dest.html"), 
              time_cols=["time_d1a", "time_d"], time_labels=["Giờ ra khỏi Kho 1A", "Giờ đến Kho đích"],
              is_dest_flow=True, buu_cuc_set=buu_cuc_set, color_col=COLOR_COL)

    print(f"\nXong! 4 file HTML (bar + violin + scatter) đã được lưu. Scatter tô màu theo '{COLOR_COL}'.")