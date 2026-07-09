import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_all_traces")
OUTPUT_DIR = 'output_plot'
BINS   = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36, 48, 72, 120, float("inf")]
LABELS = ["0-1h","1-2h","2-3h","3-4h","4-5h","5-6h","6-8h","8-10h","10-12h",
          "12-18h","18-24h","24-36h","36-48h","48-72h","72-120h",">120h"]

BAR_COLOR, VIOLIN_COLOR = "#2563EB", "#1D4ED8"
N_TOP     = 10   # chỉ giữ top 10 kho bưu cục
FIG_WIDTH = 1400 
BAR_ROW_H = 1000

# Violin: mỗi cặp kho chỉ còn 1 hàng x 2 cột (hình 1 + hình 4), giảm resolution
V_COLS, V_ROWS_PER_PAIR = 2, 1
V_ROW_H = 600  # Chiều cao cho mỗi subplot (giảm so với trước)


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

    gap_px = 30
    V_PAIR_H = 1200  # giảm resolution so với bản gốc (trước đây 3000/cặp)
    total_plot_height = rows * V_PAIR_H + max(rows - 1, 0) * gap_px

    specs = [[{"type": "xy"}, {"type": "xy"}] for _ in pairs]

    subplot_titles = []
    for pair in pairs:
        subplot_titles.append(f"{pair}<br>({time_labels[0]} vs T.gian VC)")
        subplot_titles.append(f"{pair}<br>Tương quan: {time_labels[1]} theo {time_labels[0]}")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                         vertical_spacing=vspacing(rows, 0.02),
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
                        f"Thời gian vc: %{{x:.0f}}h<br>"
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
        height=total_plot_height + 120, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=110, b=30, l=70, r=40), violingap=0)
    return fig


def build_scatter_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label, pair_shifts):
    rows = len(pairs)
    cols = 1
    
    gap_px = 60
    S_PAIR_H = 400
    total_plot_height = rows * S_PAIR_H + (rows - 1) * gap_px

    subplot_titles = [f"{pair}<br>({time_labels[0]} vs {time_labels[1]})" for pair in pairs]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                         vertical_spacing=vspacing(rows, 0.02))

    for p, pair in enumerate(pairs):
        r = p + 1
        t_col_1 = time_cols[0]
        
        sub = df.loc[df["pair"] == pair, [t_col_1, "time"]].dropna()
        sub = sub[(sub["time"] <= 100) & (sub['time'] >= 0)].copy()

        if sub.empty:
            continue
            
        dt1 = pd.to_datetime(sub[t_col_1])
        
        # Shift giờ xuất phát theo hour_shift của cặp
        hour_shift = pair_shifts.get(pair, 0)
        raw_hours = dt1.dt.hour + dt1.dt.minute / 60.0 + dt1.dt.second / 3600.0
        x_vals = (raw_hours - hour_shift) % 24
        y_vals = sub["time"]
        
        # Calculate Regression line & Confidence Band on FULL data
        n = len(x_vals)
        if n > 2:
            x_arr = x_vals.values if isinstance(x_vals, pd.Series) else np.array(x_vals)
            y_arr = y_vals.values if isinstance(y_vals, pd.Series) else np.array(y_vals)
            
            try:
                p_coef, cov = np.polyfit(x_arr, y_arr, 1, cov=True)
                m, c = p_coef
                
                x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
                y_line = m * x_line + c
                
                y_err = y_arr - (m * x_arr + c)
                dof = n - 2
                s_err = np.sqrt(np.sum(y_err**2) / max(dof, 1))
                t_val = stats.t.ppf(0.975, max(dof, 1))
                
                mean_x = np.mean(x_arr)
                ss_x = np.sum((x_arr - mean_x)**2)
                
                # Prediction Interval Band (captures the spread of data points)
                pi = t_val * s_err * np.sqrt(1 + 1/n + (x_line - mean_x)**2 / max(ss_x, 1e-6))
                y_upper = y_line + pi
                y_lower = y_line - pi
                
                # Plot PI Band FIRST (so it sits behind markers)
                x_ci = np.concatenate([x_line, x_line[::-1]]).tolist()
                y_ci = np.concatenate([y_upper, y_lower[::-1]]).tolist()
                fig.add_trace(go.Scatter(
                    x=x_ci,
                    y=y_ci,
                    fill='toself',
                    fillcolor='rgba(239, 68, 68, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name="95% PI"

                ), row=r, col=1)
                
            except Exception as e:
                print(f"  !! Cannot plot regression for {pair}: {e}")
                x_line, y_line = None, None
        else:
            x_line, y_line = None, None

        # Downsample scatter points for rendering if too many
        if len(x_vals) > 10000:
            sample_idx = sub.sample(10000, random_state=42).index
            x_plot = x_vals.loc[sample_idx]
            y_plot = y_vals.loc[sample_idx]
        else:
            x_plot = x_vals
            y_plot = y_vals

        # Plot Scatter markers using standard Scatter (not Scattergl) to avoid WebGL overlap bugs
        # Tính giờ thực để hiển thị đúng trong hover
        real_hours_plot = (x_plot + hour_shift) % 24
        fig.add_trace(go.Scatter(
            x=x_plot.tolist(), y=y_plot.tolist(),
            customdata=real_hours_plot.tolist(),
            mode='markers',
            marker=dict(size=4, color="#1B4EF5", opacity=0.4),
            name="Bill",
            hovertemplate=f"Xuất phát ({time_labels[0]}): %{{customdata:.2f}}h<br>T.gian VC: %{{y:.1f}}h<br>Cặp: {pair}<extra></extra>"
        ), row=r, col=1)

        # Plot Trend Line on top of everything
        if x_line is not None:
            real_x_line = (x_line + hour_shift) % 24
            fig.add_trace(go.Scatter(
                x=x_line.tolist(), y=y_line.tolist(),
                customdata=real_x_line.tolist(),
                mode='lines',
                line=dict(color='red', width=2),
                name="Trend",
                hovertemplate="Xuất phát: %{customdata:.2f}h<br>T.gian VC (dự báo): %{y:.1f}h<extra></extra>"
            ), row=r, col=1)

        # Tick labels hiển thị giờ thực thay vì giá trị shifted
        _tv_s = list(range(0, 25, 2))
        _tt_s = [f"{int((v + hour_shift) % 24)}h" for v in _tv_s]
        fig.update_xaxes(title_text=f"Giờ xuất phát ({time_labels[0]})", range=[0, 24], tickvals=_tv_s, ticktext=_tt_s, showgrid=True, gridcolor="#E5E7EB", row=r, col=1)
        fig.update_yaxes(title_text=f"Thời gian VC (h)", rangemode="tozero", showgrid=True, gridcolor="#E5E7EB", row=r, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} kho bưu cục theo {sort_label} — Scatter: Giờ xuất phát vs Thời gian vận chuyển</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=total_plot_height + 120, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=110, b=30, l=70, r=40))
    return fig


# ── Figure 4: Scatter End-to-End (Kho gửi nguồn → Kho nhận đích) ──────────
def build_e2e_scatter_fig(df_e2e, pairs, title_prefix, sort_label, pair_shifts):
    """Scatter plot: X = giờ check OUT kho_o, Y = giờ check IN kho_d.
    Cả 2 trục dùng offset từ 0h ngày xuất phát (giống logic violin 2).
    Lọc bỏ > 72h."""
    rows = len(pairs)
    cols = 1

    gap_px = 60
    S_PAIR_H = 400
    total_plot_height = rows * S_PAIR_H + (rows - 1) * gap_px

    subplot_titles = [f"{pair}<br>(Giờ OUT Kho gửi vs Giờ IN Kho nhận)" for pair in pairs]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                         vertical_spacing=vspacing(rows, 0.02))

    for p, pair in enumerate(pairs):
        r = p + 1
        sub = df_e2e.loc[df_e2e["pair"] == pair, ["time_o", "time_d"]].dropna().copy()
        if sub.empty:
            continue

        dt_o = pd.to_datetime(sub["time_o"])
        dt_d = pd.to_datetime(sub["time_d"])
        departure_date = dt_o.dt.normalize()  # 0h ngày xuất phát

        x_offset = (dt_o - departure_date).dt.total_seconds() / 3600.0  # giờ OUT offset
        y_offset = (dt_d - departure_date).dt.total_seconds() / 3600.0  # giờ IN offset

        sub["x_off"] = x_offset.values
        sub["y_off"] = y_offset.values

        # Lọc bỏ outlier: chỉ giữ 0 ≤ offset ≤ 72h
        sub = sub[(sub["x_off"] >= 0) & (sub["x_off"] <= 48)
                & (sub["y_off"] >= 0) & (sub["y_off"] <= 72)].copy()
        if sub.empty:
            continue

        x_vals = sub["x_off"]
        y_vals = sub["y_off"]
        n = len(x_vals)

        # Regression + PI band
        x_line, y_line = None, None
        if n > 2:
            x_arr = x_vals.values
            y_arr = y_vals.values
            try:
                p_coef, cov = np.polyfit(x_arr, y_arr, 1, cov=True)
                m, c = p_coef
                x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
                y_line = m * x_line + c

                y_err = y_arr - (m * x_arr + c)
                dof = n - 2
                s_err = np.sqrt(np.sum(y_err**2) / max(dof, 1))
                t_val = stats.t.ppf(0.975, max(dof, 1))
                mean_x = np.mean(x_arr)
                ss_x = np.sum((x_arr - mean_x)**2)
                pi = t_val * s_err * np.sqrt(1 + 1/n + (x_line - mean_x)**2 / max(ss_x, 1e-6))

                x_ci = np.concatenate([x_line, x_line[::-1]]).tolist()
                y_ci = np.concatenate([y_line + pi, (y_line - pi)[::-1]]).tolist()
                fig.add_trace(go.Scatter(
                    x=x_ci, y=y_ci,
                    fill='toself', fillcolor='rgba(239, 68, 68, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=False, name="95% PI"
                ), row=r, col=1)
            except Exception as e:
                print(f"  !! Cannot plot E2E regression for {pair}: {e}")
                x_line, y_line = None, None

        # Downsample
        if n > 10000:
            sample_idx = sub.sample(10000, random_state=42).index
            x_plot = x_vals.loc[sample_idx]
            y_plot = y_vals.loc[sample_idx]
        else:
            x_plot = x_vals
            y_plot = y_vals

        # Tính giờ thực (mod 24) cho hover
        real_x = x_plot % 24
        real_y = y_plot % 24
        fig.add_trace(go.Scatter(
            x=x_plot.tolist(), y=y_plot.tolist(),
            customdata=np.column_stack([real_x.values, real_y.values]).tolist(),
            mode='markers',
            marker=dict(size=4, color="#7C3AED", opacity=0.4),
            name="Bill",
            hovertemplate=(
                "Giờ OUT Kho gửi: %{customdata[0]:.1f}h (offset %{x:.1f}h)<br>"
                "Giờ IN Kho nhận: %{customdata[1]:.1f}h (offset %{y:.1f}h)<br>"
                f"Cặp: {pair}<extra></extra>"
            )
        ), row=r, col=1)

        # Trend line
        if x_line is not None:
            real_xl = x_line % 24
            real_yl = y_line % 24
            fig.add_trace(go.Scatter(
                x=x_line.tolist(), y=y_line.tolist(),
                customdata=np.column_stack([real_xl, real_yl]).tolist(),
                mode='lines', line=dict(color='red', width=2),
                name="Trend",
                hovertemplate="OUT: %{customdata[0]:.1f}h → IN: %{customdata[1]:.1f}h<extra></extra>"
            ), row=r, col=1)

        # Tick labels: offset → giờ thực + (+Nd)
        def _offset_ticks(max_h, step=3):
            tv = list(range(0, int(max_h) + 1, step))
            tt = []
            for v in tv:
                label = f"{v % 24}h"
                if v >= 48:
                    label += " (+2d)"
                elif v >= 24:
                    label += " (+1d)"
                tt.append(label)
            return tv, tt

        tv_x, tt_x = _offset_ticks(48, 3)
        tv_y, tt_y = _offset_ticks(72, 6)
        fig.update_xaxes(title_text="Giờ OUT Kho gửi (offset từ 0h ngày xuất phát)",
                         range=[0, 48], tickvals=tv_x, ticktext=tt_x,
                         showgrid=True, gridcolor="#E5E7EB", row=r, col=1)
        fig.update_yaxes(title_text="Giờ IN Kho nhận (offset từ 0h ngày xuất phát)",
                         range=[0, 72], tickvals=tv_y, ticktext=tt_y,
                         showgrid=True, gridcolor="#E5E7EB", row=r, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top {N_TOP} kho bưu cục theo {sort_label} — Scatter: Giờ OUT Kho gửi vs Giờ IN Kho nhận (end-to-end)</sup>",
            x=0.5, xanchor="center", font=dict(size=15)),
        height=total_plot_height + 120, width=FIG_WIDTH, showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=110, b=30, l=70, r=40))
    return fig


def write_combined_html(fig_bar, fig_violin, fig_scatter, out_file, page_title, fig_e2e=None):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    bar_div    = pio.to_html(fig_bar, full_html=False, include_plotlyjs=False)
    violin_div = pio.to_html(fig_violin, full_html=False, include_plotlyjs=False)
    scatter_div = pio.to_html(fig_scatter, full_html=False, include_plotlyjs=False)
    e2e_div = pio.to_html(fig_e2e, full_html=False, include_plotlyjs=False) if fig_e2e else ""
    e2e_tab_btn = '<button class="tab-btn" onclick="openTab(event, \'E2EScatter\')">' + 'Orgin->Destination</button>' if fig_e2e else ""
    e2e_tab_content = f'<div id="E2EScatter" class="tabcontent">{e2e_div}</div>' if fig_e2e else ""
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
    {e2e_tab_btn}
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

  {e2e_tab_content}

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


def build_all(df, col_a, col_b, sort_by, title_prefix, out_file, wt_col="actual_weight", time_cols=None, time_labels=None, is_dest_flow=False, buu_cuc_set=None, df_e2e=None):
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
        pair_shifts = {}
        for pair in pairs:
            sub = df.loc[df["pair"] == pair, t_col].dropna()
            pair_shifts[pair] = find_optimal_shift(sub)

        fig_violin = build_violin_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label, pair_shifts)
        fig_scatter = build_scatter_fig(df, pairs, time_cols, time_labels, title_prefix, sort_label, pair_shifts)

        # E2E Scatter nếu có dữ liệu merge
        fig_e2e = None
        if df_e2e is not None:
            # Gắn cột "pair" cho df_e2e dựa trên cùng logic top_pairs
            e2e_with_pair = df_e2e.copy()
            e2e_with_pair["pair"] = e2e_with_pair[col_a].astype(str) + " → " + e2e_with_pair[col_b].astype(str)
            if buu_cuc_set is not None:
                check_col = col_b if is_dest_flow else col_a
                is_bc = e2e_with_pair[check_col].fillna("").astype(str).str.strip().isin(buu_cuc_set)
                e2e_with_pair.loc[is_bc, "pair"] = e2e_with_pair.loc[is_bc, "pair"] + " [Bưu Cục]"
                e2e_with_pair.loc[~is_bc, "pair"] = e2e_with_pair.loc[~is_bc, "pair"] + " [Thường]"
            e2e_filtered = e2e_with_pair[e2e_with_pair["pair"].isin(pairs)].copy()
            if not e2e_filtered.empty:
                fig_e2e = build_e2e_scatter_fig(e2e_filtered, pairs, title_prefix, sort_label, pair_shifts)

        write_combined_html(fig_bar, fig_violin, fig_scatter, out_file, f"{title_prefix} – Top {N_TOP} {sort_label}", fig_e2e=fig_e2e)
    else:
        fig_bar.write_html(out_file, include_plotlyjs="cdn")
        print(f"  ==> {out_file}")


if __name__ == "__main__":
    print("Đọc dữ liệu kho...")
    wh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "warehouse.csv")
    wh_df = pd.read_csv(wh_path)
    buu_cuc_set = set(wh_df[wh_df['Bưu Cục'] == 'Y']['name'].dropna().str.strip())

    print("Đọc dữ liệu...")
    df_o = pd.read_csv(os.path.join(INPUT_DIR, "origin_head.csv"))
    df_d = pd.read_csv(os.path.join(INPUT_DIR, "destination_tail.csv"))
    print(f"  origin_to_1A : {len(df_o):,} bill")
    print(f"  1A_to_dest   : {len(df_d):,} bill")

    # Merge origin + destination theo bill_code để có dữ liệu end-to-end
    df_e2e = df_o.merge(
        df_d[["bill_code", "kho_d1a", "time_d1a", "kho_d", "time_d"]],
        on="bill_code", how="inner"
    )
    print(f"  E2E merged   : {len(df_e2e):,} bill")

    print("\nVẽ top 10 cặp kho BƯU CỤC NHIỀU BILL nhất...")
    build_all(df_o, "kho_o", "kho_o1a", "bill_count", "Kho gửi → Kho 1A nguồn",
              os.path.join(OUTPUT_DIR, "top10_bill_origin.html"), 
              time_cols=["time_o", "time_o1a"], time_labels=["Giờ ra khỏi Kho đầu", "Giờ đến Kho 1A"],
              is_dest_flow=False, buu_cuc_set=buu_cuc_set, df_e2e=df_e2e)
    build_all(df_d, "kho_d1a", "kho_d", "bill_count", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, "top10_bill_dest.html"), 
              time_cols=["time_d1a", "time_d"], time_labels=["Giờ ra khỏi Kho 1A", "Giờ đến Kho đích"],
              is_dest_flow=True, buu_cuc_set=buu_cuc_set, df_e2e=df_e2e)

    print("\nVẽ top 10 cặp kho BƯU CỤC NHIỀU KG nhất...")
    build_all(df_o, "kho_o", "kho_o1a", "total_kg", "Kho gửi → Kho 1A nguồn",
              os.path.join(OUTPUT_DIR, "top10_kg_origin.html"), 
              time_cols=["time_o", "time_o1a"], time_labels=["Giờ ra khỏi Kho đầu", "Giờ đến Kho 1A"],
              is_dest_flow=False, buu_cuc_set=buu_cuc_set, df_e2e=df_e2e)
    build_all(df_d, "kho_d1a", "kho_d", "total_kg", "Kho 1A đích → Kho nhận",
              os.path.join(OUTPUT_DIR, "top10_kg_dest.html"), 
              time_cols=["time_d1a", "time_d"], time_labels=["Giờ ra khỏi Kho 1A", "Giờ đến Kho đích"],
              is_dest_flow=True, buu_cuc_set=buu_cuc_set, df_e2e=df_e2e)

    print("\nXong! 4 file HTML (bar + violin + scatter + E2E) đã được lưu.")