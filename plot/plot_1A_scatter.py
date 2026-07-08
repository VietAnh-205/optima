import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ───────────── Đọc warehouse ─────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
WH_PATH      = os.path.join(SCRIPT_DIR, '..', 'warehouse.csv')
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, '..', 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

wh_df = pd.read_csv(WH_PATH)
wh_df['name'] = wh_df['name'].astype(str).str.strip()
buu_cuc_set = set(wh_df.loc[wh_df['Bưu Cục'].astype(str).str.strip() == 'Y', 'name'])


def add_regression_band(fig, x_arr, y_arr, x_line, row):
    """Tính Regression Line + 95% PI và vẽ lên subplot 'row'."""
    n = len(x_arr)
    if n < 3:
        return
    try:
        p_coef, _ = np.polyfit(x_arr, y_arr, 1, cov=True)
        m, c = p_coef

        y_line = m * x_line + c

        y_err = y_arr - (m * x_arr + c)
        dof   = max(n - 2, 1)
        s_err = np.sqrt(np.sum(y_err**2) / dof)
        t_val = stats.t.ppf(0.975, dof)

        mean_x = np.mean(x_arr)
        ss_x   = max(np.sum((x_arr - mean_x)**2), 1e-6)

        # Prediction Interval
        pi      = t_val * s_err * np.sqrt(1 + 1/n + (x_line - mean_x)**2 / ss_x)
        y_upper = y_line + pi
        y_lower = y_line - pi

        # 1. PI Band (vẽ trước để nằm dưới)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]).tolist(),
            y=np.concatenate([y_upper, y_lower[::-1]]).tolist(),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip', showlegend=False, name='95% PI'
        ), row=row, col=1)

        # 2. Trend Line (vẽ sau để nằm trên)
        fig.add_trace(go.Scatter(
            x=x_line.tolist(), y=y_line.tolist(),
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False, name='Trend',
            hovertemplate='X: %{x:.2f}h<br>Dự báo Y: %{y:.2f}h<extra></extra>'
        ), row=row, col=1)

    except Exception as e:
        print(f"  !! Regression error (row={row}): {e}")


def process_and_plot(file_path, col_start, col_end, col_buu_cuc_check,
                     title_prefix, output_filename):
    print(f"\nLoading {file_path}...")
    df = pd.read_csv(file_path)

    # Tạo tên cặp kho + đánh dấu Bưu Cục theo cột cần check
    df['pair'] = df[col_start].astype(str).str.strip() + " → " + df[col_end].astype(str).str.strip()
    df['is_buu_cuc'] = df[col_buu_cuc_check].astype(str).str.strip().isin(buu_cuc_set)

    # Tổng số bill mỗi cặp
    agg = (df.groupby(['pair', 'is_buu_cuc'])['bill_code']
             .nunique().reset_index(name='so_bill'))

    # Top 10 Bưu Cục & Top 10 Thường
    top_bc  = (agg[agg['is_buu_cuc']]
                  .nlargest(10, 'so_bill')['pair']
                  .tolist())
    top_thu = (agg[~agg['is_buu_cuc']]
                  .nlargest(10, 'so_bill')['pair']
                  .tolist())

    # Thêm nhãn
    pairs_bc  = [f"{p} [Bưu Cục]" for p in top_bc]
    pairs_thu = [f"{p} [Thường]"  for p in top_thu]
    df.loc[df['pair'].isin(top_bc),  'pair_label'] = df.loc[df['pair'].isin(top_bc),  'pair'].map(lambda x: f"{x} [Bưu Cục]")
    df.loc[df['pair'].isin(top_thu), 'pair_label'] = df.loc[df['pair'].isin(top_thu), 'pair'].map(lambda x: f"{x} [Thường]")

    all_pairs_raw    = top_bc  + top_thu
    all_pairs_labels = pairs_bc + pairs_thu

    N = len(all_pairs_labels)  # 20

    # Tiêu đề subplot: mỗi cặp → 2 hình
    subplot_titles = []
    for label in all_pairs_labels:
        subplot_titles.append(f"{label}<br><sup>Heatmap 24x24: Giờ Đến 1A vs Giờ Checkout 1A</sup>")
        subplot_titles.append(f"{label}<br><sup>Scatter: Thời gian lưu kho (Y) vs Giờ IN 1A (X) — cắt outlier 95%</sup>")

    total_rows = N * 2
    fig = make_subplots(rows=total_rows, cols=1,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.008)
    coloraxis_updates = {}  # lưu cấu hình coloraxis riêng cho từng heatmap

    for i, (pair_raw, pair_label) in enumerate(zip(all_pairs_raw, all_pairs_labels)):
        sub_df = df[df['pair'] == pair_raw].copy()

        arrival_dt  = pd.to_datetime(sub_df['time_1a_in'])
        checkout_dt = pd.to_datetime(sub_df['time_1a_out'])
        duration    = sub_df['time_in_1a']

        hour_in  = arrival_dt.dt.hour  + arrival_dt.dt.minute  / 60.0
        hour_out = checkout_dt.dt.hour + checkout_dt.dt.minute / 60.0

        # Downsample for display (regression vẫn dùng full data)
        full_hour_in  = hour_in.values
        full_hour_out = hour_out.values
        full_duration = duration.values

        if len(sub_df) > 10000:
            sample_idx  = sub_df.sample(10000, random_state=42).index
            disp_hin    = hour_in.loc[sample_idx]
            disp_hout   = hour_out.loc[sample_idx]
            disp_dur    = duration.loc[sample_idx]
        else:
            disp_hin  = hour_in
            disp_hout = hour_out
            disp_dur  = duration

        row_1 = i * 2 + 1
        row_2 = i * 2 + 2

        color_dot = '#1B4EF5' if '[Bưu Cục]' in pair_label else '#1B4EF5'

        # ────── Hình 1: Heatmap 24x24 (Giờ đến 1A vs Giờ checkout 1A) ──────
        mask1 = ~(np.isnan(full_hour_out) | np.isnan(full_hour_in))
        if mask1.sum() > 0:
            # Bucket vào ô giờ (0..23)
            bin_out = np.floor(full_hour_out[mask1]).clip(0, 23).astype(int)
            bin_in  = np.floor(full_hour_in[mask1]).clip(0, 23).astype(int)

            # Tạo ma trận đếm 24x24: hàng = giờ đến (Y), cột = giờ checkout (X)
            heatmap_z = np.zeros((24, 24), dtype=int)
            for xi, yi in zip(bin_out, bin_in):
                heatmap_z[yi, xi] += 1

            tick_labels = [f"{h}h" for h in range(24)]

            # Cắt zmax tại p95 của các ô khác 0 để màu sắc rõ nét hơn
            nonzero_vals = heatmap_z[heatmap_z > 0]
            if len(nonzero_vals) > 0:
                z_max = int(np.percentile(nonzero_vals, 95))
                z_max = max(z_max, 1)   # đảm bảo ít nhất = 1
            else:
                z_max = 1

            # Mỗi heatmap dùng coloraxis riêng (c1, c3, c5, ...) theo row_1
            ca_key = f"coloraxis{row_1}"
            coloraxis_updates[ca_key] = dict(
                colorscale='Blues',
                cmin=0,
                cmax=z_max,
                colorbar=dict(
                    title=dict(text="Số bill", side="right"),
                    thickness=12,
                    len=0.9 / total_rows,   # độ dài thanh màu thu gọn theo tủ lệ subplot
                    y=(1 - (row_1 - 0.5) / total_rows),
                    yanchor='middle'
                )
            )

            fig.add_trace(go.Heatmap(
                z=heatmap_z.tolist(),
                x=tick_labels,
                y=tick_labels,
                coloraxis=ca_key,
                zmin=0,
                zmax=z_max,
                hovertemplate="Checkout: %{x}<br>Đến 1A: %{y}<br>Số bill: %{z}<extra></extra>",
                name=f"H1: {pair_label}"
            ), row=row_1, col=1)

            fig.update_xaxes(title_text="Giờ checkout 1A", row=row_1, col=1)
            fig.update_yaxes(title_text="Giờ đến 1A", row=row_1, col=1)

        # ────── Hình 2: Scatter lưu kho (Y) vs Giờ IN (X) — cắt outlier 95% ──────
        mask2 = ~(np.isnan(full_hour_in) | np.isnan(full_duration))
        dur_valid = full_duration[mask2]
        hin_valid = full_hour_in[mask2]

        # Cắt outlier: chỉ giữ duration trong khoảng [0, p95]
        if len(dur_valid) > 0:
            p95 = np.percentile(dur_valid[dur_valid >= 0], 95) if (dur_valid >= 0).sum() > 0 else 100
            keep = (dur_valid >= 0) & (dur_valid <= p95)
            dur_plot = dur_valid[keep]
            hin_plot = hin_valid[keep]
        else:
            dur_plot = dur_valid
            hin_plot = hin_valid
            p95 = 100

        # Regression + PI đã bỏ theo yêu cầu — chỉ giữ scatter

        # Scatter dots (sample nếu cần)
        if len(hin_plot) > 10000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(hin_plot), 10000, replace=False)
            disp_hin2 = hin_plot[idx]
            disp_dur2 = dur_plot[idx]
        else:
            disp_hin2 = hin_plot
            disp_dur2 = dur_plot

        fig.add_trace(go.Scatter(
            x=disp_hin2.tolist(), y=disp_dur2.tolist(),
            mode='markers',
            marker=dict(opacity=0.25, size=4, color=color_dot),
            name=f"H2: {pair_label}",
            hovertemplate="Giờ IN 1A: %{x:.2f}h<br>Lưu kho: %{y:.2f}h<extra></extra>"
        ), row=row_2, col=1)

        fig.update_xaxes(title_text="Giờ IN 1A (0-24h)", range=[0, 24], dtick=2,
                         showgrid=True, gridcolor="#E5E7EB", row=row_2, col=1)
        fig.update_yaxes(title_text=f"Thời gian lưu 1A (h, ≤p95={p95:.1f}h)",
                         range=[0, max(p95 * 1.05, 1)],
                         showgrid=True, gridcolor="#E5E7EB", row=row_2, col=1)

    layout_extra = {k: v for k, v in coloraxis_updates.items()}
    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}</b><br><sup>Top 20 cặp kho (10 Bưu Cục + 10 Thường) — Heatmap giờ | Scatter lưu kho</sup>",
            x=0.5, xanchor='center', font=dict(size=15)),
        height=500 * total_rows,
        showlegend=False,
        plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial", size=11),
        margin=dict(t=100, b=30, l=80, r=40),
        **layout_extra
    )

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    fig.write_html(output_path)
    print(f"  ==> {output_path}")


# ─── Luồng 1: Kho gửi → Kho 1A (check Bưu Cục theo kho GỬI = col_start) ───
process_and_plot(
    file_path         = os.path.join(SCRIPT_DIR, '../output_all_traces/origin_inout_1a.csv'),
    col_start         = 'kho_o',
    col_end           = 'kho_o1a',
    col_buu_cuc_check = 'kho_o',   # Bưu Cục theo kho đầu (kho gửi)
    title_prefix      = 'Luồng Kho Gốc → Kho 1A',
    output_filename   = 'scatter_origin_inout_1a.html'
)

# ─── Luồng 2: Kho 1A → Kho nhận (check Bưu Cục theo kho NHẬN = col_end) ───
process_and_plot(
    file_path         = os.path.join(SCRIPT_DIR, '../output_all_traces/destination_inout_1a.csv'),
    col_start         = 'kho_d1a',
    col_end           = 'kho_d',
    col_buu_cuc_check = 'kho_d',   # Bưu Cục theo kho nhận (kho đích)
    title_prefix      = 'Luồng Kho 1A → Kho Đích',
    output_filename   = 'scatter_destination_inout_1a.html'
)

print("\nXong! 2 file HTML đã được lưu.")
