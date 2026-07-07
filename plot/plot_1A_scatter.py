import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def process_and_plot(file_path, col_start, col_end, title_prefix, output_filename):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Tạo tên cặp kho
    df['pair'] = df[col_start].astype(str) + " → " + df[col_end].astype(str)
    
    # Lọc top 20 cặp kho nhiều bill nhất
    agg = df.groupby("pair")["bill_code"].nunique().reset_index(name="so_bill")
    top_pairs = agg.nlargest(20, "so_bill")["pair"].tolist()
    
    # Tạo tiêu đề cho 40 hình
    subplot_titles = []
    for pair in top_pairs:
        subplot_titles.append(f"1. {pair}: Đến (Y) vs Checkout (X)")
        subplot_titles.append(f"2. {pair}: Thời gian lưu (Y) vs Giờ IN (X)")
        
    # Tạo subplot: 40 hàng, 1 cột
    fig = make_subplots(rows=40, cols=1, subplot_titles=subplot_titles, vertical_spacing=0.008)
    
    for i, pair in enumerate(top_pairs):
        sub_df = df[df['pair'] == pair].copy()
        
        # Lấy thời gian từ các cột mới thêm
        arrival_dt = pd.to_datetime(sub_df['time_1a_in'])
        checkout_dt = pd.to_datetime(sub_df['time_1a_out'])
        duration = sub_df['time_in_1a']
        
        # Khung giờ (0-24)
        hour_in = arrival_dt.dt.hour + arrival_dt.dt.minute / 60.0
        hour_out = checkout_dt.dt.hour + checkout_dt.dt.minute / 60.0
        
        # Sample để tránh file HTML quá nặng (WebGL context limits or heavy SVG)
        if len(hour_in) > 10000:
            sample_idx = sub_df.sample(10000, random_state=42).index
            hour_in = hour_in.loc[sample_idx]
            hour_out = hour_out.loc[sample_idx]
            duration = duration.loc[sample_idx]
            
        row_1 = i*2 + 1
        row_2 = i*2 + 2
        
        # Hình 1: Đến (Y) vs Checkout (X)
        fig.add_trace(
            go.Scatter(
                x=hour_out,
                y=hour_in,
                mode='markers',
                marker=dict(opacity=0.1, size=4, color='blue'),
                name=f"H1: {pair}"
            ),
            row=row_1, col=1
        )
        fig.update_xaxes(title_text="Khung giờ checkout 1A (0-24h)", range=[0, 24], dtick=2, row=row_1, col=1)
        fig.update_yaxes(title_text="Khung giờ đến 1A (0-24h)", range=[0, 24], dtick=2, row=row_1, col=1)
        
        # Hình 2: Thời gian lưu (Y) vs Giờ IN (X)
        fig.add_trace(
            go.Scatter(
                x=hour_in,
                y=duration,
                mode='markers',
                marker=dict(opacity=0.1, size=4, color='red'),
                name=f"H2: {pair}"
            ),
            row=row_2, col=1
        )
        fig.update_xaxes(title_text="Khung giờ IN 1A (0-24h)", range=[0, 24], dtick=2, row=row_2, col=1)
        fig.update_yaxes(title_text="Thời gian lưu kho 1A (Giờ)", rangemode='tozero', row=row_2, col=1)

    fig.update_layout(
        title_text=f"{title_prefix} - Top 20 Cặp Kho",
        height=400 * 40, # Chiều cao 350px mỗi hình
        showlegend=False
    )
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output_plot')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.write_html(output_path)
    print(f"Success! Output saved to: {os.path.abspath(output_path)}")

script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Vẽ cho luồng từ kho gửi (origin) đến kho 1A đầu tiên
process_and_plot(
    file_path=os.path.join(script_dir, '../output_all_traces/origin_to_1A_filter.csv'),
    col_start='kho_o',
    col_end='kho_o1a',
    title_prefix='Luồng Kho Gốc -> Kho 1A',
    output_filename='scatter_origin_to_1A.html'
)

# 2. Vẽ cho luồng từ kho 1A cuối cùng đến kho nhận (destination)
process_and_plot(
    file_path=os.path.join(script_dir, '../output_all_traces/1A_to_destination_filter.csv'),
    col_start='kho_d1a',
    col_end='kho_d',
    title_prefix='Luồng Kho 1A -> Kho Đích',
    output_filename='scatter_1A_to_destination.html'
)
