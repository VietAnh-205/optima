import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_trips(file_path, col_end, output_prefix):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    df['time_1a_out'] = pd.to_datetime(df['time_1a_out'])

    df['out_15min'] = df['time_1a_out'].dt.floor('15min')
    
    trip_df = df.groupby([col_end, 'out_15min']).agg(                   
        total_bills=('bill_code', 'nunique'),
        total_weight=('actual_weight', 'sum'),
        avg_wait_time=('time_in_1a', 'mean')
    ).reset_index()
    
    trip_df = trip_df[trip_df['total_bills'] >= 10].copy()
    
    trip_df['hour_out'] = trip_df['out_15min'].dt.hour + trip_df['out_15min'].dt.minute / 60.0
    
    print(f"Extracted {len(trip_df)} trips.")
    
    # Lưu bảng Chuyến xe tổng ra file CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output_all_traces', f'trips_{output_prefix}.csv')
    trip_df.to_csv(csv_path, index=False)
    print(f"CSV data saved: {os.path.abspath(csv_path)}")

    # Lấy danh sách các kho có đủ dữ liệu (>= 10 chuyến)
    kho_counts = trip_df[col_end].value_counts()
    valid_khos = kho_counts[kho_counts >= 10].index
    
    # Tính tổng số bill của mỗi kho để sắp xếp (từ cao xuống thấp)
    kho_bill_sums = trip_df[trip_df[col_end].isin(valid_khos)].groupby(col_end)['total_bills'].sum()
    kho_list = kho_bill_sums.sort_values(ascending=False).index.tolist()
    
    print(f"Starting to plot for {len(kho_list)} warehouses...")
    
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .plot-container {{ margin-bottom: 80px; padding-bottom: 20px; border-bottom: 3px dashed #ccc; }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Phân tích Thống kê Chuyến Xe ({output_prefix})</h1>
        <hr>
    """
    
    for i, kho in enumerate(kho_list):
        sub_df = trip_df[trip_df[col_end] == kho].copy()
        
        fig = make_subplots(
            rows=5, cols=1, 
            subplot_titles=[
                "1. Phân phối Giờ Xe Chạy",
                "2. Phân phối Số lượng Bill/Chuyến",
                "3. Phân phối Khối lượng/Chuyến (kg)",
                "4. Tương quan Giờ Xe Chạy vs Số lượng Bill",
                "5. Tương quan Giờ Xe Chạy vs Khối lượng"
            ],
            vertical_spacing=0.06
        )
        
        # Hình 1: Histogram Giờ OUT
        fig.add_trace(
            go.Histogram(
                x=sub_df['hour_out'], 
                nbinsx=48, # Mỗi cột 30 phút
                name='Số chuyến xe',
                marker_color='#3874FF'
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text="Khung giờ trong ngày (0-24h)", tickmode='linear', dtick=1, row=1, col=1)
        fig.update_yaxes(title_text="Tần suất (Số chuyến)", row=1, col=1)
        
        # Thêm đường trung bình cho Hình 1
        mean_hour = sub_df['hour_out'].mean()
        fig.add_vline(x=mean_hour, line_dash="dash", line_color="#FF3366", line_width=2, 
                      annotation_text=f"Trung bình: {mean_hour:.1f}h", annotation_position="top right", 
                      annotation_font=dict(color="#FF3366", size=12), row=1, col=1)
        
        # Tính toán mốc 98% (percentile 98) để ẩn đi 2% các chuyến đột biến cực đoan
        p98_bills = sub_df['total_bills'].quantile(0.98)
        p98_weight = sub_df['total_weight'].quantile(0.98)
        
        max_x_bills = p98_bills if pd.notnull(p98_bills) else sub_df['total_bills'].max()
        max_x_weight = p98_weight if pd.notnull(p98_weight) else sub_df['total_weight'].max()

        # Hình 2: Histogram Phân phối Số lượng Bill (thay cho Boxplot)
        fig.add_trace(
            go.Histogram(
                x=sub_df['total_bills'],
                name='Tần suất (Số Bill)',
                nbinsx=200, # Chia siêu nhỏ
                marker_color='#3874FF'
            ),
            row=2, col=1
        )
        # Giới hạn trục X từ 0 đến mốc 98% để tránh bị giãn bởi outlier
        fig.update_xaxes(title_text="Số lượng Bill trên 1 chuyến xe", range=[0, max_x_bills], row=2, col=1)

        # Thêm đường trung bình cho Hình 2
        mean_bills = sub_df['total_bills'].mean()
        fig.add_vline(x=mean_bills, line_dash="dash", line_color="#FF3366", line_width=2, 
                      annotation_text=f"Trung bình: {mean_bills:.0f} bill", annotation_position="top right", 
                      annotation_font=dict(color="#FF3366", size=12), row=2, col=1)

        # Hình 3: Histogram Phân phối Khối lượng (thay cho Boxplot)
        fig.add_trace(
            go.Histogram(
                x=sub_df['total_weight'],
                name='Tần suất (Khối lượng)',
                nbinsx=200, # Chia siêu nhỏ
                marker_color='#3874FF'
            ),
            row=3, col=1
        )
        # Giới hạn trục X từ 0 đến mốc 98%
        fig.update_xaxes(title_text="Khối lượng trên 1 chuyến (kg)", range=[0, max_x_weight], row=3, col=1)
        
        # Thêm đường trung bình cho Hình 3
        mean_weight = sub_df['total_weight'].mean()
        fig.add_vline(x=mean_weight, line_dash="dash", line_color="#FF3366", line_width=2, 
                      annotation_text=f"Trung bình: {mean_weight:.0f} kg", annotation_position="top right", 
                      annotation_font=dict(color="#FF3366", size=12), row=3, col=1)
        
        # Tính toán Trendline (Đường trung vị theo từng giờ)
        sub_df['hour_int'] = sub_df['hour_out'].astype(int)
        hourly_trend = sub_df.groupby('hour_int').agg({
            'total_bills': 'median',
            'total_weight': 'median'
        }).reset_index()
        hourly_trend['plot_hour'] = hourly_trend['hour_int'] + 0.5 # Để điểm neo vào giữa mốc giờ

        # Hình 4: Scatter Giờ Chạy (X) và Số lượng Bill (Y)
        fig.add_trace(
            go.Scatter(
                x=sub_df['hour_out'],
                y=sub_df['total_bills'],
                mode='markers',
                marker=dict(opacity=0.3, color='#3874FF'),
                name='Chuyến xe (Bill)'
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=hourly_trend['plot_hour'],
                y=hourly_trend['total_bills'],
                mode='lines+markers',
                line=dict(color='#FF3366', width=3),
                marker=dict(size=6, color='#FF3366'),
                name='Trung vị Bill/Giờ'
            ),
            row=4, col=1
        )
        fig.update_xaxes(title_text="Khung giờ (0-24h)", tickmode='linear', dtick=1, row=4, col=1)
        fig.update_yaxes(title_text="Số lượng Bill", range=[0, max_x_bills], row=4, col=1)

        # Hình 5: Scatter Giờ Chạy (X) và Khối lượng (Y)
        fig.add_trace(
            go.Scatter(
                x=sub_df['hour_out'],
                y=sub_df['total_weight'],
                mode='markers',
                marker=dict(opacity=0.3, color='#3874FF'),
                name='Chuyến xe (Khối lượng)'
            ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=hourly_trend['plot_hour'],
                y=hourly_trend['total_weight'],
                mode='lines+markers',
                line=dict(color='#FF3366', width=3),
                marker=dict(size=6, color='#FF3366'),
                name='Trung vị Khối lượng/Giờ'
            ),
            row=5, col=1
        )
        fig.update_xaxes(title_text="Khung giờ (0-24h)", tickmode='linear', dtick=1, row=5, col=1)
        fig.update_yaxes(title_text="Khối lượng (kg)", range=[0, max_x_weight], row=5, col=1)
        
        fig.update_layout(
            title_text=f"Kho 1A: {kho}",
            height=1600,
            showlegend=False
        )
        
        html_content += f"<div class='plot-container'>\n"
        html_content += fig.to_html(full_html=False, include_plotlyjs=False)
        html_content += "</div>\n"
        
    html_content += "</body></html>"
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output_plot', f'trips_dashboard_{output_prefix}.html')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Plots saved to: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, '..', 'output_plot'), exist_ok=True)
    
    # 1. Luồng Gửi -> 1A (OUT khỏi 1A)
    analyze_trips(
        file_path=os.path.join(base_dir, '..', 'output_all_traces', 'origin_to_1A_filter.csv'),
        col_end='kho_o1a',
        output_prefix='origin_to_1A'
    )
    
    # 2. Luồng 1A -> Nhận (OUT khỏi 1A)
    analyze_trips(
        file_path=os.path.join(base_dir, '..', 'output_all_traces', '1A_to_destination_filter.csv'),
        col_end='kho_d1a',
        output_prefix='1A_to_destination'
    )
