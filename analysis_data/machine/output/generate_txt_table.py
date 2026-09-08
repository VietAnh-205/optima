import pandas as pd
import numpy as np

# Load comparison data
df = pd.read_csv(r"d:\VietAnd\optima\VietAnh\analysis_data\machine\output\comparison_v3_vs_flow_v2.csv")

lines = []
lines.append("=" * 105)
lines.append("           BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH DỰ BÁO: V3 (LEGACY) VS FLOW V2 (MỚI)")
lines.append("=" * 105)
lines.append(f"{'Tên Kho':<25} | {'Samples (V3/V2)':<16} | {'MAE giờ (V3/V2)':<17} | {'MedAE (V3/V2)':<15} | {'R² (V3/V2)':<15} | {'Tốt hơn'}")
lines.append("-" * 105)

v2_wins = 0
v3_wins = 0
tie = 0

for _, row in df.iterrows():
    wh = row['warehouse'].strip()
    v3_status = row['v3_status']
    v2_status = str(row['v2_status'])

    v3_n = f"{int(row['v3_samples']):,}" if pd.notna(row['v3_samples']) else "-"
    v2_n = f"{int(row['v2_samples']):,}" if pd.notna(row['v2_samples']) and row['v2_samples'] != '-' else "-"
    samples_str = f"{v3_n} / {v2_n}"

    if v3_status == 'SUCCESS' and 'SUCCESS' in v2_status:
        v3_mae = f"{row['v3_mae']:.2f}"
        v2_mae = f"{row['v2_mae']:.2f}"
        mae_str = f"{v3_mae} / {v2_mae}"

        v3_med = f"{row['v3_medae']:.2f}"
        v2_med = f"{row['v2_medae']:.2f}"
        med_str = f"{v3_med} / {v2_med}"

        v3_r2 = f"{row['v3_r2']:.2f}" if pd.notna(row['v3_r2']) else "N/A"
        v2_r2 = f"{row['v2_r2']:.2f}" if pd.notna(row['v2_r2']) else "N/A"
        r2_str = f"{v3_r2} / {v2_r2}"

        # Determine winner based on MAE
        diff = row['v3_mae'] - row['v2_mae']
        if abs(diff) < 0.05:
            winner = "Ngang nhau"
            tie += 1
        elif diff > 0:
            winner = f"V2 (+{diff:.2f}h)"
            v2_wins += 1
        else:
            winner = f"V3 ({diff:.2f}h)"
            v3_wins += 1
    elif v3_status != 'SUCCESS' and 'SUCCESS' in v2_status:
        mae_str = f"- / {row['v2_mae']:.2f}"
        med_str = f"- / {row['v2_medae']:.2f}"
        r2_str = f"- / {row['v2_r2']:.2f}"
        winner = "Chỉ V2 chạy được"
    elif v3_status == 'SUCCESS' and 'SUCCESS' not in v2_status:
        mae_str = f"{row['v3_mae']:.2f} / -"
        med_str = f"{row['v3_medae']:.2f} / -"
        r2_str = f"{row['v3_r2']:.2f} / -"
        winner = "Chỉ V3 chạy được"
    else:
        mae_str = "- / -"
        med_str = "- / -"
        r2_str = "- / -"
        winner = "Cả 2 thiếu data"

    lines.append(f"{wh:<25} | {samples_str:<16} | {mae_str:<17} | {med_str:<15} | {r2_str:<15} | {winner}")

lines.append("=" * 105)
lines.append("TỔNG KẾT:")
lines.append(f"- Tổng số kho so sánh: {len(df)} kho")
lines.append(f"- Số kho cả 2 đều chạy thành công: {v2_wins + v3_wins + tie} kho")
lines.append(f"  + Flow V2 có MAE tốt hơn: {v2_wins} kho")
lines.append(f"  + V3 Legacy có MAE tốt hơn: {v3_wins} kho (chủ yếu kho nhỏ/ít mẫu)")
lines.append(f"  + Tương đương: {tie} kho (Kho Huế)")

# Average metrics on common successful warehouses
common = df[(df['v3_status'] == 'SUCCESS') & (df['v2_status'] == 'SUCCESS')].copy()
v3_avg_mae = common['v3_mae'].mean()
v2_avg_mae = common['v2_mae'].mean()
v3_avg_med = common['v3_medae'].mean()
v2_avg_med = common['v2_medae'].mean()

lines.append(f"- MAE trung bình (trên {len(common)} kho chung):")
lines.append(f"  + V3 Legacy: {v3_avg_mae:.2f} giờ")
lines.append(f"  + Flow V2:   {v2_avg_mae:.2f} giờ  (V2 giảm sai số trung bình {v3_avg_mae - v2_avg_mae:.2f} giờ)")
lines.append(f"- MedAE trung bình:")
lines.append(f"  + V3 Legacy: {v3_avg_med:.2f} giờ")
lines.append(f"  + Flow V2:   {v2_avg_med:.2f} giờ")
lines.append("")
lines.append("ĐÁNH GIÁ CHÍNH:")
lines.append("1. Flow V2 vượt trội ở các kho trọng điểm (Sóng Thần, Văn Giang, Đà Nẵng, Diên Khánh, Hải Phòng).")
lines.append("2. Flow V2 sinh nhiều mẫu huấn luyện hơn nhờ cơ chế multi-snapshot dọc hành trình.")
lines.append("3. V3 Legacy có MAE thấp hơn ở một số kho rất ít đơn (Buôn Ma Thuột, Thanh Hóa, Nghệ An, Bình Thuận)")
lines.append("   do cách tính target của V3 là từ 0h đêm thay vì thời điểm snapshot thực tế.")
lines.append("=" * 105)

txt_content = "\n".join(lines)
output_path = r"d:\VietAnd\optima\VietAnh\analysis_data\machine\output\so_sanh_v2_vs_v3.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(txt_content)

print("Saved successfully to", output_path)
print(txt_content)
