import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import time
from preprocessor import prepare_training_dataset
from arrival_forecaster import ArrivalForecaster

import os
data_dir = "." if os.path.exists("bill_schedule.csv") else "../../"
sched_file = os.path.join(data_dir, "bill_schedule.csv")
bill_file = os.path.join(data_dir, "bill.csv")
wh1a_file = os.path.join(data_dir, "warehouse_1A.csv")

# Đọc bill_schedule theo chunks (file 16 triệu dòng, rất nặng)
print("[1] Đọc bill_schedule.csv theo chunks...")
t0 = time.time()
chunks = []
for chunk in pd.read_csv(sched_file, chunksize=500_000):
    chunk["io_time"] = pd.to_datetime(chunk["io_time"])
    chunks.append(chunk)
sche = pd.concat(chunks, ignore_index=True)
del chunks
print(f"  -> Xong! {sche.shape[0]:,} dòng ({time.time()-t0:.1f}s)")

print("[2] Đọc bill.csv & warehouse_1A.csv...")
bill = pd.read_csv(bill_file)
wh1a = pd.read_csv(wh1a_file)

print("[3] Chuẩn bị dữ liệu huấn luyện cho Kho Sóng Thần...")
t1 = time.time()
clean_train_df = prepare_training_dataset(sche, bill, wh1a, target_warehouse="Kho Bình Định")
print(f"  -> Xong! {clean_train_df.shape[0]:,} dòng, {clean_train_df.shape[1]} cột ({time.time()-t1:.1f}s)")

print("[4] Huấn luyện mô hình...")
t2 = time.time()
model = ArrivalForecaster()
model.fit(clean_train_df, target_warehouse="Kho Bình Định")
print(f"  -> Xong! ({time.time()-t2:.1f}s)")

print("\n[6] Thử predict 1 đơn...")
res = model.predict_one({
    "warehouse_name": "Kho Bến Cát",
    "io_status": "OUT",
    "io_time": "2026-08-19 08:30:00",
    "actual_weight": 2.5
})
print(f"  -> Dự báo: {res['predicted_arrival_offset_hours']:.2f} giờ")
print(f"  -> Thời điểm đến: {res['predicted_arrival_time']}")
print(f"\nTổng thời gian: {time.time()-t0:.1f}s")