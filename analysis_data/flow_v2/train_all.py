# -*- coding: utf-8 -*-
"""
train_all.py — Huấn luyện toàn bộ các kho 1A trong warehouse_1A.csv
"""

import os
import sys
import time
import gc
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from preprocessor import prepare_training_dataset
from arrival_forecaster import ArrivalForecaster, slugify_name


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình cho toàn bộ các kho 1A.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../",
        help="Thư mục chứa 3 file CSV: bill_schedule.csv, bill.csv, warehouse_1A.csv (mặc định: ../../)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_models",
        help="Thư mục lưu các file .pkl model (mặc định: saved_models)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bỏ qua các kho đã có model .pkl trong thư mục lưu"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Số lượng mẫu tối thiểu để train mô hình (mặc định: 100)"
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    summary_csv = os.path.join(args.save_dir, "training_summary.csv")

    print("=" * 70)
    print("      HUẤN LUYỆN TOÀN BỘ CÁC KHO 1A (FLOW V2)")
    print("=" * 70)

    # 1. Đọc dữ liệu 1 LẦN DUY NHẤT vào RAM
    t_start = time.time()
    sched_path = os.path.join(args.data_dir, "bill_schedule.csv")
    bill_path = os.path.join(args.data_dir, "bill.csv")
    wh1a_path = os.path.join(args.data_dir, "warehouse_1A.csv")

    if not os.path.exists(wh1a_path):
        if os.path.exists("warehouse_1A.csv"):
            wh1a_path, bill_path, sched_path = "warehouse_1A.csv", "bill.csv", "bill_schedule.csv"
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {args.data_dir}")

    print("\n[BƯỚC 1/3] Đang nạp danh sách kho 1A và bill.csv...")
    wh1a = pd.read_csv(wh1a_path)
    bill = pd.read_csv(bill_path)
    warehouse_list = sorted(wh1a["name"].dropna().str.strip().unique().tolist())
    print(f"  -> Tìm thấy {len(warehouse_list)} kho 1A trong danh sách.")

    print("\n[BƯỚC 2/3] Đang nạp bill_schedule.csv theo chunks (file lớn ~16M dòng)...")
    t_load = time.time()
    chunks = []
    for chunk in pd.read_csv(sched_path, chunksize=500_000):
        chunk["io_time"] = pd.to_datetime(chunk["io_time"])
        chunks.append(chunk)
    sche = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    print(f"  -> Nạp thành công {len(sche):,} dòng bill_schedule ({time.time() - t_load:.1f}s)")

    # 2. Vòng lặp huấn luyện từng kho
    print(f"\n[BƯỚC 3/3] Bắt đầu huấn luyện {len(warehouse_list)} kho...")
    print("-" * 70)

    results = []

    for idx, wh_name in enumerate(warehouse_list, 1):
        slug = slugify_name(wh_name)
        expected_pkl = os.path.join(args.save_dir, f"{slug}.pkl")

        print(f"\n[{idx}/{len(warehouse_list)}] Kho: {wh_name}")

        # Kiểm tra nếu đã có model và bật cờ skip
        if args.skip_existing and os.path.exists(expected_pkl):
            print(f"  -> [BỎ QUA] Đã tồn tại model: {expected_pkl}")
            results.append({
                "warehouse": wh_name,
                "status": "SKIPPED",
                "samples": "-",
                "test_mae": "-",
                "test_medae": "-",
                "test_r2": "-",
                "duration_s": "-"
            })
            continue

        t_wh = time.time()
        try:
            clean_df = prepare_training_dataset(
                bill_schedule_df=sche,
                bill_df=bill,
                warehouse_1a_df=wh1a,
                target_warehouse=wh_name
            )

            n_samples = len(clean_df)
            if n_samples < args.min_samples:
                print(f"  -> [BỎ QUA] Không đủ dữ liệu ({n_samples} mẫu < {args.min_samples})")
                results.append({
                    "warehouse": wh_name,
                    "status": "INSUFFICIENT_DATA",
                    "samples": n_samples,
                    "test_mae": "-",
                    "test_medae": "-",
                    "test_r2": "-",
                    "duration_s": f"{time.time() - t_wh:.1f}"
                })
                del clean_df
                gc.collect()
                continue

            print(f"  -> Trích xuất được {n_samples:,} mẫu huấn luyện ({time.time() - t_wh:.1f}s)")

            model = ArrivalForecaster(target_warehouse=wh_name)
            model.fit(clean_df, target_warehouse=wh_name, save_dir=args.save_dir)

            m = model.bundle.get("train_metrics", {})
            duration = time.time() - t_wh

            results.append({
                "warehouse": wh_name,
                "status": "SUCCESS",
                "samples": n_samples,
                "test_mae": round(m.get("test_mae", 0.0), 2),
                "test_medae": round(m.get("test_medae", 0.0), 2),
                "test_r2": round(m.get("test_r2", 0.0), 4),
                "duration_s": round(duration, 1)
            })

            del clean_df, model
            gc.collect()

        except Exception as e:
            print(f"  -> [LỖI]: {e}")
            results.append({
                "warehouse": wh_name,
                "status": f"ERROR: {str(e)[:40]}",
                "samples": "-",
                "test_mae": "-",
                "test_medae": "-",
                "test_r2": "-",
                "duration_s": f"{time.time() - t_wh:.1f}"
            })
            gc.collect()

    # 3. Tổng kết kết quả
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("                    BẢNG TỔNG KẾT HUẤN LUYỆN")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("-" * 70)
    print(f"Tổng thời gian toàn bộ tiến trình: {time.time() - t_start:.1f}s (~{(time.time() - t_start)/60:.1f} phút)")
    print(f"Báo cáo chi tiết đã lưu tại: {summary_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
