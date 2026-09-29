# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_data.end_to_end.multi_label import EndToEndPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện End-to-End Multi-Label Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=PROJECT_ROOT,
        help=f"Thư mục chứa 3 file CSV: bill.csv, bill_schedule.csv, warehouse_1A.csv (mặc định: {PROJECT_ROOT})"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "saved_e2e_models"),
        help="Thư mục lưu file bundle mô hình (mặc định: saved_e2e_models)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="e2e_pipeline.pkl",
        help="Tên file bundle mô hình (mặc định: e2e_pipeline.pkl)"
    )
    parser.add_argument(
        "--snapshot-hour",
        type=int,
        default=20,
        help="Giờ lấy snapshot Point-In-Time (mặc định: 20)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.75,
        help="Tỷ lệ tập huấn luyện (mặc định: 0.75)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Tỷ lệ tập kiểm định (mặc định: 0.15)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("KHỞI CHẠY TRAINING SCRIPT: END-TO-END MULTI-LABEL")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Save dir     : {args.save_dir}")
    print(f"  Model name   : {args.model_name}")
    print(f"  Snapshot hour: {args.snapshot_hour}:00")
    print("=" * 70)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.model_name)

    pipeline = EndToEndPipeline()
    metrics = pipeline.train(
        data_dir=args.data_dir,
        save_path=save_path,
        train_split=args.train_split,
        val_split=args.val_split,
        snapshot_hour=args.snapshot_hour,
    )

    report_path = os.path.join(args.save_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[train_e2e] Đã lưu báo cáo đánh giá tại: {report_path}")


if __name__ == "__main__":
    main()
