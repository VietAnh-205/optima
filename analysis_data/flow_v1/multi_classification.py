# -*- coding: utf-8 -*-
"""
multi_classification.py
========================
Pipeline dự báo thời gian đến kho 1A (Refactored):
  1. Đọc dữ liệu (bill, bill_schedule, warehouse_1A).
  2. Lọc bills đi qua kho 1A đích (mọi kho đầu, chỉ dừng tại lần đầu đến kho 1A,
     loại bỏ các bill đã từng qua 1A khác trước đó).
  3. Xây dựng Trajectory & Stage features cho từng bước hành trình.
  4. Lọc outlier bằng IQR trên arrival_offset_hours.
  5. Feature engineering (~30 features) & PIT snapshot lúc 20:00 (mô phỏng thực tế).
  6. Luồng mô hình chuẩn hóa:
     - DBSCAN phân cụm target arrival_offset_hours.
     - Phân loại cụm bằng RandomForestClassifier.
     - Hồi quy số giờ bằng LGBMRegressor cho từng cụm.
     - Mô hình Single-modal / Fallback LGBMRegressor.
  7. Đóng gói và lưu trữ model bundle dạng slug:
     VD: "Kho Sóng Thần" -> saved_forecasting_models/kho_song_than/model_bundle.pkl

Cách dùng:
  # Train cho 1 kho 1A cụ thể:
  python multi_classification.py --warehouse "Kho Sóng Thần"

  # Train cho tất cả kho 1A:
  python multi_classification.py --all
"""

import os
import sys
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)


_flow_dir = os.path.dirname(os.path.abspath(__file__))
if _flow_dir not in sys.path:
    sys.path.insert(0, _flow_dir)

from forecasting import (
    train_pipeline,
    ForecastingPredictor,
    save_bundle,
    load_bundle,
    list_available_models,
    _slug,
)
from forecasting.config import (
    SNAPSHOT_HOURS,
    TRAIN_SPLIT,
    VAL_SPLIT,
    FEATURE_COLS,
    CATEGORICAL_FEATURES,
)

DEFAULT_DATA_DIR = _base_dir
DEFAULT_MODELS_DIR = os.path.join(_flow_dir, "saved_forecasting_models")


def get_all_1a_warehouses(data_dir: str) -> list:
    """Đọc warehouse_1A.csv và lấy danh sách các kho 1A."""
    path = os.path.join(data_dir, "warehouse_1A.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file warehouse_1A.csv tại: {path}")
    wh1a = pd.read_csv(path)
    return sorted(wh1a["name"].dropna().str.strip().unique().tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình dự báo thời gian đến kho 1A (DBSCAN + RF + LightGBM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--warehouse",
        type=str,
        default="Kho Sóng Thần",
        help='Tên kho 1A cần train (mặc định: "Kho Sóng Thần")',
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train cho TẤT CẢ các kho 1A có trong warehouse_1A.csv",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Thư mục chứa bill.csv, bill_schedule.csv (mặc định: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Thư mục lưu model bundle (mặc định: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--snapshot-hours",
        type=int,
        nargs="+",
        default=SNAPSHOT_HOURS,
        help=f"Danh sách giờ snapshot hàng ngày (mặc định: {SNAPSHOT_HOURS})",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=TRAIN_SPLIT,
        help=f"Tỷ lệ train (mặc định: {TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=VAL_SPLIT,
        help=f"Tỷ lệ val (mặc định: {VAL_SPLIT})",
    )

    args = parser.parse_args()

    if args.all:
        warehouses = get_all_1a_warehouses(args.data_dir)
        print(f"Tìm thấy {len(warehouses)} kho 1A:")
        for i, w in enumerate(warehouses, 1):
            print(f"  {i}. {w}")
    else:
        warehouses = [args.warehouse]

    for wh in warehouses:
        print(f"\n{'='*70}")
        print(f"  BẮT ĐẦU HUẤN LUYỆN: {wh}")
        print(f"{'='*70}")
        try:
            bundle = train_pipeline(
                selected_set={wh},
                data_dir=args.data_dir,
                train_split=args.train_split,
                val_split=args.val_split,
                snapshot_hours=args.snapshot_hours,
                output_dir=args.output_dir,
            )
            print(f"  --> Đã hoàn tất & lưu bundle cho {wh} tại: {_slug(wh)}/model_bundle.pkl")
        except Exception as e:
            print(f"  --> [LỖI] khi train {wh}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()