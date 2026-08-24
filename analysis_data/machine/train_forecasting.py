# -*- coding: utf-8 -*-
"""
train_forecasting.py
=====================
CLI để train model forecasting cho 1 hoặc tất cả kho 1A.

Cách dùng:
    # Train cho 1 kho cụ thể
    python train_forecasting.py --warehouse "Kho Sóng Thần"

    # Train cho TẤT CẢ kho 1A (đọc từ warehouse_1A.csv)
    python train_forecasting.py --all

    # Chỉ định thư mục dữ liệu
    python train_forecasting.py --warehouse "Kho Sóng Thần" --data-dir /path/to/data
"""

import argparse
import os
import sys
import time

import pandas as pd

# Đảm bảo import được package forecasting/ cùng thư mục
_base_dir = os.path.dirname(os.path.abspath(__file__))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

from forecasting.trainer import train_pipeline  # noqa: E402


# Thư mục mặc định chứa model bundles
DEFAULT_MODELS_DIR = os.path.join(_base_dir, "saved_forecasting_models")

# Thư mục mặc định chứa CSV data
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(_base_dir, "..", ".."))


def _get_all_1a_warehouses(data_dir: str) -> list[str]:
    """Đọc warehouse_1A.csv và trả về danh sách tên kho 1A."""
    path = os.path.join(data_dir, "warehouse_1A.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file warehouse_1A.csv tại: {path}"
        )
    wh1a = pd.read_csv(path)
    names = wh1a["name"].dropna().str.strip().unique().tolist()
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(
        description="Train model forecasting LightGBM cho kho 1A.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python train_forecasting.py --warehouse "Kho Sóng Thần"
  python train_forecasting.py --all
  python train_forecasting.py --all --data-dir /path/to/data
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--warehouse",
        type=str,
        help='Tên kho 1A cần train, VD: "Kho Sóng Thần"',
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train cho TẤT CẢ kho 1A (đọc từ warehouse_1A.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Thư mục chứa bill.csv, bill_schedule.csv, warehouse_1A.csv (mặc định: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Thư mục lưu model bundles (mặc định: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Tỉ lệ train/test split (mặc định: 0.9)",
    )

    args = parser.parse_args()

    # Xác định danh sách kho cần train
    if args.all:
        warehouses = _get_all_1a_warehouses(args.data_dir)
        print(f"Tìm thấy {len(warehouses)} kho 1A trong warehouse_1A.csv:")
        for i, w in enumerate(warehouses, 1):
            print(f"  {i:2d}. {w}")
        print()
    else:
        warehouses = [args.warehouse]

    # Train từng kho
    results = []
    for i, wh_name in enumerate(warehouses, 1):
        print(f"\n{'#' * 60}")
        print(f"# [{i}/{len(warehouses)}] {wh_name}")
        print(f"{'#' * 60}\n")

        t0 = time.time()
        try:
            bundle = train_pipeline(
                selected_set={wh_name},
                data_dir=args.data_dir,
                train_ratio=args.train_ratio,
                output_dir=args.output_dir,
            )
            elapsed = time.time() - t0
            results.append({
                "warehouse": wh_name,
                "status": "OK",
                "test_mae": bundle["train_metrics"]["test_mae"],
                "test_r2": bundle["train_metrics"]["test_r2"],
                "elapsed_min": elapsed / 60,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ERROR: {e}")
            results.append({
                "warehouse": wh_name,
                "status": f"ERROR: {e}",
                "test_mae": None,
                "test_r2": None,
                "elapsed_min": elapsed / 60,
            })

    # Tổng kết
    if len(warehouses) > 1:
        print("\n" + "=" * 60)
        print("TỔNG KẾT TRAINING")
        print("=" * 60)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        n_ok = sum(1 for r in results if r["status"] == "OK")
        print(f"\nThành công: {n_ok}/{len(warehouses)}")
        print(f"Model bundles: {args.output_dir}")


if __name__ == "__main__":
    main()
