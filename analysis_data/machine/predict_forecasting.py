# -*- coding: utf-8 -*-
"""
predict_forecasting.py
=======================
CLI + importable function: predict cho data mới dùng model đã train.

Cách dùng CLI:
    # Liệt kê models đã có
    python predict_forecasting.py --list

    # Predict 1 bill trực tiếp (key=value)
    python predict_forecasting.py \\
        --model saved_forecasting_models/kho_song_than/model_bundle.pkl \\
        --bill warehouse_name="Kho Van Giang" io_status=OUT io_time="2026-08-24 08:00:00" actual_weight=5.0

    # Predict từ file CSV
    python predict_forecasting.py \\
        --model saved_forecasting_models/kho_song_than/model_bundle.pkl \\
        --input data.csv \\
        --output predictions.csv

Cách dùng trong Python:
    from forecasting import ForecastingPredictor

    predictor = ForecastingPredictor("model_bundle.pkl")

    # Predict batch
    results = predictor.predict(df)

    # Predict 1 đơn
    result = predictor.predict_one({
        "warehouse_name": "Kho HN", "io_status": "OUT", "io_time": "2026-01-15 08:00:00",
        "bill_code": "B_123", "actual_weight": 5.0,
    })
"""

import argparse
import os
import sys

import pandas as pd

# Đảm bảo import được package forecasting/ cùng thư mục
_base_dir = os.path.dirname(os.path.abspath(__file__))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

from forecasting import ForecastingPredictor, list_available_models  # noqa: E402


# Thư mục mặc định chứa model bundles
DEFAULT_MODELS_DIR = os.path.join(_base_dir, "saved_forecasting_models")


def _parse_bill_args(tokens: list[str]) -> dict:
    """
    Chuyển danh sách 'key=value' thành dict.
    VD: ["warehouse_name=Kho HN", "actual_weight=5.0"] -> {"warehouse_name": "Kho HN", "actual_weight": 5.0}
    """
    bill = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Tham số không hợp lệ: '{token}'. Cú pháp: key=value")
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Tự động parse số
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass  # Giữ nguyên string
        bill[k] = v
    return bill


def main():
    parser = argparse.ArgumentParser(
        description="Du doan thoi gian den kho 1A tu model da train.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  python predict_forecasting.py --list

  # Predict 1 bill truc tiep:
  python predict_forecasting.py --model saved_forecasting_models/kho_song_than/model_bundle.pkl --bill warehouse_name="Kho Van Giang" io_status=OUT io_time="2026-08-24 08:00:00" actual_weight=31.0

  # Predict tu file CSV:
  python predict_forecasting.py --model saved_forecasting_models/kho_song_than/model_bundle.pkl --input data.csv
        """,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liet ke tat ca model da train",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Duong dan toi file model_bundle.pkl",
    )
    parser.add_argument(
        "--bill",
        nargs="+",
        metavar="key=value",
        help='Thong tin 1 bill, dang key=value. VD: warehouse_name="Kho HN" io_status=OUT actual_weight=5.0',
    )
    parser.add_argument(
        "--input",
        type=str,
        help="File CSV chua thong tin don hang (1 don = 1 dong)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="File CSV ket qua (mac dinh: predictions.csv). Chi dung khi --input",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Thu muc chua model bundles (mac dinh: {DEFAULT_MODELS_DIR})",
    )

    args = parser.parse_args()

    # ── --list ──
    if args.list:
        models = list_available_models(args.models_dir)
        if not models:
            print(f"Chua co model nao trong: {args.models_dir}")
        else:
            print(f"Co {len(models)} model da train:\n")
            for i, m in enumerate(models, 1):
                metrics = m.get("train_metrics", {})
                mae_str = f"  MAE={metrics['test_mae']:.2f}h" if "test_mae" in metrics else ""
                r2_str = f"  R2={metrics['test_r2']:.4f}" if "test_r2" in metrics else ""
                print(f"  {i}. {m['warehouse_name']}")
                print(f"     Path: {m['path']}")
                print(f"     Trained: {m['trained_at']}{mae_str}{r2_str}")
                print()
        return

    # ── Cần --model ──
    if not args.model:
        parser.print_help()
        print("\nLoi: Can --model hoac --list")
        return

    # Load model
    print(f"Loading model: {args.model}")
    predictor = ForecastingPredictor(args.model)
    print(f"  Kho: {predictor.warehouse_name}")
    metrics = predictor.train_metrics
    if metrics:
        print(f"  MAE: {metrics.get('test_mae', 'N/A'):.2f}h  |  R2: {metrics.get('test_r2', 'N/A'):.4f}")

    # ── --bill: predict 1 bill trực tiếp ──
    if args.bill:
        bill_info = _parse_bill_args(args.bill)
        print(f"\nInput: {bill_info}")
        result = predictor.predict_one(bill_info)
        print("\n--- KET QUA ---")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    # ── --input: predict từ file CSV ──
    if not args.input:
        parser.print_help()
        print("\nLoi: Can --bill (1 don) hoac --input (file CSV)")
        return

    print(f"\nLoading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} don hang")

    print("\nPredicting...")
    results = predictor.predict(df)

    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nKet qua ({len(results)} don):")
    print(results.to_string(index=False))
    print(f"\nDa luu: {args.output}")


if __name__ == "__main__":
    main()
