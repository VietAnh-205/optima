# -*- coding: utf-8 -*-
"""
predict_arrival.py
==================
Inference Module độc lập dùng để dự báo thời gian đến kho 1A cho 1 bill bất kỳ hoặc 1 file CSV.

Đặc điểm:
  - Hoàn toàn độc lập, không cần đọc file dữ liệu lớn (bill.csv, bill_schedule.csv).
  - Chỉ cần file `model_bundle.pkl` đã được huấn luyện.
  - Có feature gì dùng lấy, thiếu trường nào tự động fill NaN/null.

Cách dùng CLI:
  # 1. Liệt kê các model đã có
  python predict_arrival.py --list

  # 2. Dự báo trực tiếp 1 bill qua command line
  python predict_arrival.py \\
      --model ../machine/saved_forecasting_models/kho_song_than/model_bundle.pkl \\
      --bill warehouse_name="Kho Bưu Cục Hoàn Kiếm" io_status=OUT io_time="2026-08-24 18:30:00" actual_weight=3.5

  # 3. Dự báo hàng loạt từ file CSV
  python predict_arrival.py \\
      --model ../machine/saved_forecasting_models/kho_song_than/model_bundle.pkl \\
      --input lo_hang_moi.csv \\
      --output ket_qua_du_bao.csv

Cách dùng trong Python:
  from predict_arrival import predict_one, predict_batch

  # Dự báo 1 đơn
  res = predict_one(
      model_path="../machine/saved_forecasting_models/kho_song_than/model_bundle.pkl",
      bill_info={
          "warehouse_name": "Kho Văn Giang",
          "io_status": "OUT",
          "io_time": "2026-08-24 08:00:00",
          "actual_weight": 12.0,
      }
  )
  print(res)
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

_machine_dir = os.path.abspath(os.path.join(_flow_dir, "..", "machine"))
if _machine_dir not in sys.path:
    sys.path.insert(0, _machine_dir)

from forecasting import (
    ForecastingPredictor,
    list_available_models,
    _slug,
)

DEFAULT_MODELS_DIR = os.path.join(_flow_dir, "saved_forecasting_models")


def _resolve_model_path(model_arg: str, models_dir: str = DEFAULT_MODELS_DIR) -> str:
    """Tự động tìm file model_bundle.pkl theo nhiều cách chỉ định khác nhau (tên kho, slug, đường dẫn tương đối/tuyệt đối)."""
    if not model_arg:
        return ""
    
    if os.path.exists(model_arg):
        if os.path.isdir(model_arg):
            cand = os.path.join(model_arg, "model_bundle.pkl")
            if os.path.exists(cand):
                return cand
        return model_arg
        
    candidates = [
        os.path.join(models_dir, _slug(model_arg), "model_bundle.pkl"),
        os.path.join(models_dir, model_arg, "model_bundle.pkl"),
        os.path.join(models_dir, model_arg),
        os.path.join(_flow_dir, "saved_forecasting_models", _slug(model_arg), "model_bundle.pkl"),
        os.path.join(_flow_dir, model_arg),
        os.path.join(_machine_dir, "saved_forecasting_models", _slug(model_arg), "model_bundle.pkl"),
        os.path.join(_machine_dir, model_arg),
        os.path.join(_base_dir, model_arg),
        os.path.join(_base_dir, "analysis_data", "flow", "saved_forecasting_models", _slug(model_arg), "model_bundle.pkl"),
        os.path.join(_base_dir, "analysis_data", "machine", "saved_forecasting_models", _slug(model_arg), "model_bundle.pkl"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            if os.path.isdir(cand):
                sub = os.path.join(cand, "model_bundle.pkl")
                if os.path.exists(sub):
                    return sub
            return cand
            
    available = list_available_models(models_dir)
    avail_names = [m["warehouse_name"] for m in available]
    raise FileNotFoundError(
        f"Không tìm thấy file model bundle cho: '{model_arg}'.\n"
        f"Các model hiện có trong hệ thống: {avail_names}"
    )


def _parse_bill_args(tokens: list) -> dict:
    """Chuyển danh sách 'key=value' thành dict."""
    bill = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Tham số không hợp lệ: '{token}'. Cú pháp: key=value")
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        bill[k] = v
    return bill


def predict_one(model_path: str, bill_info: dict) -> dict:
    """Dự báo cho 1 đơn hàng duy nhất."""
    resolved_path = _resolve_model_path(model_path)
    predictor = ForecastingPredictor(resolved_path)
    return predictor.predict_one(bill_info)


def predict_batch(model_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """Dự báo cho 1 DataFrame gồm nhiều đơn hàng."""
    resolved_path = _resolve_model_path(model_path)
    predictor = ForecastingPredictor(resolved_path)
    return predictor.predict(df)


def main():
    parser = argparse.ArgumentParser(
        description="Dự báo thời gian đến kho 1A từ model đã đóng gói (Inference-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liệt kê tất cả các model đã train sẵn",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Đường dẫn file model_bundle.pkl hoặc tên kho 1A (VD: 'Kho Sóng Thần' / 'kho_song_than')",
    )
    parser.add_argument(
        "--bill",
        nargs="+",
        metavar="key=value",
        help='Thông tin 1 bill dạng key=value. VD: warehouse_name="Kho HN" io_status=OUT io_time="2026-08-24 18:00:00"',
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Đường dẫn file CSV chứa các đơn hàng cần dự báo",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Đường dẫn file CSV kết quả (mặc định: predictions.csv)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Thư mục chứa model bundles (mặc định: {DEFAULT_MODELS_DIR})",
    )

    args = parser.parse_args()

    if args.list:
        models = list_available_models(args.models_dir)
        if not models:
            print(f"Chưa có model nào trong: {args.models_dir}")
        else:
            print(f"Tìm thấy {len(models)} model đã sẵn sàng:")
            for i, m in enumerate(models, 1):
                metrics = m.get("train_metrics", {})
                mae_str = f" | Test MAE: {metrics['test_mae']:.2f}h" if "test_mae" in metrics else ""
                r2_str = f" | R²: {metrics['test_r2']:.4f}" if "test_r2" in metrics else ""
                print(f"  {i}. {m['warehouse_name']} ({m['path']}){mae_str}{r2_str}")
        return

    if not args.model:
        parser.print_help()
        print("\n[LỖI] Vui lòng chỉ định --model hoặc dùng --list")
        return

    # Xử lý đường dẫn model
    model_path = _resolve_model_path(args.model, args.models_dir)
    predictor = ForecastingPredictor(model_path)
    print(f"Đã nạp model: {predictor.warehouse_name} ({model_path})")

    # 1. Predict 1 bill
    if args.bill:
        bill_info = _parse_bill_args(args.bill)
        print(f"\nThông tin đầu vào: {bill_info}")
        res = predictor.predict_one(bill_info)
        print("\n--- KẾT QUẢ DỰ BÁO ---")
        for k, v in res.items():
            print(f"  {k:30s}: {v}")
        return

    # 2. Predict từ CSV
    if args.input:
        print(f"\nĐang đọc dữ liệu từ: {args.input}")
        df = pd.read_csv(args.input)
        print(f"  Số lượng đơn: {len(df)}")
        res_df = predictor.predict(df)
        res_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\nĐã xuất kết quả ra file: {args.output}")
        print(res_df.head(10).to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()