# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PREDICT_ARRIVAL.PY — Inference-only: dự đoán THỜI GIAN TỚI NƠI            ║
╚══════════════════════════════════════════════════════════════════════════════╝

MỤC ĐÍCH
  File này KHÔNG lọc data, KHÔNG làm EDA, KHÔNG train lại model. Nó chỉ:
    1. Load model đã được multi_classification.py train & lưu sẵn
       (output_plot/saved_models/<pair_slug>/*.joblib)
    2. Nhận dữ liệu lô hàng mới (1 dict hoặc nhiều dòng trong DataFrame/CSV)
    3. Tái tạo feature CHÍNH XÁC như lúc train (dùng lại build_features_X2
       và pick_cluster_by_hour_from_info từ multi_classification.py)
    4. Áp dụng best_combo (chiến lược chọn cụm + feature option + model hồi quy
       đã được chọn là tốt nhất lúc train) để dự đoán y_offset (số giờ đi đường)
    5. Suy ra giờ dự kiến đến nơi = ngày khởi hành (00:00) + y_offset (giờ)

ĐIỀU KIỆN BẮT BUỘC
  - Đã chạy multi_classification.py (bản đã patch, có lưu thêm 'tv_cols',
    'top_origin', 'top_dest', 'w_median', 'is_dest_flow', 'y_min', 'y_max'
    vào meta.joblib) ÍT NHẤT 1 LẦN để sinh ra saved_models/.
  - Nếu saved_models/ đang là cache CŨ (trước khi patch), multi_classification.py
    (bản patch) sẽ tự phát hiện thiếu 'tv_cols' và tự train lại 1 lần cho từng
    pair để bổ sung đủ metadata — không cần bạn tự xoá cache tay.

DỮ LIỆU ĐẦU VÀO CẦN CÓ (mỗi dòng = 1 lô hàng cần dự đoán)
  departure_time        (BẮT BUỘC) thời điểm hàng khởi hành khỏi kho đầu chặng,
                         dạng datetime parse được, VD "2026-08-05 09:15:00".
                         Đây là mốc để tính departure_hour_numeric VÀ để suy ra
                         ngày khởi hành (giờ đến dự kiến = ngày này + y_offset).
  actual_weight          Trọng lượng bill (kg).
  VD_type, service       Loại VD / loại dịch vụ (nếu có).
  day_of_week            "T2".."CN" — nếu để trống sẽ tự tính từ departure_time.
  creation_hour           Giờ tạo bill, dạng "8h" hoặc "N/A" — nếu để trống sẽ
                         dùng "N/A".
  origin_province        Tỉnh kho gửi — CHỈ cần khi dự đoán cho pair thuộc
                         Luồng 2 (Kho 1A đích → Kho nhận).
  destination_province   Tỉnh kho nhận — CHỈ cần khi dự đoán cho pair thuộc
                         Luồng 1 (Kho gửi → Kho 1A nguồn).
  Thiếu cột nào không bắt buộc thì feature đó được coi là NaN/one-hot toàn 0,
  KHÔNG làm chương trình lỗi.

CÁCH DÙNG NHANH
  # 1) Xem các pair đã có model sẵn (copy đúng chuỗi 'pair' để dùng ở bước 2)
  python predict_arrival.py --list

  # 2) Dự đoán từ file CSV
  python predict_arrival.py \\
      --pair "Kho ABC (Hà Nội) → Kho XYZ (Hải Phòng)" \\
      --input lo_hang_moi.csv --output du_doan.csv

  # 3) Hoặc gọi trực tiếp trong Python
  from predict_arrival import predict_batch
  import pandas as pd
  df_new = pd.DataFrame([{
      "departure_time": "2026-08-05 09:15:00",
      "actual_weight": 12.5,
      "destination_province": "Hải Phòng",
      "VD_type": "VD1", "service": "Chuẩn", "creation_hour": "8h",
  }])
  ket_qua = predict_batch("Kho ABC (Hà Nội) → Kho XYZ (Hải Phòng)", df_new)
  print(ket_qua[["predicted_hours", "predicted_arrival_time", "model_used"]])
"""

import os
import sys
import argparse

import joblib
import numpy as np
import pandas as pd

# multi_classification.py nằm cùng thư mục với file này
_base_dir = os.path.dirname(os.path.abspath(__file__))
if _base_dir not in sys.path:
    sys.path.append(_base_dir)

from multi_classification import (   # noqa: E402  (import sau khi sửa sys.path là chủ ý)
    load_models,
    build_features_X2,
    pick_cluster_by_hour_from_info,
    MODELS_DIR,
    HYBRID_THRESHOLD,
)

# Các khóa metadata BẮT BUỘC phải có trong meta.joblib để inference đúng.
_REQUIRED_META_KEYS = [
    "tv_cols", "top_origin", "top_dest", "w_median", "is_dest_flow",
    "cluster_info", "best_combo", "dominant_cluster",
]

_DAY_MAP = {0: "T2", 1: "T3", 2: "T4", 3: "T5", 4: "T6", 5: "T7", 6: "CN"}


# ── Chuẩn bị dữ liệu đầu vào ────────────────────────────────────────────────
def _prepare_input(df_new: pd.DataFrame) -> pd.DataFrame:
    """Tính các cột phái sinh (departure_hour_numeric, departure_hour, day_of_week...)
    từ dữ liệu thô, y hệt cách multi_classification.py đã tính lúc train.
    """
    if "departure_time" not in df_new.columns:
        raise ValueError(
            "Thiếu cột bắt buộc 'departure_time' (thời điểm khởi hành khỏi kho đầu chặng)."
        )

    df = df_new.copy()
    dep = pd.to_datetime(df["departure_time"], errors="coerce")
    if dep.isna().any():
        n_bad = int(dep.isna().sum())
        raise ValueError(
            f"Có {n_bad} giá trị 'departure_time' không parse được thành datetime."
        )

    df["_departure_date"] = dep.dt.normalize()
    df["departure_hour_numeric"] = dep.dt.hour + dep.dt.minute / 60.0
    df["departure_hour"] = dep.dt.hour.astype(int).astype(str) + "h"

    if "day_of_week" in df.columns:
        computed = dep.dt.dayofweek.map(_DAY_MAP)
        df["day_of_week"] = df["day_of_week"].where(df["day_of_week"].notna(), computed)
    else:
        df["day_of_week"] = dep.dt.dayofweek.map(_DAY_MAP)

    if "creation_hour" in df.columns:
        df["creation_hour"] = df["creation_hour"].fillna("N/A")
    else:
        df["creation_hour"] = "N/A"

    for col in ("VD_type", "service", "origin_province", "destination_province", "actual_weight"):
        if col not in df.columns:
            df[col] = np.nan

    return df


# ── Dự đoán ──────────────────────────────────────────────────────────────
def predict_batch(pair: str, df_new: pd.DataFrame) -> pd.DataFrame:
    """Dự đoán số giờ đi đường (y_offset) và giờ đến dự kiến cho mỗi dòng trong df_new.

    Trả về df_new kèm 3 cột mới:
      - cluster_used          : cụm được chọn để áp model (theo best_combo)
      - predicted_hours       : số giờ dự kiến từ lúc khởi hành đến khi tới nơi
      - predicted_arrival_time: thời điểm dự kiến tới nơi (datetime)
      - model_used            : combo (chiến lược cụm | feature option | model) đã dùng
    """
    final_reg_models, final_clf, meta = load_models(pair)
    if final_reg_models is None:
        available = [p for p, _ in list_available_pairs()]
        hint = f" Các pair hiện có: {available}" if available else " Chưa có pair nào có model."
        raise FileNotFoundError(f"Chưa có model đã lưu cho pair: {pair!r}.{hint}")

    missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise RuntimeError(
            f"meta.joblib của pair {pair!r} thiếu khóa {missing}. "
            f"Model này được lưu bởi bản multi_classification.py CHƯA có patch bổ sung "
            f"metadata cho inference. Hãy chạy lại multi_classification.py (bản đã patch) "
            f"— nó sẽ tự phát hiện cache thiếu 'tv_cols' và tự train lại để bổ sung."
        )

    df = _prepare_input(df_new)

    is_dest_flow     = meta["is_dest_flow"]
    tv_cols          = meta["tv_cols"]
    top_origin       = meta["top_origin"]
    top_dest         = meta["top_dest"]
    w_median         = meta["w_median"]
    cluster_info     = meta["cluster_info"]
    best_combo       = meta["best_combo"]
    dominant_cluster = meta["dominant_cluster"]
    y_min, y_max     = meta.get("y_min"), meta.get("y_max")

    X1 = df[["departure_hour_numeric"]].copy()
    X2 = build_features_X2(df, is_dest_flow, top_origin, top_dest, w_median)
    X2 = X2.reindex(columns=tv_cols, fill_value=0)  # khớp chính xác schema lúc train

    dep_h = X1["departure_hour_numeric"].values

    cs_name  = best_combo["cs_name"]
    fo_name  = best_combo["fo_name"]
    reg_name = best_combo["reg_name"]

    # ── Chọn cụm theo đúng chiến lược đã thắng lúc val selection (best_combo) ──
    if cs_name == "Cách 1 (Dominant Cluster)":
        cluster_used = np.array([pick_cluster_by_hour_from_info(h, cluster_info) for h in dep_h])
    elif cs_name == "Cách 2 (ML Classifier)":
        cp = final_clf.predict_proba(X2)
        cluster_used = final_clf.classes_[np.argmax(cp, axis=1)]
    elif cs_name == "Cách 3 (Hybrid)":
        cp = final_clf.predict_proba(X2)
        cpred = final_clf.classes_[np.argmax(cp, axis=1)]
        cmax = np.max(cp, axis=1)
        dominant_by_hour = np.array([pick_cluster_by_hour_from_info(h, cluster_info) for h in dep_h])
        cluster_used = np.where(cmax >= HYBRID_THRESHOLD, cpred, dominant_by_hour)
    else:  # "Cách 0 (Không phân cụm)"
        cluster_used = np.zeros(len(df), dtype=int)

    X_opt, opt_key = (X1, "Opt1") if fo_name == "Option 1 (Hour)" else (X2, "Opt2")

    # ── Dự đoán y_offset theo model của best_combo, clip như lúc train ──
    y_pred = np.full(len(df), np.nan)
    for cid in np.unique(cluster_used):
        idx_c = np.where(cluster_used == cid)[0]
        key = (cid, reg_name, opt_key)
        if key not in final_reg_models:
            key = (dominant_cluster, reg_name, opt_key)  # fallback như lúc train
        if key not in final_reg_models:
            continue
        lo = (y_min - 2) if y_min is not None else -2
        hi = (y_max + 2) if y_max is not None else 50
        y_pred[idx_c] = np.clip(final_reg_models[key].predict(X_opt.iloc[idx_c]), lo, hi)

    out = df_new.copy()
    out["cluster_used"] = cluster_used
    # predicted_hours: y_offset dự đoán = số giờ tính từ 00:00 NGÀY KHỞI HÀNH (đúng định
    # nghĩa target lúc train, xem docstring multi_classification.py), KHÔNG PHẢI số giờ
    # tính từ departure_time. Dùng predicted_transit_hours bên dưới nếu cần thời lượng
    # đi đường thực tế tính từ đúng thời điểm khởi hành.
    out["predicted_hours"] = y_pred
    out["predicted_arrival_time"] = df["_departure_date"] + pd.to_timedelta(y_pred, unit="h")
    out["predicted_transit_hours"] = (
        (out["predicted_arrival_time"] - pd.to_datetime(df["departure_time"])).dt.total_seconds() / 3600.0
    )
    out["model_used"] = f"{cs_name} | {fo_name} | {reg_name}"
    return out


def predict_one(pair: str, shipment: dict) -> dict:
    """Tiện ích dự đoán cho 1 lô hàng duy nhất (dict), trả về dict kết quả."""
    df_out = predict_batch(pair, pd.DataFrame([shipment]))
    return df_out.iloc[0].to_dict()


def list_available_pairs():
    """Quét saved_models/ để liệt kê các pair đã có model (đọc trực tiếp meta.joblib,
    không cần biết trước tên pair)."""
    results = []
    if not os.path.isdir(MODELS_DIR):
        return results
    for slug in sorted(os.listdir(MODELS_DIR)):
        meta_path = os.path.join(MODELS_DIR, slug, "meta.joblib")
        if os.path.exists(meta_path):
            try:
                meta = joblib.load(meta_path)
                results.append((meta.get("pair", slug), meta.get("flow", "?")))
            except Exception:
                continue
    return results


# ── CLI ──────────────────────────────────────────────────────────────────
def _main():
    parser = argparse.ArgumentParser(
        description="Dự đoán thời gian tới nơi từ model đã train sẵn (inference-only)."
    )
    parser.add_argument("--pair", type=str, help="Tên pair, phải khớp CHÍNH XÁC lúc train.")
    parser.add_argument("--input", type=str, help="File CSV chứa các lô hàng mới cần dự đoán.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="File CSV kết quả.")
    parser.add_argument("--list", action="store_true", help="Liệt kê các pair đã có model.")
    args = parser.parse_args()

    if args.list:
        pairs = list_available_pairs()
        if not pairs:
            print(f"Chưa có model nào trong {MODELS_DIR}")
        else:
            print(f"Có {len(pairs)} pair đã có model:")
            for p, f in pairs:
                print(f"  [{f}] {p}")
        return

    if not args.pair or not args.input:
        parser.print_help()
        return

    df_new = pd.read_csv(args.input)
    result = predict_batch(args.pair, df_new)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Đã lưu kết quả dự đoán: {args.output}")


if __name__ == "__main__":
    _main()