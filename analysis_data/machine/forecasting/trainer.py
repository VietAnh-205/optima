# -*- coding: utf-8 -*-
"""
forecasting/trainer.py
=======================
Training pipeline — refactor từ toàn bộ bước 1→9 của forecating.py.
Tham số hóa selected_set (tên kho 1A) để train cho bất kỳ kho nào.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

from .config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    HIGH_CARD_COLS,
    LGBM_PARAMS,
    MISSING_OUT_WEIGHT,
    RARE_CATEGORY_MIN_COUNT,
    TRAIN_RATIO,
)
from .features import (
    align_categoricals,
    create_stage_features,
    engineer_bill_features,
    fit_and_apply_rare_categories,
    remove_outliers_iqr,
)
from .model_io import save_bundle


# ════════════════════════════════════════════════════════════════════════
# Load data helpers
# ════════════════════════════════════════════════════════════════════════

def _load_bill_schedule(path: str, chunksize: int = 500_000) -> pd.DataFrame:
    """Đọc bill_schedule.csv theo chunk (dữ liệu lớn), parse io_time."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk["io_time"] = pd.to_datetime(chunk["io_time"])
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df.sort_values(["bill_code", "io_time"]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ════════════════════════════════════════════════════════════════════════

def train_pipeline(
    selected_set: set,
    data_dir: str,
    train_ratio: float = TRAIN_RATIO,
    output_dir: str | None = None,
) -> dict:
    """
    Chạy full pipeline: đọc data → lọc → trajectory → features → train.

    Parameters
    ----------
    selected_set : set
        Tên kho 1A đích. VD: {'Kho Sóng Thần'}
    data_dir : str
        Thư mục chứa bill.csv, bill_schedule.csv, warehouse_1A.csv
    train_ratio : float
        Tỉ lệ train/test split (mặc định 0.9)
    output_dir : str or None
        Thư mục lưu model bundle. Nếu None → không tự động lưu.

    Returns
    -------
    dict : model_bundle chứa tất cả cần thiết để predict
    """
    destination = next(iter(selected_set))
    print("=" * 60)
    print(f"TRAINING PIPELINE — {destination}")
    print("=" * 60)

    # ── 1. ĐỌC DỮ LIỆU ─────────────────────────────────────────────
    print("\n[1] Đọc dữ liệu...")
    warehouse_1a_path = os.path.join(data_dir, "warehouse_1A.csv")
    bill_path = os.path.join(data_dir, "bill.csv")
    bill_schedule_path = os.path.join(data_dir, "bill_schedule.csv")

    wh1a = pd.read_csv(warehouse_1a_path)
    set_1a = set(wh1a["name"].dropna().str.strip())

    bill_full = pd.read_csv(bill_path)
    sche = _load_bill_schedule(bill_schedule_path)
    print(f"  bill_schedule shape: {sche.shape}")

    # ── 3. LỌC BILLS ĐI QUA KHO 1A ĐƯỢC CHỌN ───────────────────────
    print("\n[3] Lọc bills đi qua kho 1A được chọn...")

    # Chỉ giữ các bill có ít nhất 1 lần đến kho đích
    valid_bills = sche.loc[
        sche["warehouse_name"].isin(selected_set),
        "bill_code",
    ].unique()

    if len(valid_bills) == 0:
        raise ValueError(
            f"Không tìm thấy bill nào đi qua kho '{destination}'.\n"
            f"Kiểm tra lại tên kho — phải khớp chính xác (kể cả hoa/thường) với\n"
            f"cột warehouse_name trong bill_schedule.csv.\n"
            f"Gợi ý: dùng tên đúng từ warehouse_1A.csv, VD: 'Kho Sóng Thần'"
        )

    # Chỉ xử lý bills có đi qua kho đích
    sche = sche[sche["bill_code"].isin(valid_bills)].copy()
    sche["row_id"] = sche.groupby("bill_code").cumcount()

    # Tìm row_id đầu tiên tại kho đích cho mỗi bill
    first_selected = (
        sche[sche["warehouse_name"].isin(selected_set)]
        .groupby("bill_code")["row_id"]
        .min()
        .rename("first_selected_row")
    )

    sche = sche.merge(first_selected, on="bill_code", how="inner")

    # Chỉ giữ hành trình TỪ ĐẦU ĐẾN (và gồm) lần đến kho đích đầu tiên
    sche_filtered = sche[
        sche["row_id"] <= sche["first_selected_row"]
    ].drop(columns=["row_id", "first_selected_row"]).copy()

    # Chỉ giữ bills chưa qua kho 1A nào khác trước khi đến kho đích
    set_1a_without_selected = set_1a - selected_set

    bills_to_remove = sche_filtered.loc[
        sche_filtered["warehouse_name"].isin(set_1a_without_selected),
        "bill_code",
    ].unique()

    sche_filtered = sche_filtered[
        ~sche_filtered["bill_code"].isin(bills_to_remove)
    ].copy()

    print(f"  valid_bills: {len(valid_bills)}, sau lọc kho 1A khác: {sche_filtered['bill_code'].nunique()} bills")
    print(f"  sche_filtered shape: {sche_filtered.shape}")

    if sche_filtered.empty:
        raise ValueError(
            f"Sau khi lọc, không còn bill nào hợp lệ cho kho '{destination}'.\n"
            f"Tất cả bills đều đã qua kho 1A khác trước khi đến kho này."
        )

    sche = sche_filtered.sort_values(["bill_code", "io_time"]).copy()

    # ── 4. XÂY DỰNG TRAJECTORY ──────────────────────────────────────
    print("\n[4] Xây dựng trajectory...")
    trajectory_df = (
        sche.groupby("bill_code")
        .agg({
            "warehouse_name": list,
            "io_status": list,
            "io_time": list,
        })
        .reset_index()
    )

    trajectory_df["trajectory"] = trajectory_df.apply(
        lambda r: list(zip(r.warehouse_name, r.io_status)),
        axis=1,
    )

    # Thời gian đến kho 1A
    arrival = (
        sche[
            (sche.warehouse_name == destination)
            & (sche.io_status == "IN")
        ]
        .groupby("bill_code")
        .first()["io_time"]
    )

    trajectory_df["arrival_time"] = trajectory_df.bill_code.map(arrival)
    trajectory_df = trajectory_df.dropna(subset=["arrival_time"])

    trajectory_df["arrival_sec"] = (
        trajectory_df.arrival_time.astype("int64") // 10**9
    )

    # Lọc trajectory có độ dài ít nhất 2
    trajectory_df = trajectory_df[
        trajectory_df["io_status"].apply(len) >= 2
    ].copy()

    # Kho trước kho 1A (warehouse_2last)
    trajectory_df["warehouse_2last"] = trajectory_df["warehouse_name"].apply(
        lambda x: x[-2]
    )

    print(f"  trajectory_df shape: {trajectory_df.shape}")
    print(f"  destination: {destination}")

    # ── 5. TẠO STAGE FEATURES ───────────────────────────────────────
    print("\n[5] Tạo stage features...")
    stage_features = create_stage_features(trajectory_df)
    stage_features["io_status_source"] = stage_features["io_status_source"].astype(str)
    print(f"  stage_features shape: {stage_features.shape}")

    # ── 6. LOẠI BỎ OUTLIER ──────────────────────────────────────────
    print("\n[6] Loại bỏ outlier (IQR)...")
    stage_features_clean = remove_outliers_iqr(stage_features)

    # ── 7. MERGE BILL INFO + FEATURE ENGINEERING ────────────────────
    print("\n[7] Feature engineering...")
    bill_full_subset = bill_full.merge(
        stage_features_clean,
        on="bill_code",
        how="inner",
    )
    print(f"  bill_full_subset shape: {bill_full_subset.shape}")

    df = engineer_bill_features(bill_full_subset)

    # Sort theo thời gian
    df = df.sort_values("arrival_time").reset_index(drop=True)
    df = df.dropna(subset=["arrival_offset_hours"]).copy()
    print(f"  Engineered df shape: {df.shape}")

    # ── 8. CHIA TRAIN / TEST ────────────────────────────────────────
    print("\n[8] Chia train / test...")
    bill_arrival = df.groupby("bill_code")["arrival_time"].first().sort_values()
    bill_split_idx = int(len(bill_arrival) * train_ratio)
    train_bills = set(bill_arrival.index[:bill_split_idx])
    test_bills = set(bill_arrival.index[bill_split_idx:])

    train = df[df["bill_code"].isin(train_bills)].copy()
    test = df[df["bill_code"].isin(test_bills)].copy()

    print(f"  Total : {len(df)}")
    print(f"  Train : {len(train)}")
    print(f"  Test  : {len(test)}")

    # ── 8b. GỘP CATEGORY HIẾM ──────────────────────────────────────
    print("\n[8b] Gộp category hiếm...")
    train, test, rare_keep_sets = fit_and_apply_rare_categories(
        train, test,
        cols=HIGH_CARD_COLS,
        min_count=RARE_CATEGORY_MIN_COUNT,
    )

    # ── 9. HUẤN LUYỆN MÔ HÌNH LIGHTGBM ─────────────────────────────
    print("\n[9] Huấn luyện mô hình LightGBM...")

    # Make categorical
    category_mappings = {}
    for col in CATEGORICAL_FEATURES:
        train[col] = train[col].astype("category")
        category_mappings[col] = train[col].cat.categories.tolist()
        test[col] = pd.Categorical(
            test[col],
            categories=train[col].cat.categories,
        )

    X_train = train[FEATURE_COLS]
    y_train = train["arrival_offset_hours"]

    X_test = test[FEATURE_COLS]
    y_test = test["arrival_offset_hours"]

    # Sample weight
    sample_weight = np.where(
        train["is_missing_out"] == 1, MISSING_OUT_WEIGHT, 1.0
    )

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    print("  Model training complete!")

    # ── METRICS ─────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = {
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_medae": float(median_absolute_error(y_test, y_pred_test)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "train_size": len(train),
        "test_size": len(test),
    }

    print(f"\n  Train MAE : {train_metrics['train_mae']:.4f} hours")
    print(f"  Test  MAE : {train_metrics['test_mae']:.4f} hours")
    print(f"  Test  MedAE: {train_metrics['test_medae']:.4f} hours")
    print(f"  Test  R²  : {train_metrics['test_r2']:.4f}")

    # ── BUILD BUNDLE ────────────────────────────────────────────────
    bundle = {
        "model": model,
        "selected_set": selected_set,
        "feature_cols": FEATURE_COLS,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_mappings": category_mappings,
        "rare_keep_sets": rare_keep_sets,
        "train_metrics": train_metrics,
        "trained_at": datetime.now().isoformat(),
    }

    # ── SAVE (nếu có output_dir) ────────────────────────────────────
    if output_dir is not None:
        save_bundle(bundle, output_dir)

    print("\n" + "=" * 60)
    print(f"DONE — {destination}")
    print("=" * 60)

    return bundle
