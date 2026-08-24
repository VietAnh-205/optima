# -*- coding: utf-8 -*-
"""
forecasting/features.py
========================
Feature engineering — dùng chung cho cả training và prediction.
Trích xuất từ bước 5, 6, 7, 8b của forecating.py gốc.

Nguyên tắc: viết 1 lần, dùng 2 nơi → train và predict luôn nhất quán.
"""

import numpy as np
import pandas as pd

from .config import (
    ARRIVAL_OFFSET_MAX_HOURS,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    HIGH_CARD_COLS,
    RARE_CATEGORY_MIN_COUNT,
)


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 5 — Tạo stage features
# ════════════════════════════════════════════════════════════════════════

def create_stage_features(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng tại mỗi bước hành trình của đơn.
    Mỗi lần đơn check in/out tại một kho → 1 điểm dữ liệu.
    Thêm phiên bản missing data cho OUT (io_status=-1, io_time=-1).

    Parameters
    ----------
    trajectory_df : DataFrame
        Cần có cột: bill_code, warehouse_name, io_status, io_time, arrival_time
        (warehouse_name, io_status, io_time là list-of-values cho mỗi bill)

    Returns
    -------
    DataFrame với cột:
        bill_code, warehouse_source, io_status_source, io_time_source,
        arrival_time, arrival_offset_hours, is_missing_out
    """
    # 1. Select required columns
    df = trajectory_df[
        ["bill_code", "warehouse_name", "io_status", "io_time", "arrival_time"]
    ].copy()

    # 2. Convert arrival_time once
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")

    # 3. Remove LAST stage (it is the output/destination)
    df["warehouse_name"] = df["warehouse_name"].str[:-1]
    df["io_status"] = df["io_status"].str[:-1]
    df["io_time"] = df["io_time"].str[:-1]

    # 4. Explode stages
    df = df.explode(
        ["warehouse_name", "io_status", "io_time"],
        ignore_index=True,
    )
    df = df.dropna(subset=["io_status"])

    # 5. Rename source features
    df = df.rename(columns={
        "warehouse_name": "warehouse_source",
        "io_status": "io_status_source",
    })

    # 6. Convert source time to datetime
    df["io_time_source_datetime"] = pd.to_datetime(df["io_time"], errors="coerce")

    # 7. Decimal clock hour (e.g. 22:20:06 -> 22.335)
    t = df["io_time_source_datetime"]
    df["io_time_source"] = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600

    # 8. Arrival offset from SOURCE DATE MIDNIGHT
    df["arrival_offset_hours"] = (
        df["arrival_time"] - df["io_time_source_datetime"].dt.normalize()
    ).dt.total_seconds() / 3600

    # 9. Create missing-data version for OUT
    df["is_missing_out"] = 0
    out_rows = df[df["io_status_source"].eq("OUT")].copy()
    out_rows["io_status_source"] = -1
    out_rows["io_time_source"] = -1
    out_rows["is_missing_out"] = 1

    # 10. Combine normal + missing versions
    result = pd.concat([df, out_rows], ignore_index=True)

    # 11. Keep required columns
    result = result[[
        "bill_code",
        "warehouse_source",
        "io_status_source",
        "io_time_source",
        "arrival_time",
        "arrival_offset_hours",
        "is_missing_out",
    ]]

    return result


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 6 — Loại bỏ outlier (IQR)
# ════════════════════════════════════════════════════════════════════════

def remove_outliers_iqr(
    df: pd.DataFrame,
    max_hours: float = ARRIVAL_OFFSET_MAX_HOURS,
) -> pd.DataFrame:
    """
    Loại bỏ outlier bằng IQR theo nhóm (warehouse_source, io_status_source).

    Parameters
    ----------
    df : DataFrame
        Cần có cột: warehouse_source, io_status_source, arrival_offset_hours
    max_hours : float
        Ngưỡng tối đa tuyệt đối cho arrival_offset_hours (mặc định 72h)

    Returns
    -------
    DataFrame đã loại bỏ outlier
    """
    df = df.dropna(
        subset=["warehouse_source", "io_status_source",
                "io_time_source", "arrival_offset_hours"]
    ).copy()

    group_cols = ["warehouse_source", "io_status_source"]

    Q1 = df.groupby(group_cols)["arrival_offset_hours"].transform("quantile", 0.25)
    Q3 = df.groupby(group_cols)["arrival_offset_hours"].transform("quantile", 0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    is_outlier = (
        (df["arrival_offset_hours"] < lower_bound)
        | (df["arrival_offset_hours"] > upper_bound)
    )

    n_before = len(df)
    n_outlier = int(is_outlier.sum())
    df = df[~is_outlier].copy()
    df = df[df["arrival_offset_hours"] <= max_hours]

    print(f"  [remove_outliers_iqr] Before: {n_before}, "
          f"Outliers: {n_outlier}, After: {len(df)}")

    return df


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 7 — Feature engineering từ bill info
# ════════════════════════════════════════════════════════════════════════

def engineer_bill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo ~25 features từ thông tin bill.
    Nhận DataFrame đã merge giữa bill_full và stage_features_clean.

    Các nhóm features:
        7.1  Bill creation date features
        7.2  Estimated delivery date features
        7.3  Estimated delivery duration
        7.4  Relationship between creation and delivery dates
        7.5  Cyclic time features
        7.6  Route features
        7.7  Weight features
        7.8  Categorical interaction features

    Parameters
    ----------
    df : DataFrame
        Cần có cột: bill_creation_date, estimated_delivery_date,
        origin_province, origin_district, destination_province,
        destination_district, service, VD_type, actual_weight

    Returns
    -------
    DataFrame với các features mới được thêm vào
    """
    df = df.copy()

    # Fill optional columns with NaN if missing
    optional_cols = [
        "bill_creation_date", "estimated_delivery_date",
        "origin_province", "origin_district", 
        "destination_province", "destination_district", 
        "service", "VD_type", "actual_weight"
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --- Datetime conversion ---
    df["bill_creation_date"] = pd.to_datetime(
        df["bill_creation_date"], errors="coerce"
    )
    df["estimated_delivery_date"] = pd.to_datetime(
        df["estimated_delivery_date"], errors="coerce"
    )

    # --- 7.1 Bill creation date features ---
    df["creation_hour"] = (
        df["bill_creation_date"].dt.hour
        + df["bill_creation_date"].dt.minute / 60
        + df["bill_creation_date"].dt.second / 3600
    )
    df["creation_day_of_week"] = df["bill_creation_date"].dt.dayofweek
    df["creation_day_of_month"] = df["bill_creation_date"].dt.day
    df["creation_month"] = df["bill_creation_date"].dt.month
    df["creation_week_of_year"] = (
        df["bill_creation_date"].dt.isocalendar().week.astype("float")
    )
    df["creation_day_of_year"] = df["bill_creation_date"].dt.dayofyear
    df["creation_weekend"] = (df["creation_day_of_week"] >= 5).astype(int)

    # --- 7.2 Estimated delivery date features ---
    df["estimated_day_of_week"] = df["estimated_delivery_date"].dt.dayofweek
    df["estimated_day_of_month"] = df["estimated_delivery_date"].dt.day
    df["estimated_month"] = df["estimated_delivery_date"].dt.month
    df["estimated_week_of_year"] = (
        df["estimated_delivery_date"].dt.isocalendar().week.astype("float")
    )
    df["estimated_day_of_year"] = df["estimated_delivery_date"].dt.dayofyear
    df["estimated_weekend"] = (df["estimated_day_of_week"] >= 5).astype(int)

    # --- 7.3 Estimated delivery duration ---
    df["estimated_delivery_days"] = (
        df["estimated_delivery_date"] - df["bill_creation_date"]
    ).dt.total_seconds() / (24 * 3600)

    df["estimated_delivery_days_cat"] = pd.cut(
        df["estimated_delivery_days"],
        bins=[-np.inf, 0, 1, 2, 3, 5, 7, 14, np.inf],
        labels=["<=0", "0-1", "1-2", "2-3", "3-5", "5-7", "7-14", ">14"],
    )

    # --- 7.4 Relationship between creation and delivery dates ---
    df["creation_delivery_dow_diff"] = (
        df["estimated_day_of_week"] - df["creation_day_of_week"]
    )
    df["creation_delivery_dow_diff_circular"] = (
        df["estimated_day_of_week"] - df["creation_day_of_week"]
    ) % 7
    df["same_creation_delivery_dow"] = (
        df["creation_day_of_week"] == df["estimated_day_of_week"]
    ).astype(int)
    df["same_creation_delivery_month"] = (
        df["creation_month"] == df["estimated_month"]
    ).astype(int)

    # --- 7.5 Cyclic time features ---
    df["creation_hour_sin"] = np.sin(2 * np.pi * df["creation_hour"] / 24)
    df["creation_hour_cos"] = np.cos(2 * np.pi * df["creation_hour"] / 24)
    df["creation_dow_sin"] = np.sin(2 * np.pi * df["creation_day_of_week"] / 7)
    df["creation_dow_cos"] = np.cos(2 * np.pi * df["creation_day_of_week"] / 7)
    df["creation_month_sin"] = np.sin(2 * np.pi * df["creation_month"] / 12)
    df["creation_month_cos"] = np.cos(2 * np.pi * df["creation_month"] / 12)

    # --- 7.6 Route features ---
    df["province_route"] = (
        df["origin_province"].astype(str) + " -> " + df["destination_province"].astype(str)
    )
    df["district_route"] = (
        df["origin_district"].astype(str) + " -> " + df["destination_district"].astype(str)
    )
    df["full_route"] = (
        df["origin_province"].astype(str) + "_" + df["origin_district"].astype(str)
        + " -> "
        + df["destination_province"].astype(str) + "_" + df["destination_district"].astype(str)
    )

    # --- 7.7 Weight features ---
    df["actual_weight"] = pd.to_numeric(df["actual_weight"], errors="coerce")
    df["actual_weight"] = df["actual_weight"].clip(lower=0)
    df["log_actual_weight"] = np.log1p(df["actual_weight"])
    df["weight_bin"] = pd.cut(
        df["actual_weight"],
        bins=[-np.inf, 0.5, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["<=0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">50"],
    )

    # --- 7.8 Categorical interaction features ---
    df["service_VD_type"] = df["service"].astype(str) + "_" + df["VD_type"].astype(str)
    df["service_creation_dow"] = df["service"].astype(str) + "_" + df["creation_day_of_week"].astype(str)
    df["service_creation_hour"] = df["service"].astype(str) + "_" + df["creation_hour"].round().astype(str)
    df["VD_type_creation_dow"] = df["VD_type"].astype(str) + "_" + df["creation_day_of_week"].astype(str)
    df["route_service"] = df["province_route"].astype(str) + "_" + df["service"].astype(str)
    df["route_VD_type"] = df["province_route"].astype(str) + "_" + df["VD_type"].astype(str)

    return df


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 8b — Gộp category hiếm
# ════════════════════════════════════════════════════════════════════════

def fit_rare_categories(
    series: pd.Series,
    min_count: int = RARE_CATEGORY_MIN_COUNT,
) -> set:
    """
    Học keep_set từ train data: chỉ giữ các category có tần suất >= min_count.

    Returns
    -------
    set : tập hợp các category được giữ lại
    """
    freq = series.value_counts()
    return set(freq[freq >= min_count].index)


def apply_rare_categories(
    series: pd.Series,
    keep_set: set,
    other_label: str = "OTHER",
) -> pd.Series:
    """
    Gộp các category không nằm trong keep_set thành other_label.
    """
    return series.where(series.isin(keep_set), other_label)


def fit_and_apply_rare_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str] | None = None,
    min_count: int = RARE_CATEGORY_MIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set]]:
    """
    Fit rare categories trên train, apply lên cả train và test.

    Returns
    -------
    (train, test, rare_keep_sets)
    """
    if cols is None:
        cols = HIGH_CARD_COLS

    rare_keep_sets = {}
    for col in cols:
        n_before = train[col].nunique()
        keep_set = fit_rare_categories(train[col], min_count)
        train[col] = apply_rare_categories(train[col], keep_set)
        test[col] = apply_rare_categories(test[col], keep_set)
        rare_keep_sets[col] = keep_set
        print(f"  [rare_categories] {col}: {n_before} -> "
              f"{train[col].nunique()} categories (min_count={min_count})")

    return train, test, rare_keep_sets


# ════════════════════════════════════════════════════════════════════════
# Align categorical dtype giữa train/test/predict
# ════════════════════════════════════════════════════════════════════════

def align_categoricals(
    df: pd.DataFrame,
    category_mappings: dict[str, list],
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Align categorical dtype cho df dựa trên category_mappings đã learn từ train.

    Parameters
    ----------
    df : DataFrame cần align
    category_mappings : dict {col_name: list_of_categories}
    cols : danh sách cột cần align (mặc định CATEGORICAL_FEATURES)

    Returns
    -------
    DataFrame đã align
    """
    if cols is None:
        cols = CATEGORICAL_FEATURES

    df = df.copy()
    for col in cols:
        if col in df.columns and col in category_mappings:
            df[col] = pd.Categorical(
                df[col],
                categories=category_mappings[col],
            )
    return df


# ════════════════════════════════════════════════════════════════════════
# Hàm tổng hợp cho prediction
# ════════════════════════════════════════════════════════════════════════

def prepare_prediction_features(
    df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    """
    Hàm tổng hợp: nhận raw data dạng phẳng → trả X sẵn sàng cho model.predict().

    Xử lý:
        1. Tự sinh bill_code nếu thiếu
        2. Tạo dynamic stage features từ warehouse_name, io_status, io_time
        3. Engineer bill features (tất cả cột đều optional)
        4. Apply rare categories + align categoricals
        5. Trả về X đúng 41 features, đúng thứ tự (cột thiếu được fill NaN)

    Parameters
    ----------
    df : DataFrame
        Thông tin đơn hàng và trạng thái hiện tại trên 1 dòng.
        Các cột (tất cả là optional):
            warehouse_name, io_status, io_time, bill_code, VD_type,
            bill_creation_date, estimated_delivery_date, origin_province,
            origin_district, destination_province, destination_district,
            service, actual_weight

    bundle : dict
        Model bundle chứa: category_mappings, rare_keep_sets, feature_cols,
        categorical_features

    Returns
    -------
    DataFrame (result_df) với các cột:
        bill_code, warehouse_source, io_status_source, io_time_source,
        + tất cả feature_cols (sẵn sàng cho model.predict)
    """
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)
    categorical_features = bundle.get("categorical_features", CATEGORICAL_FEATURES)
    category_mappings = bundle.get("category_mappings", {})
    rare_keep_sets = bundle.get("rare_keep_sets", {})

    result = df.copy()

    # 1. Tự sinh bill_code nếu thiếu
    if "bill_code" not in result.columns:
        result["bill_code"] = [f"bill_{i}" for i in range(len(result))]

    # 2. Xử lý stage features
    if "warehouse_name" in result.columns:
        result = result.rename(columns={"warehouse_name": "warehouse_source"})
    else:
        result["warehouse_source"] = np.nan

    if "io_status" in result.columns:
        result = result.rename(columns={"io_status": "io_status_source"})
    else:
        result["io_status_source"] = np.nan

    if "io_time" in result.columns:
        io_time = pd.to_datetime(result["io_time"], errors="coerce")
        result["io_time_source"] = (
            io_time.dt.hour + io_time.dt.minute / 60 + io_time.dt.second / 3600
        )
    else:
        result["io_time_source"] = np.nan

    # 3. Engineer bill features
    result = engineer_bill_features(result)

    # 4. Apply rare categories
    for col, keep_set in rare_keep_sets.items():
        if col in result.columns:
            result[col] = apply_rare_categories(result[col], keep_set)

    # 5. Align categoricals
    result["io_status_source"] = result["io_status_source"].astype(str)
    result["warehouse_source"] = result["warehouse_source"].astype(str)
    result = align_categoricals(result, category_mappings, categorical_features)

    # 6. Fill missing features with NaN so model doesn't crash
    for c in feature_cols:
        if c not in result.columns:
            result[c] = np.nan

    return result
