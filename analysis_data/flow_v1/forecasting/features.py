# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .config import (
    ARRIVAL_OFFSET_MAX_HOURS,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    HIGH_CARD_COLS,
    RARE_CATEGORY_MIN_COUNT,
    SNAPSHOT_HOURS,
    SNAPSHOT_HOUR,
)


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 5 — Tạo stage features
# ════════════════════════════════════════════════════════════════════════

def create_stage_features(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng tại mỗi bước hành trình của đơn.
    Mỗi lần đơn check in/out tại một kho → 1 điểm dữ liệu.

    Parameters
    ----------
    trajectory_df : DataFrame
        Cần có cột: bill_code, warehouse_name, io_status, io_time, arrival_time
        (warehouse_name, io_status, io_time là list-of-values cho mỗi bill)

    Returns
    -------
    DataFrame với các cột:
        bill_code, warehouse_source, io_status_source, io_time_source,
        io_time_source_datetime, stage_depth, arrival_time, arrival_offset_hours
    """
    df = trajectory_df[
        ["bill_code", "warehouse_name", "io_status", "io_time", "arrival_time"]
    ].copy()

    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")

    # Bỏ chặng cuối (là điểm đến kho 1A đích)
    df["warehouse_name"] = df["warehouse_name"].str[:-1]
    df["io_status"] = df["io_status"].str[:-1]
    df["io_time"] = df["io_time"].str[:-1]

    # Explode từng chặng
    df = df.explode(
        ["warehouse_name", "io_status", "io_time"],
        ignore_index=True,
    )
    df = df.dropna(subset=["io_status", "warehouse_name", "io_time"])
    df["stage_depth"] = df.groupby("bill_code").cumcount()

    df = df.rename(columns={
        "warehouse_name": "warehouse_source",
        "io_status": "io_status_source",
    })

    df["io_time_source_datetime"] = pd.to_datetime(df["io_time"], errors="coerce")

    # Giờ thập phân (VD 22:20:06 -> 22.335)
    t = df["io_time_source_datetime"]
    df["io_time_source"] = t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0

    # #4: Tính chính xác thời gian di chuyển còn lại thực tế từ lúc quét đến khi tới nơi
    df["arrival_offset_hours"] = (
        df["arrival_time"] - df["io_time_source_datetime"]
    ).dt.total_seconds() / 3600.0

    result = df[[
        "bill_code",
        "warehouse_source",
        "io_status_source",
        "io_time_source",
        "io_time_source_datetime",
        "stage_depth",
        "arrival_time",
        "arrival_offset_hours",
    ]].copy()

    return result


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 6 — Outlier Bounds Fit & Apply (Tránh Leakage #2)
# ════════════════════════════════════════════════════════════════════════

def fit_outlier_bounds(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Tính ngưỡng outlier (Q1 - 1.5*IQR, Q3 + 1.5*IQR) CHỈ trên tập Train.
    """
    if group_cols is None:
        group_cols = ["warehouse_source", "io_status_source"]

    valid = df.dropna(subset=group_cols + ["arrival_offset_hours"])
    grouped = valid.groupby(group_cols)["arrival_offset_hours"]
    Q1 = grouped.quantile(0.25)
    Q3 = grouped.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound


def apply_outlier_bounds(
    df: pd.DataFrame,
    lower_bound: pd.Series,
    upper_bound: pd.Series,
    max_hours: float = ARRIVAL_OFFSET_MAX_HOURS,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Áp dụng ngưỡng outlier đã fit từ train lên dataframe (train/val/test).
    """
    if df.empty:
        return df

    if group_cols is None:
        group_cols = ["warehouse_source", "io_status_source"]

    df = df.copy()
    key = list(zip(*(df[c] for c in group_cols)))
    lo = pd.Series(key, index=df.index).map(lower_bound).fillna(0.0)
    hi = pd.Series(key, index=df.index).map(upper_bound).fillna(max_hours)

    keep = (
        (df["arrival_offset_hours"] >= lo)
        & (df["arrival_offset_hours"] <= hi)
        & (df["arrival_offset_hours"] <= max_hours)
        & (df["arrival_offset_hours"] >= 0.0)
    )

    n_before = len(df)
    df_filtered = df[keep].copy()
    n_after = len(df_filtered)
    print(f"  [apply_outlier_bounds] Before: {n_before}, Outliers removed: {n_before - n_after}, After: {n_after}")

    return df_filtered


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 7 — Feature engineering từ bill info
# ════════════════════════════════════════════════════════════════════════

def engineer_bill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các features phong phú từ thông tin bill.
    """
    df = df.copy()

    # --- 7.1 Date conversions ---
    if "bill_creation_date" in df.columns:
        df["bill_creation_date"] = pd.to_datetime(df["bill_creation_date"], errors="coerce")
        c = df["bill_creation_date"]
        df["creation_hour"] = c.dt.hour + c.dt.minute / 60.0
        df["creation_day_of_week"] = c.dt.dayofweek
        df["creation_month"] = c.dt.month
        df["creation_day"] = c.dt.day
    else:
        df["creation_hour"] = np.nan
        df["creation_day_of_week"] = np.nan
        df["creation_month"] = np.nan
        df["creation_day"] = np.nan

    if "estimated_delivery_date" in df.columns:
        df["estimated_delivery_date"] = pd.to_datetime(df["estimated_delivery_date"], errors="coerce")
        e = df["estimated_delivery_date"]
        df["estimated_day_of_week"] = e.dt.dayofweek
        df["estimated_month"] = e.dt.month
        df["estimated_day"] = e.dt.day
    else:
        df["estimated_day_of_week"] = np.nan
        df["estimated_month"] = np.nan
        df["estimated_day"] = np.nan

    # --- 7.2 Estimated delivery duration ---
    if "estimated_delivery_date" in df.columns and "bill_creation_date" in df.columns:
        diff_days = (df["estimated_delivery_date"] - df["bill_creation_date"]).dt.total_seconds() / (24 * 3600)
        df["estimated_delivery_days"] = diff_days
        
        # #17: Cảnh báo nếu estimated_delivery_date < bill_creation_date
        n_neg = (df["estimated_delivery_days"] < 0).sum()
        if n_neg > 0:
            print(f"  [WARN] Phát hiện {n_neg} bills có estimated_delivery_date < bill_creation_date")
    else:
        df["estimated_delivery_days"] = np.nan

    # --- 7.3 Binned delivery days ---
    df["estimated_delivery_days_cat"] = pd.cut(
        df["estimated_delivery_days"],
        bins=[-np.inf, 0, 1, 2, 3, 5, 7, np.inf],
        labels=["<=0", "1", "2", "3", "4-5", "6-7", ">7"],
    )

    # --- 7.4 Date relationship features ---
    df["creation_delivery_dow_diff"] = df["estimated_day_of_week"] - df["creation_day_of_week"]
    df["creation_delivery_dow_diff_circular"] = (
        (df["estimated_day_of_week"] - df["creation_day_of_week"]) % 7
    )
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
    origin_p = df["origin_province"].fillna("UNKNOWN").astype(str) if "origin_province" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    dest_p = df["destination_province"].fillna("UNKNOWN").astype(str) if "destination_province" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    origin_d = df["origin_district"].fillna("UNKNOWN").astype(str) if "origin_district" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    dest_d = df["destination_district"].fillna("UNKNOWN").astype(str) if "destination_district" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)

    df["province_route"] = origin_p + " -> " + dest_p
    df["district_route"] = origin_d + " -> " + dest_d

    # --- 7.7 Weight features ---
    if "actual_weight" in df.columns:
        df["actual_weight"] = pd.to_numeric(df["actual_weight"], errors="coerce")
        df["actual_weight"] = df["actual_weight"].clip(lower=0)
    else:
        df["actual_weight"] = np.nan

    df["log_actual_weight"] = np.log1p(df["actual_weight"])
    df["weight_bin"] = pd.cut(
        df["actual_weight"],
        bins=[-np.inf, 0.5, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["<=0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">50"],
    )

    # --- 7.8 Categorical interaction features ---
    svc = df["service"].fillna("UNKNOWN").astype(str) if "service" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    vd = df["VD_type"].fillna("UNKNOWN").astype(str) if "VD_type" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    df["service_VD_type"] = svc + "_" + vd

    return df


# ════════════════════════════════════════════════════════════════════════
# BƯỚC F — Snapshot 20:00 & Real-Time Point-In-Time Generation
# ════════════════════════════════════════════════════════════════════════

def generate_pit_snapshot(
    df: pd.DataFrame,
    snapshot_hour: int = SNAPSHOT_HOUR,
    snapshot_hours: list[int] | None = None,
) -> pd.DataFrame:
    """
    Tạo dữ liệu Point-In-Time (PIT) snapshot cho từng đơn hàng (Chạy siêu tốc Vectorized):
      - Sinh mốc snapshot cố định lúc 20:00 hàng ngày.
      - Thêm mốc snapshot ngay tại thời điểm quét của mỗi chặng (hours_since_last_scan = 0)
        để phục vụ real-time serving và bao quát các đơn giao nhanh trong ngày (#1, #7).
      - Áp dụng Feature Dropout ngẫu nhiên cho bản ghi OUT sau khi merge (#6).
    """
    if "io_time_source_datetime" not in df.columns or df.empty:
        return df

    if snapshot_hours is not None and len(snapshot_hours) == 1:
        snapshot_hour = snapshot_hours[0]

    df = df.sort_values("io_time_source_datetime").copy()

    # 1. Mốc quét thực tế của mỗi chặng (real-time point)
    scan_points = df[["bill_code", "io_time_source_datetime"]].copy()
    scan_points = scan_points.rename(columns={"io_time_source_datetime": "snapshot_time"})

    # 2. Vectorized 20:00 snapshot cho từng ngày giữa start_time và arrival_time
    bill_bounds = df.groupby("bill_code").agg(
        start_time=("io_time_source_datetime", "min"),
        arrival_time=("arrival_time", "first"),
    ).reset_index()

    bill_bounds["days"] = (
        bill_bounds["arrival_time"].dt.normalize() - bill_bounds["start_time"].dt.normalize()
    ).dt.days

    max_days = int(bill_bounds["days"].max()) if not bill_bounds.empty and bill_bounds["days"].max() > 0 else 0
    snap_dfs = []
    for d in range(max_days + 1):
        sub = bill_bounds[bill_bounds["days"] >= d].copy()
        sub["snapshot_time"] = sub["start_time"].dt.normalize() + pd.Timedelta(days=d, hours=snapshot_hour)
        sub = sub[
            (sub["snapshot_time"] >= sub["start_time"])
            & (sub["snapshot_time"] < sub["arrival_time"])
        ]
        if not sub.empty:
            snap_dfs.append(sub[["bill_code", "snapshot_time"]])

    if snap_dfs:
        snap_df_20h = pd.concat(snap_dfs, ignore_index=True)
    else:
        snap_df_20h = pd.DataFrame(columns=["bill_code", "snapshot_time"])

    # 3. Kết hợp mốc 20h và điểm scan thực tế
    grid_df = pd.concat([scan_points, snap_df_20h], ignore_index=True)
    grid_df = grid_df.drop_duplicates(subset=["bill_code", "snapshot_time"])
    grid_df = grid_df.sort_values("snapshot_time")

    # 4. Merge as-of backward để lấy trạng thái quét mới nhất trước snapshot_time
    result = pd.merge_asof(
        grid_df,
        df,
        left_on="snapshot_time",
        right_on="io_time_source_datetime",
        by="bill_code",
        direction="backward",
    )

    result = result.dropna(subset=["io_time_source_datetime", "arrival_time"]).copy()

    # Tính biến mục tiêu: Số giờ còn lại tính từ thời điểm snapshot_time
    result["arrival_offset_hours"] = (
        result["arrival_time"] - result["snapshot_time"]
    ).dt.total_seconds() / 3600.0

    # Thời gian ngâm kho (từ lần quét gần nhất đến thời điểm snapshot)
    result["hours_since_last_scan"] = (
        result["snapshot_time"] - result["io_time_source_datetime"]
    ).dt.total_seconds() / 3600.0

    # Giờ trong ngày của snapshot_time (thập phân 0.0 - 24.0)
    st = result["snapshot_time"]
    result["snapshot_hour_numeric"] = st.dt.hour + st.dt.minute / 60.0 + st.dt.second / 3600.0

    # #6: Feature Dropout ngẫu nhiên sau khi merge: mô phỏng 30% thiếu bản ghi OUT
    result["is_missing_out"] = 0
    mask_out = result["io_status_source"] == "OUT"
    if mask_out.sum() > 0:
        drop_idx = result[mask_out].sample(frac=0.3, random_state=42).index
        result.loc[drop_idx, "io_status_source"] = "MISSING"
        result.loc[drop_idx, "io_time_source"] = np.nan
        result.loc[drop_idx, "is_missing_out"] = 1

    return result


# ════════════════════════════════════════════════════════════════════════
# BƯỚC 8b — Gộp category hiếm
# ════════════════════════════════════════════════════════════════════════

def fit_rare_categories(
    series: pd.Series,
    min_count: int = RARE_CATEGORY_MIN_COUNT,
) -> set:
    """Học keep_set từ train data: chỉ giữ các category có tần suất >= min_count."""
    freq = series.value_counts()
    return set(freq[freq >= min_count].index)


def apply_rare_categories(
    series: pd.Series,
    keep_set: set,
    other_label: str = "OTHER",
) -> pd.Series:
    """Gộp các category không nằm trong keep_set thành other_label."""
    return series.where(series.isin(keep_set), other_label)


def fit_and_apply_rare_categories(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str] | None = None,
    min_count: int = RARE_CATEGORY_MIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set]]:
    """Fit rare categories trên train, apply lên cả train, val và test."""
    if cols is None:
        cols = HIGH_CARD_COLS

    rare_keep_sets = {}
    for col in cols:
        if col not in train.columns:
            continue
        n_before = train[col].nunique()
        keep_set = fit_rare_categories(train[col], min_count)
        train[col] = apply_rare_categories(train[col], keep_set)
        if val is not None and col in val.columns:
            val[col] = apply_rare_categories(val[col], keep_set)
        if test is not None and col in test.columns:
            test[col] = apply_rare_categories(test[col], keep_set)
        rare_keep_sets[col] = keep_set
        print(f"  [rare_categories] {col}: {n_before} -> "
              f"{train[col].nunique()} categories (min_count={min_count})")

    return train, val, test, rare_keep_sets


# ════════════════════════════════════════════════════════════════════════
# Align categorical dtype giữa train/test/predict
# ════════════════════════════════════════════════════════════════════════

def align_categoricals(
    df: pd.DataFrame,
    category_mappings: dict[str, list],
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Align categorical dtype cho df dựa trên category_mappings đã learn từ train."""
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
# Hàm tổng hợp cho prediction (Inference #1, #16)
# ════════════════════════════════════════════════════════════════════════

def prepare_prediction_features(
    df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    """
    Nhận raw data dạng phẳng → trả X sẵn sàng cho model.predict().
    Đồng bộ real-time serving: nếu không có snapshot_time, tự động tính snapshot_time = io_time,
    hours_since_last_scan = 0.0, snapshot_hour_numeric = giờ quét.
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
    elif "warehouse_source" not in result.columns:
        result["warehouse_source"] = np.nan

    if "io_status" in result.columns:
        result = result.rename(columns={"io_status": "io_status_source"})
    elif "io_status_source" not in result.columns:
        result["io_status_source"] = np.nan

    if "io_time" in result.columns:
        io_time = pd.to_datetime(result["io_time"], errors="coerce")
        result["io_time_source"] = (
            io_time.dt.hour + io_time.dt.minute / 60.0 + io_time.dt.second / 3600.0
        )
    elif "io_time_source" not in result.columns:
        result["io_time_source"] = np.nan

    if "stage_depth" not in result.columns:
        result["stage_depth"] = 0

    # 3. Engineer bill features
    result = engineer_bill_features(result)

    # #1: Đồng bộ tính snapshot_hour_numeric và hours_since_last_scan cho inference
    if "snapshot_time" in result.columns:
        snap_time = pd.to_datetime(result["snapshot_time"], errors="coerce")
    elif "io_time" in result.columns:
        snap_time = pd.to_datetime(result["io_time"], errors="coerce")
    else:
        snap_time = pd.Timestamp.now()

    result["snapshot_hour_numeric"] = snap_time.dt.hour + snap_time.dt.minute / 60.0 + snap_time.dt.second / 3600.0

    if "io_time" in result.columns and "snapshot_time" in result.columns:
        io_time = pd.to_datetime(result["io_time"], errors="coerce")
        result["hours_since_last_scan"] = (snap_time - io_time).dt.total_seconds() / 3600.0
        result["hours_since_last_scan"] = result["hours_since_last_scan"].clip(lower=0.0)
    else:
        result["hours_since_last_scan"] = 0.0

    # 4. Fill missing features first with NaN
    for c in feature_cols:
        if c not in result.columns:
            result[c] = np.nan

    # 5. Apply rare categories
    for col, keep_set in rare_keep_sets.items():
        if col in result.columns:
            result[col] = apply_rare_categories(result[col], keep_set)

    # 6. Align categoricals strictly according to category_mappings from bundle
    if "io_status_source" in result.columns:
        result["io_status_source"] = result["io_status_source"].astype(str)
    if "warehouse_source" in result.columns:
        result["warehouse_source"] = result["warehouse_source"].astype(str)

    align_cols = list(category_mappings.keys())
    result = align_categoricals(result, category_mappings, cols=align_cols)

    return result
