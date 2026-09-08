# -*- coding: utf-8 -*-
"""
forecasting/config.py
=====================
Constants & hyperparameters cho pipeline Forecasting LightGBM.
Tất cả module khác import từ đây → thay đổi 1 chỗ, áp dụng mọi nơi.
"""
import math

# ── Snapshot evaluation ─────────────────────────────────────────────────
SNAPSHOT_HOURS = [20]
SNAPSHOT_HOUR = 20
SNAPSHOT_MINUTE = 0

# ── Processing ───────────────────────────────────────────────────────────
CHUNK_SIZE = 500_000

# ── Train / Val / Test split ─────────────────────────────────────────────
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10   # = 1.0 - TRAIN_SPLIT - VAL_SPLIT

# ── Rare category ────────────────────────────────────────────────────────
RARE_CATEGORY_MIN_COUNT = 50

# ── Multi-modal threshold ────────────────────────────────────────────────
MIN_SAMPLES_MULTIMODAL = 200

# ── Outlier threshold ────────────────────────────────────────────────────
ARRIVAL_OFFSET_MAX_HOURS = 72.0

# ── LightGBM hyperparameters ────────────────────────────────────────────
def lgbm_spec(n_samples: int) -> dict:
    """Trả về bộ tham số co giãn theo quy mô dữ liệu của từng phân khúc."""
    n = max(int(n_samples), 1)
    
    depth = max(4, min(10, int(math.log2(max(n, 2)))))
    leaves = max(15, min(63, int(n / 100)))
    leaf = max(1, min(50, int(n * 0.02)))  # Tỷ lệ theo n, tránh ép sàn 10 khi n nhỏ
    lr = 0.05 if n < 5000 else 0.03
    n_est = 300 if n < 5000 else 800

    return {
        "objective": "regression",
        "n_estimators": n_est,
        "learning_rate": lr,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_child_samples": leaf,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

# ── Feature columns ─────────────────────────────────────────────────────
FEATURE_COLS = [
    # Stage features
    "warehouse_source",
    "io_status_source",
    "io_time_source",
    "snapshot_hour_numeric",  # Giờ lúc lấy snapshot (0-24h)
    "hours_since_last_scan",  # Thời gian ngâm kho tính đến lúc snapshot
    "stage_depth",            # Chặng thứ mấy của bill
    # Basic categorical
    "VD_type",
    "origin_province",
    "origin_district",
    "destination_province",
    "destination_district",
    "service",
    # Estimated delivery
    "estimated_delivery_days",
    "estimated_delivery_days_cat",
    # Date relationships
    "creation_delivery_dow_diff",
    "creation_delivery_dow_diff_circular",
    "same_creation_delivery_dow",
    "same_creation_delivery_month",
    # Cyclic features
    "creation_hour_sin",
    "creation_hour_cos",
    "creation_dow_sin",
    "creation_dow_cos",
    "creation_month_sin",
    "creation_month_cos",
    # Route
    "province_route",
    "district_route",
    # Weight
    "actual_weight",
    "log_actual_weight",
    "weight_bin",
    # Interactions
    "service_VD_type",
]

# ── Categorical features ────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "warehouse_source",
    "io_status_source",
    "VD_type",
    "origin_province",
    "origin_district",
    "destination_province",
    "destination_district",
    "service",
    "estimated_delivery_days_cat",
    "province_route",
    "district_route",
    "weight_bin",
    "service_VD_type",
]

# ── High cardinality columns (cần gộp rare → "OTHER") ───────────────────
HIGH_CARD_COLS = [
    "province_route",
    "district_route",
    "origin_district",
    "destination_district",
    "service_VD_type",
]
