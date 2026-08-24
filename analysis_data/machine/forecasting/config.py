# -*- coding: utf-8 -*-
"""
forecasting/config.py
=====================
Constants & hyperparameters cho pipeline Forecasting LightGBM.
Tất cả module khác import từ đây → thay đổi 1 chỗ, áp dụng mọi nơi.
"""

# ── Snapshot evaluation ─────────────────────────────────────────────────
SNAPSHOT_HOUR = 20
SNAPSHOT_MINUTE = 0

# ── Processing ───────────────────────────────────────────────────────────
CHUNK_SIZE = 5000

# ── Train/Test split ─────────────────────────────────────────────────────
TRAIN_RATIO = 0.9

# ── Test sampling ────────────────────────────────────────────────────────
SAMPLE_SIZE = 100_000

# ── Rare category ────────────────────────────────────────────────────────
RARE_CATEGORY_MIN_COUNT = 50

# ── Sample weight cho bản OUT-missing ────────────────────────────────────
MISSING_OUT_WEIGHT = 0.5

# ── Outlier threshold ────────────────────────────────────────────────────
ARRIVAL_OFFSET_MAX_HOURS = 72.0

# ── LightGBM hyperparameters ────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# ── Feature columns (41 features) ───────────────────────────────────────
FEATURE_COLS = [
    # Stage features
    "warehouse_source",
    "io_status_source",
    "io_time_source",
    # Basic categorical
    "VD_type",
    "origin_province",
    "origin_district",
    "destination_province",
    "destination_district",
    "service",
    # Creation time
    "creation_hour",
    "creation_day_of_week",
    "creation_day_of_month",
    "creation_month",
    "creation_week_of_year",
    "creation_day_of_year",
    "creation_weekend",
    # Estimated delivery
    "estimated_day_of_week",
    "estimated_day_of_month",
    "estimated_month",
    "estimated_week_of_year",
    "estimated_day_of_year",
    "estimated_weekend",
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
    "full_route",
    # Weight
    "actual_weight",
    "log_actual_weight",
    "weight_bin",
    # Interactions
    "service_VD_type",
    "service_creation_dow",
    "service_creation_hour",
    "VD_type_creation_dow",
    "route_service",
    "route_VD_type",
]

# ── Categorical features (19 features) ──────────────────────────────────
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
    "full_route",
    "weight_bin",
    "service_VD_type",
    "service_creation_dow",
    "service_creation_hour",
    "VD_type_creation_dow",
    "route_service",
    "route_VD_type",
]

# ── High cardinality columns (cần gộp rare → "OTHER") ───────────────────
HIGH_CARD_COLS = [
    "full_route",
    "district_route",
    "route_service",
    "route_VD_type",
]
