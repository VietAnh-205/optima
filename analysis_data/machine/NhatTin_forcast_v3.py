"""
final_forecasting.py
====================
Nhật Tín Forecasting — LightGBM Regression Pipeline
Gộp từ notebook NhatTin-forcast-v3-regression.ipynb
Pipeline:
  1. Đọc dữ liệu (warehouse_1A, bill, bill_schedule)
  2. Lọc bills đi qua kho 1A được chọn (selected_set)
  3. Xây dựng trajectory (hành trình)
  4. Tạo stage features (đặc trưng tại mỗi bước hành trình)
  5. Loại bỏ outlier (IQR)
  6. Feature engineering (thời gian, route, weight, interaction)
  7. Huấn luyện mô hình LightGBM Regressor
  8. Đánh giá mô hình theo snapshot hàng ngày
  9. Tính metrics tổng và theo từng kho
"""
import os
import sys
import gc
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WAREHOUSE_1A_PATH = os.path.join(DATA_DIR, "warehouse_1A.csv")
BILL_PATH = os.path.join(DATA_DIR, "bill.csv")
BILL_SCHEDULE_PATH = os.path.join(DATA_DIR, "bill_schedule.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

SNAPSHOT_HOUR = 20
SNAPSHOT_MINUTE = 0
CHUNK_SIZE = 5000
TRAIN_RATIO = 0.9
SAMPLE_SIZE = 100000
RARE_CATEGORY_MIN_COUNT = 50

FEATURE_COLS = [
    'warehouse_source', 'io_status_source', 'io_time_source',
    "VD_type",
    "origin_province",
    "origin_district",
    "destination_province",
    "destination_district",
    "service",
    "creation_hour",
    "creation_day_of_week",
    "creation_day_of_month",
    "creation_month",
    "creation_week_of_year",
    "creation_day_of_year",
    "creation_weekend",
    "estimated_day_of_week",
    "estimated_day_of_month",
    "estimated_month",
    "estimated_week_of_year",
    "estimated_day_of_year",
    "estimated_weekend",
    "estimated_delivery_days",
    "estimated_delivery_days_cat",
    "creation_delivery_dow_diff",
    "creation_delivery_dow_diff_circular",
    "same_creation_delivery_dow",
    "same_creation_delivery_month",
    "creation_hour_sin",
    "creation_hour_cos",
    "creation_dow_sin",
    "creation_dow_cos",
    "creation_month_sin",
    "creation_month_cos",
    "province_route",
    "district_route",
    "full_route",
    "actual_weight",
    "log_actual_weight",
    "weight_bin",
    "service_VD_type",
    "service_creation_dow",
    "service_creation_hour",
    "VD_type_creation_dow",
    "route_service",
    "route_VD_type",
]

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

HIGH_CARD_COLS = ["full_route", "district_route", "route_service", "route_VD_type"]


def load_data():
    print("=" * 60)
    print("1. ĐỌC DỮ LIỆU (1 lần duy nhất)")
    print("=" * 60)

    wh1a = pd.read_csv(WAREHOUSE_1A_PATH)
    set_1a = set(wh1a['name'].dropna().str.strip())
    warehouse_list = sorted(set_1a)

    bill_full = pd.read_csv(BILL_PATH)
    bill_full["bill_code"] = bill_full["bill_code"].astype(str)

    chunks = []
    for chunk in pd.read_csv(BILL_SCHEDULE_PATH, chunksize=500_000):
        chunk['io_time'] = pd.to_datetime(chunk['io_time'])
        chunk['bill_code'] = chunk['bill_code'].astype(str)
        chunks.append(chunk)
    sche = pd.concat(chunks, ignore_index=True)
    sche = sche.sort_values(['bill_code', 'io_time']).reset_index(drop=True)

    print(f"  warehouse_1A: {len(warehouse_list)} kho")
    print(f"  bill shape: {bill_full.shape}")
    print(f"  bill_schedule shape: {sche.shape}")

    return warehouse_list, set_1a, bill_full, sche


def filter_bills_for_warehouse(sche_raw, selected_set, set_1a):
    sche = sche_raw.copy()

    valid_bills = sche.loc[
        sche['warehouse_name'].isin(selected_set),
        'bill_code'
    ].unique()

    sche = sche[sche['bill_code'].isin(valid_bills)].copy()

    sche['row_id'] = sche.groupby('bill_code').cumcount()

    first_selected = (
        sche[sche['warehouse_name'].isin(selected_set)]
        .groupby('bill_code')['row_id']
        .min()
        .rename('first_selected_row')
    )

    sche = sche.merge(first_selected, on='bill_code', how='left')

    sche = sche[
        sche['row_id'] <= sche['first_selected_row']
    ].copy()

    sche = sche.drop(columns=['row_id', 'first_selected_row'])

    set_1a_without_selected = set_1a - selected_set

    bills_to_remove = sche.loc[
        sche['warehouse_name'].isin(set_1a_without_selected),
        'bill_code'
    ].unique()

    sche = sche[~sche['bill_code'].isin(bills_to_remove)].copy()

    sche = sche.sort_values(["bill_code", "io_time"]).copy()

    return sche


def build_trajectory(sche, destination):
    trajectory_df = (
        sche.groupby("bill_code")
            .agg({
                "warehouse_name": list,
                "io_status": list,
                "io_time": list
            })
            .reset_index()
    )

    trajectory_df["trajectory"] = trajectory_df.apply(
        lambda r: list(zip(r.warehouse_name, r.io_status)),
        axis=1
    )

    arrival = (
        sche[
            (sche.warehouse_name == destination) &
            (sche.io_status == "IN")
        ]
        .groupby("bill_code")
        .first()["io_time"]
    )

    trajectory_df["arrival_time"] = trajectory_df.bill_code.map(arrival)
    trajectory_df = trajectory_df.dropna(subset=["arrival_time"])

    trajectory_df["arrival_sec"] = (
        trajectory_df.arrival_time.astype("int64") // 10**9
    )

    trajectory_df = trajectory_df[
        trajectory_df["io_status"].apply(len) >= 2
    ].copy()

    trajectory_df["warehouse_2last"] = trajectory_df["warehouse_name"].apply(lambda x: x[-2])

    return trajectory_df


def create_stage_features_fast(trajectory_df):
    df = trajectory_df[
        ["bill_code", "warehouse_name", "io_status", "io_time", "arrival_time"]
    ].copy()

    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")

    df["warehouse_name"] = df["warehouse_name"].str[:-1]
    df["io_status"] = df["io_status"].str[:-1]
    df["io_time"] = df["io_time"].str[:-1]

    df = df.explode(
        ["warehouse_name", "io_status", "io_time"],
        ignore_index=True
    )
    df = df.dropna(subset=["io_status"])

    df = df.rename(columns={
        "warehouse_name": "warehouse_source",
        "io_status": "io_status_source"
    })

    df["io_time_source_datetime"] = pd.to_datetime(df["io_time"], errors="coerce")

    t = df["io_time_source_datetime"]
    df["io_time_source"] = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600

    df["arrival_offset_hours"] = (
        df["arrival_time"] - df["io_time_source_datetime"].dt.normalize()
    ).dt.total_seconds() / 3600

    df["is_missing_out"] = 0
    out_rows = df[df["io_status_source"].eq("OUT")].copy()
    out_rows["io_status_source"] = -1
    out_rows["io_time_source"] = -1
    out_rows["is_missing_out"] = 1

    result = pd.concat([df, out_rows], ignore_index=True)

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


def remove_outliers(stage_features):
    df = stage_features.copy()

    df = df.dropna(
        subset=["warehouse_source", "io_status_source", "io_time_source", "arrival_offset_hours"]
    ).copy()

    group_cols = ["warehouse_source", "io_status_source"]

    Q1 = df.groupby(group_cols)["arrival_offset_hours"].transform("quantile", 0.25)
    Q3 = df.groupby(group_cols)["arrival_offset_hours"].transform("quantile", 0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["is_outlier"] = (
        (df["arrival_offset_hours"] < lower_bound)
        | (df["arrival_offset_hours"] > upper_bound)
    )

    stage_features_clean = df[~df["is_outlier"]].drop(columns="is_outlier").copy()
    stage_features_clean = stage_features_clean[stage_features_clean.arrival_offset_hours <= 72.0]

    return stage_features_clean


def feature_engineering(stage_features_clean, bill_full):
    bill_full_subset = bill_full.merge(
        stage_features_clean,
        on="bill_code",
        how="inner"
    )

    bill_full_subset["bill_creation_date"] = pd.to_datetime(
        bill_full_subset["bill_creation_date"], errors="coerce"
    )
    bill_full_subset["estimated_delivery_date"] = pd.to_datetime(
        bill_full_subset["estimated_delivery_date"], errors="coerce"
    )

    df = bill_full_subset.copy()

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

    df["estimated_day_of_week"] = df["estimated_delivery_date"].dt.dayofweek
    df["estimated_day_of_month"] = df["estimated_delivery_date"].dt.day
    df["estimated_month"] = df["estimated_delivery_date"].dt.month
    df["estimated_week_of_year"] = (
        df["estimated_delivery_date"].dt.isocalendar().week.astype("float")
    )
    df["estimated_day_of_year"] = df["estimated_delivery_date"].dt.dayofyear
    df["estimated_weekend"] = (df["estimated_day_of_week"] >= 5).astype(int)

    df["estimated_delivery_days"] = (
        df["estimated_delivery_date"] - df["bill_creation_date"]
    ).dt.total_seconds() / (24 * 3600)

    df["estimated_delivery_days_cat"] = pd.cut(
        df["estimated_delivery_days"],
        bins=[-np.inf, 0, 1, 2, 3, 5, 7, 14, np.inf],
        labels=["<=0", "0-1", "1-2", "2-3", "3-5", "5-7", "7-14", ">14"]
    )

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

    df["creation_hour_sin"] = np.sin(2 * np.pi * df["creation_hour"] / 24)
    df["creation_hour_cos"] = np.cos(2 * np.pi * df["creation_hour"] / 24)
    df["creation_dow_sin"] = np.sin(2 * np.pi * df["creation_day_of_week"] / 7)
    df["creation_dow_cos"] = np.cos(2 * np.pi * df["creation_day_of_week"] / 7)
    df["creation_month_sin"] = np.sin(2 * np.pi * df["creation_month"] / 12)
    df["creation_month_cos"] = np.cos(2 * np.pi * df["creation_month"] / 12)

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

    df["actual_weight"] = pd.to_numeric(df["actual_weight"], errors="coerce")
    df["actual_weight"] = df["actual_weight"].clip(lower=0)
    df["log_actual_weight"] = np.log1p(df["actual_weight"])
    df["weight_bin"] = pd.cut(
        df["actual_weight"],
        bins=[-np.inf, 0.5, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["<=0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">50"]
    )

    df["service_VD_type"] = df["service"].astype(str) + "_" + df["VD_type"].astype(str)
    df["service_creation_dow"] = df["service"].astype(str) + "_" + df["creation_day_of_week"].astype(str)
    df["service_creation_hour"] = df["service"].astype(str) + "_" + df["creation_hour"].round().astype(str)
    df["VD_type_creation_dow"] = df["VD_type"].astype(str) + "_" + df["creation_day_of_week"].astype(str)
    df["route_service"] = df["province_route"].astype(str) + "_" + df["service"].astype(str)
    df["route_VD_type"] = df["province_route"].astype(str) + "_" + df["VD_type"].astype(str)

    df = df.sort_values("arrival_time").reset_index(drop=True)
    df = df.dropna(subset=["arrival_offset_hours"]).copy()

    return df


def fit_rare_category_keep_set(series, min_count):
    freq = series.value_counts()
    return set(freq[freq >= min_count].index)


def apply_rare_category_map(series, keep_set, other_label="OTHER"):
    return series.where(series.isin(keep_set), other_label)


def split_and_train(df, destination, output_dir):
    bill_arrival = df.groupby("bill_code")["arrival_time"].first().sort_values()
    bill_split_idx = int(len(bill_arrival) * TRAIN_RATIO)
    train_bills = set(bill_arrival.index[:bill_split_idx])
    test_bills = set(bill_arrival.index[bill_split_idx:])

    train = df[df["bill_code"].isin(train_bills)].copy()
    test = df[df["bill_code"].isin(test_bills)].copy()

    print(f"    Total : {len(df)}")
    print(f"    Train : {len(train)}")
    print(f"    Test  : {len(test)}")

    for col in HIGH_CARD_COLS:
        keep_set = fit_rare_category_keep_set(train[col], RARE_CATEGORY_MIN_COUNT)
        train[col] = apply_rare_category_map(train[col], keep_set)
        test[col] = apply_rare_category_map(test[col], keep_set)

    for col in CATEGORICAL_FEATURES:
        train[col] = train[col].astype("category")
        test[col] = pd.Categorical(
            test[col],
            categories=train[col].cat.categories
        )

    X_train = train[FEATURE_COLS]
    y_train = train["arrival_offset_hours"]

    MISSING_OUT_WEIGHT = 0.5
    sample_weight = np.where(train["is_missing_out"] == 1, MISSING_OUT_WEIGHT, 1.0)

    model = LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    print(f"    Model training complete!")

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    })
    feature_importance["gain_pct"] = (
        feature_importance["gain"] / feature_importance["gain"].sum() * 100
    )
    feature_importance = feature_importance.sort_values(
        "gain", ascending=False
    ).reset_index(drop=True)

    top_n = 30
    fi_top = feature_importance.head(top_n).sort_values("gain", ascending=True)

    plt.figure(figsize=(10, 10))
    plt.barh(fi_top["feature"], fi_top["gain"])
    plt.xlabel("Feature importance (gain)")
    plt.ylabel("Feature")
    plt.title(f"LightGBM Feature Importance — {destination}")
    plt.tight_layout()

    slug = destination.replace(" ", "_")
    fi_path = os.path.join(output_dir, f"feature_importance_{slug}.png")
    plt.savefig(fi_path, dpi=100)
    plt.close()
    print(f"    Feature importance saved: {fi_path}")

    return model, train, test


def evaluate_snapshot(model, train, test, trajectory_df, destination, output_dir):
    test = test.copy()
    test["bill_code"] = test["bill_code"].astype(str)
    test["arrival_time"] = pd.to_datetime(test["arrival_time"], errors="coerce")

    if int(len(test["bill_code"].unique())) >= SAMPLE_SIZE:
        random_bill_codes = np.random.choice(
            test["bill_code"].unique(),
            size=SAMPLE_SIZE,
            replace=False,
        )
        test = test[test["bill_code"].isin(random_bill_codes)].copy()
        trajectory_df_sample = trajectory_df[
            trajectory_df["bill_code"].isin(random_bill_codes)
        ].copy()
    else:
        trajectory_df_sample = trajectory_df.copy()

    traj = trajectory_df_sample[
        ["bill_code", "warehouse_name", "io_status", "io_time", "arrival_time"]
    ].copy()

    traj["bill_code"] = traj["bill_code"].astype(str)

    test_bill_codes = set(test["bill_code"].dropna().unique())
    traj = traj[traj["bill_code"].isin(test_bill_codes)].copy()

    traj["arrival_time"] = pd.to_datetime(traj["arrival_time"], errors="coerce")

    traj["warehouse_name"] = traj["warehouse_name"].str[:-1]
    traj["io_status"] = traj["io_status"].str[:-1]
    traj["io_time"] = traj["io_time"].str[:-1]

    traj = traj.explode(
        ["warehouse_name", "io_status", "io_time"],
        ignore_index=True,
    )
    traj = traj.dropna(subset=["warehouse_name", "io_status", "io_time"]).copy()

    traj["io_time"] = pd.to_datetime(traj["io_time"], errors="coerce")
    traj = traj.dropna(subset=["io_time"]).copy()

    traj = traj.sort_values(["io_time", "bill_code"]).reset_index(drop=True)
    traj = traj[["bill_code", "warehouse_name", "io_status", "io_time", "arrival_time"]].copy()

    if len(traj) == 0:
        print(f"    No trajectory data for evaluation")
        return None

    TEST_START_DATE = traj["io_time"].min().normalize()
    LATEST_DATE = traj["arrival_time"].max().normalize()

    evaluation_dates = pd.date_range(start=TEST_START_DATE, end=LATEST_DATE, freq="D")

    snapshot_dates = pd.DataFrame({"snapshot_date": evaluation_dates})
    snapshot_dates["snapshot_time"] = (
        snapshot_dates["snapshot_date"]
        + pd.Timedelta(hours=SNAPSHOT_HOUR)
        + pd.Timedelta(minutes=SNAPSHOT_MINUTE)
    )

    bill_features = test.copy()
    bill_features["bill_code"] = bill_features["bill_code"].astype(str)

    if bill_features["bill_code"].duplicated().any():
        bill_features = bill_features.drop_duplicates("bill_code", keep="last").copy()

    columns_to_drop = [
        "warehouse_source",
        "io_status_source",
        "io_time_source",
        "arrival_offset_hours",
        "arrival_time",
    ]
    bill_features = bill_features.drop(
        columns=[c for c in columns_to_drop if c in bill_features.columns]
    )

    test_bill_codes_array = bill_features["bill_code"].drop_duplicates().values
    n_bills = len(test_bill_codes_array)

    all_results = []

    for start in range(0, n_bills, CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, n_bills)
        chunk_codes = test_bill_codes_array[start:end]

        traj_chunk = traj[traj["bill_code"].isin(chunk_codes)].copy()

        if len(traj_chunk) == 0:
            del traj_chunk
            gc.collect()
            continue

        bill_days = pd.DataFrame({
            "bill_code": np.repeat(chunk_codes, len(snapshot_dates)),
            "snapshot_date": np.tile(snapshot_dates["snapshot_date"].values, len(chunk_codes)),
            "snapshot_time": np.tile(snapshot_dates["snapshot_time"].values, len(chunk_codes)),
        })

        bill_days = bill_days.sort_values(
            ["snapshot_time", "bill_code"]
        ).reset_index(drop=True)

        traj_chunk = traj_chunk.sort_values(
            ["io_time", "bill_code"]
        ).reset_index(drop=True)

        current_stage = pd.merge_asof(
            bill_days,
            traj_chunk,
            by="bill_code",
            left_on="snapshot_time",
            right_on="io_time",
            direction="backward",
        )

        current_stage = current_stage[
            current_stage["arrival_time"] > current_stage["snapshot_time"]
        ].copy()

        current_stage = current_stage[
            current_stage["io_time"].notna()
        ].copy()

        if len(current_stage) == 0:
            del traj_chunk, bill_days, current_stage
            gc.collect()
            continue

        current_stage = current_stage.rename(columns={
            "warehouse_name": "warehouse_source",
            "io_status": "io_status_source",
        })

        t = current_stage["io_time"]
        current_stage["io_time_source"] = (
            t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
        )

        current_stage["arrival_offset_hours"] = (
            current_stage["arrival_time"]
            - current_stage["io_time"].dt.normalize()
        ).dt.total_seconds() / 3600

        bill_features_chunk = bill_features[
            bill_features["bill_code"].isin(chunk_codes)
        ].copy()

        eval_chunk = current_stage.merge(
            bill_features_chunk,
            on="bill_code",
            how="left",
        )

        eval_chunk["io_status_source"] = eval_chunk["io_status_source"].astype(str)
        eval_chunk["warehouse_source"] = eval_chunk["warehouse_source"].astype(str)

        for col in CATEGORICAL_FEATURES:
            if col in eval_chunk.columns:
                eval_chunk[col] = pd.Categorical(
                    eval_chunk[col],
                    categories=train[col].cat.categories,
                )

        missing_features = [
            col for col in FEATURE_COLS if col not in eval_chunk.columns
        ]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        X_chunk = eval_chunk[FEATURE_COLS]
        eval_chunk["y_pred"] = model.predict(X_chunk)

        result_chunk = eval_chunk[[
            "bill_code",
            "snapshot_date",
            "snapshot_time",
            "warehouse_source",
            "io_status_source",
            "arrival_offset_hours",
            "y_pred",
        ]].copy()

        all_results.append(result_chunk)

        del traj_chunk, bill_days, current_stage, eval_chunk, X_chunk, result_chunk, bill_features_chunk
        gc.collect()

    if len(all_results) > 0:
        eval_df = pd.concat(all_results, ignore_index=True)
    else:
        eval_df = pd.DataFrame(
            columns=[
                "bill_code", "snapshot_date", "snapshot_time",
                "warehouse_source", "io_status_source",
                "arrival_offset_hours", "y_pred",
            ]
        )

    del all_results
    gc.collect()

    if len(eval_df) == 0:
        print(f"    No evaluation data for {destination}")
        return None

    y_true = eval_df["arrival_offset_hours"]
    y_pred = eval_df["y_pred"]

    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"    Snapshot eval — MAE: {mae:.4f}h, MedAE: {medae:.4f}h, R²: {r2:.4f}")

    return {"mae": mae, "medae": medae, "r2": r2, "eval_rows": len(eval_df)}


def train_warehouse(destination, sche_raw, set_1a, bill_full, output_dir):
    t0 = time.time()
    selected_set = {destination}

    print(f"\n{'=' * 60}")
    print(f"  KHO: {destination}")
    print(f"{'=' * 60}")

    try:
        sche = filter_bills_for_warehouse(sche_raw, selected_set, set_1a)
        print(f"    sche_filtered shape: {sche.shape}")

        if len(sche) == 0:
            return {"warehouse": destination, "status": "NO_DATA", "samples": 0,
                    "test_mae": "-", "test_medae": "-", "test_r2": "-",
                    "duration_s": round(time.time() - t0, 1)}

        trajectory_df = build_trajectory(sche, destination)
        print(f"    trajectory_df shape: {trajectory_df.shape}")

        if len(trajectory_df) < 100:
            return {"warehouse": destination, "status": "INSUFFICIENT_DATA",
                    "samples": len(trajectory_df),
                    "test_mae": "-", "test_medae": "-", "test_r2": "-",
                    "duration_s": round(time.time() - t0, 1)}

        stage_features = create_stage_features_fast(trajectory_df)
        stage_features["io_status_source"] = stage_features["io_status_source"].astype(str)
        print(f"    stage_features shape: {stage_features.shape}")

        stage_features_clean = remove_outliers(stage_features)
        print(f"    after outlier removal: {stage_features_clean.shape}")

        df = feature_engineering(stage_features_clean, bill_full)
        print(f"    engineered df shape: {df.shape}")

        if len(df) < 100:
            return {"warehouse": destination, "status": "INSUFFICIENT_DATA",
                    "samples": len(df),
                    "test_mae": "-", "test_medae": "-", "test_r2": "-",
                    "duration_s": round(time.time() - t0, 1)}

        model, train, test = split_and_train(df, destination, output_dir)

        metrics = evaluate_snapshot(model, train, test, trajectory_df, destination, output_dir)

        del model, train, test, df, stage_features, stage_features_clean, trajectory_df, sche
        gc.collect()

        duration = round(time.time() - t0, 1)

        if metrics is None:
            return {"warehouse": destination, "status": "NO_EVAL_DATA",
                    "samples": 0,
                    "test_mae": "-", "test_medae": "-", "test_r2": "-",
                    "duration_s": duration}

        return {
            "warehouse": destination,
            "status": "SUCCESS",
            "samples": metrics["eval_rows"],
            "test_mae": round(metrics["mae"], 2),
            "test_medae": round(metrics["medae"], 2),
            "test_r2": round(metrics["r2"], 4),
            "duration_s": duration,
        }

    except Exception as e:
        duration = round(time.time() - t0, 1)
        error_msg = str(e)[:80]
        print(f"    ERROR: {error_msg}")
        return {
            "warehouse": destination,
            "status": f"ERROR: {error_msg}",
            "samples": "-",
            "test_mae": "-",
            "test_medae": "-",
            "test_r2": "-",
            "duration_s": duration,
        }


def generate_comparison(summary_v3_path, output_dir):
    flow_v2_path = os.path.join(DATA_DIR, "analysis_data", "flow_v2", "saved_models", "training_summary.csv")

    if not os.path.exists(flow_v2_path):
        print(f"  Flow V2 summary not found: {flow_v2_path}")
        return

    v3 = pd.read_csv(summary_v3_path)
    v2 = pd.read_csv(flow_v2_path)

    v3_cols = {
        "warehouse": "warehouse",
        "status": "v3_status",
        "samples": "v3_samples",
        "test_mae": "v3_mae",
        "test_medae": "v3_medae",
        "test_r2": "v3_r2",
        "duration_s": "v3_duration_s",
    }
    v3_renamed = v3.rename(columns=v3_cols)

    v2_cols = {
        "warehouse": "warehouse",
        "status": "v2_status",
        "samples": "v2_samples",
        "test_mae": "v2_mae",
        "test_medae": "v2_medae",
        "test_r2": "v2_r2",
        "duration_s": "v2_duration_s",
    }
    v2_renamed = v2.rename(columns=v2_cols)

    comparison = v3_renamed.merge(v2_renamed, on="warehouse", how="outer")

    for col in ["v3_mae", "v3_medae", "v3_r2", "v2_mae", "v2_medae", "v2_r2"]:
        comparison[col] = pd.to_numeric(comparison[col], errors="coerce")

    comparison["mae_diff"] = comparison["v3_mae"] - comparison["v2_mae"]
    comparison["medae_diff"] = comparison["v3_medae"] - comparison["v2_medae"]
    comparison["r2_diff"] = comparison["v3_r2"] - comparison["v2_r2"]

    comparison_path = os.path.join(output_dir, "comparison_v3_vs_flow_v2.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"\n  Comparison saved: {comparison_path}")

    print("\n" + "=" * 100)
    print("  SO SÁNH V3 vs Flow V2")
    print("=" * 100)

    display_cols = [
        "warehouse",
        "v3_status", "v3_mae", "v3_medae", "v3_r2",
        "v2_status", "v2_mae", "v2_medae", "v2_r2",
        "mae_diff", "r2_diff",
    ]
    available_cols = [c for c in display_cols if c in comparison.columns]
    print(comparison[available_cols].to_string(index=False))

    both_success = comparison[
        (comparison["v3_status"] == "SUCCESS") &
        (comparison["v2_status"] == "SUCCESS")
    ]

    if len(both_success) > 0:
        v3_wins_mae = (both_success["mae_diff"] < 0).sum()
        v2_wins_mae = (both_success["mae_diff"] > 0).sum()
        tie_mae = (both_success["mae_diff"] == 0).sum()

        print(f"\n  Số kho cả hai đều SUCCESS: {len(both_success)}")
        print(f"  V3 thắng MAE: {v3_wins_mae} | V2 thắng MAE: {v2_wins_mae} | Hòa: {tie_mae}")
        print(f"  V3 avg MAE: {both_success['v3_mae'].mean():.2f} | V2 avg MAE: {both_success['v2_mae'].mean():.2f}")
        print(f"  V3 avg R²:  {both_success['v3_r2'].mean():.4f} | V2 avg R²:  {both_success['v2_r2'].mean():.4f}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="NhatTin V3 Forecasting — Train all warehouses")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Train all 26 warehouses")
    group.add_argument("--warehouse", type=str, help="Train a specific warehouse by name")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip warehouses already in training_summary_v3.csv")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    warehouse_list, set_1a, bill_full, sche_raw = load_data()

    if args.all:
        targets = warehouse_list
    else:
        targets = [args.warehouse]
        if args.warehouse not in set_1a:
            print(f"WARNING: '{args.warehouse}' not found in warehouse_1A.csv")
            print(f"Available: {sorted(set_1a)}")
            sys.exit(1)

    summary_path = os.path.join(OUTPUT_DIR, "training_summary_v3.csv")
    existing_warehouses = set()

    if args.skip_existing and os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        existing_warehouses = set(existing_df["warehouse"].values)
        print(f"\n  Skip existing: {len(existing_warehouses)} warehouses already trained")

    results = []

    for i, destination in enumerate(targets, 1):
        if destination in existing_warehouses:
            print(f"\n  [{i}/{len(targets)}] SKIP (already exists): {destination}")
            continue

        print(f"\n  [{i}/{len(targets)}] Training: {destination}")
        result = train_warehouse(destination, sche_raw, set_1a, bill_full, OUTPUT_DIR)
        results.append(result)

        result_df = pd.DataFrame([result])
        if os.path.exists(summary_path):
            result_df.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            result_df.to_csv(summary_path, index=False)

        print(f"    → {result['status']} | MAE={result['test_mae']} | "
              f"MedAE={result['test_medae']} | R²={result['test_r2']} | "
              f"Duration={result['duration_s']}s")

    if len(results) > 0:
        print("\n" + "=" * 60)
        print("  TRAINING SUMMARY (this run)")
        print("=" * 60)
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))

    if args.all:
        generate_comparison(summary_path, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()