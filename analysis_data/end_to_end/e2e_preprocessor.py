# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_data.flow_v1.forecasting.config import (
    CHUNK_SIZE,
    FEATURE_COLS,
    CATEGORICAL_FEATURES,
    HIGH_CARD_COLS,
    RARE_CATEGORY_MIN_COUNT,
    SNAPSHOT_HOUR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    ARRIVAL_OFFSET_MAX_HOURS,
)
from analysis_data.flow_v1.forecasting.features import (
    create_stage_features,
    engineer_bill_features,
    generate_pit_snapshot,
    fit_outlier_bounds,
    apply_outlier_bounds,
    fit_and_apply_rare_categories,
    align_categoricals,
)


def load_raw_data(
    data_dir: str = None,
    bill_path: str = None,
    bill_schedule_path: str = None,
    warehouse_1a_path: str = None,
    chunksize: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data_dir is not None:
        if bill_path is None:
            bill_path = os.path.join(data_dir, "bill.csv")
        if bill_schedule_path is None:
            bill_schedule_path = os.path.join(data_dir, "bill_schedule.csv")
        if warehouse_1a_path is None:
            warehouse_1a_path = os.path.join(data_dir, "warehouse_1A.csv")

    for path, name in [
        (bill_path, "bill.csv"),
        (bill_schedule_path, "bill_schedule.csv"),
        (warehouse_1a_path, "warehouse_1A.csv"),
    ]:
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file: {path} ({name})")

    print(f"[load_raw_data] Đang đọc {warehouse_1a_path}...")
    wh1a = pd.read_csv(warehouse_1a_path)

    print(f"[load_raw_data] Đang đọc {bill_path}...")
    bill = pd.read_csv(bill_path)

    print(f"[load_raw_data] Đang đọc {bill_schedule_path} theo chunk {chunksize:,}...")
    chunks = []
    for chunk in pd.read_csv(bill_schedule_path, chunksize=chunksize):
        chunk["io_time"] = pd.to_datetime(chunk["io_time"], errors="coerce")
        chunks.append(chunk)
    sche = pd.concat(chunks, ignore_index=True)
    del chunks

    sche = sche.sort_values(["bill_code", "io_time"]).reset_index(drop=True)
    print(f"[load_raw_data] Hoàn tất: sche={sche.shape}, bill={bill.shape}, wh1a={wh1a.shape}")

    return sche, bill, wh1a


def extract_1a_journey(
    bill_schedule_df: pd.DataFrame,
    warehouse_1a_df: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[extract_1a_journey] Bắt đầu trích xuất hành trình kho 1A...")
    set_1a = set(warehouse_1a_df["name"].dropna().str.strip().unique())
    print(f"  Số kho 1A mục tiêu: {len(set_1a)}")

    sche = bill_schedule_df.copy()
    sche["warehouse_name"] = sche["warehouse_name"].astype(str).str.strip()

    mask_in_1a = (sche["warehouse_name"].isin(set_1a)) & (sche["io_status"] == "IN")
    in_1a = sche[mask_in_1a].dropna(subset=["io_time"]).sort_values(["bill_code", "io_time"]).copy()

    if in_1a.empty:
        raise ValueError("Không tìm thấy bản ghi IN nào tại các kho 1A!")

    in_1a["prev_wh"] = in_1a.groupby("bill_code")["warehouse_name"].shift(1)
    in_1a_transitions = in_1a[in_1a["warehouse_name"] != in_1a["prev_wh"]].copy()

    distinct_counts = in_1a_transitions.groupby("bill_code")["warehouse_name"].nunique()
    valid_bills = distinct_counts[distinct_counts >= 2].index

    print(f"  Tổng bill có scan IN tại kho 1A: {in_1a['bill_code'].nunique():,}")
    print(f"  Số bill đi qua >= 2 kho 1A khác nhau: {len(valid_bills):,}")

    in_valid = in_1a_transitions[in_1a_transitions["bill_code"].isin(valid_bills)].copy()

    first_rows = in_valid.groupby("bill_code").first().reset_index()
    last_rows = in_valid.groupby("bill_code").last().reset_index()
    all_wh_series = in_valid.groupby("bill_code")["warehouse_name"].apply(list)

    journey_df = pd.DataFrame({
        "bill_code": first_rows["bill_code"],
        "first_1A": first_rows["warehouse_name"],
        "first_1a_arrival_time": first_rows["io_time"],
        "last_1A": last_rows["warehouse_name"],
        "all_1a_warehouses": first_rows["bill_code"].map(all_wh_series),
        "n_distinct_1a": first_rows["bill_code"].map(distinct_counts),
    })

    diff_mask = journey_df["first_1A"] != journey_df["last_1A"]
    journey_df = journey_df[diff_mask].reset_index(drop=True)

    print(f"  Số bill hợp lệ sau khi xác nhận first_1A != last_1A: {len(journey_df):,}")
    return journey_df


def compute_time_to_first_1a(
    bill_schedule_df: pd.DataFrame,
    journey_df: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[compute_time_to_first_1a] Đang tính thời gian từ OUT kho đầu tiên đến IN kho 1A...")
    valid_bills = set(journey_df["bill_code"])
    sche = bill_schedule_df[bill_schedule_df["bill_code"].isin(valid_bills)].dropna(subset=["io_time"]).copy()
    sche = sche.sort_values(["bill_code", "io_time"]).reset_index(drop=True)

    sche["prev_wh"] = sche.groupby("bill_code")["warehouse_name"].shift(1)
    sche["wh_change"] = (sche["warehouse_name"] != sche["prev_wh"]).astype(int)
    sche["stay_id"] = sche.groupby("bill_code")["wh_change"].cumsum() - 1

    first_stay = sche[sche["stay_id"] == 0].copy()

    origin_wh = first_stay.groupby("bill_code")["warehouse_name"].first().rename("origin_warehouse")

    first_stay_out = first_stay[first_stay["io_status"] == "OUT"]
    last_out_time = first_stay_out.groupby("bill_code")["io_time"].last().rename("first_wh_out_time")

    missing_out_bills = set(first_stay["bill_code"]) - set(last_out_time.index)
    if missing_out_bills:
        fallback_time = (
            first_stay[first_stay["bill_code"].isin(missing_out_bills)]
            .groupby("bill_code")["io_time"]
            .last()
            .rename("first_wh_out_time")
        )
        last_out_time = pd.concat([last_out_time, fallback_time])

    res = journey_df.merge(origin_wh, on="bill_code", how="inner")
    res = res.merge(last_out_time, on="bill_code", how="inner")

    res["time_to_first_1a_hours"] = (
        res["first_1a_arrival_time"] - res["first_wh_out_time"]
    ).dt.total_seconds() / 3600.0

    valid_time_mask = res["time_to_first_1a_hours"] >= 0.0
    res = res[valid_time_mask].reset_index(drop=True)

    print(f"  Hoàn tất: {len(res):,} bills có thời gian hợp lệ.")
    print(f"  Phân phối time_to_first_1a_hours (từ OUT kho đầu tiên -> IN kho 1A): "
          f"Min={res['time_to_first_1a_hours'].min():.2f}h, "
          f"Median={res['time_to_first_1a_hours'].median():.2f}h, "
          f"Max={res['time_to_first_1a_hours'].max():.2f}h")

    return res


def build_e2e_training_data(
    bill_schedule_df: pd.DataFrame,
    bill_df: pd.DataFrame,
    warehouse_1a_df: pd.DataFrame,
    snapshot_hour: int = SNAPSHOT_HOUR,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("XÂY DỰNG DỮ LIỆU HUẤN LUYỆN END-TO-END")
    print("=" * 60)

    journey_df = extract_1a_journey(bill_schedule_df, warehouse_1a_df)
    journey_df = compute_time_to_first_1a(bill_schedule_df, journey_df)

    print("\n[build_e2e_training_data] Cắt trajectory trước khi đến kho 1A đầu tiên...")
    valid_bills = set(journey_df["bill_code"])
    sche = bill_schedule_df[bill_schedule_df["bill_code"].isin(valid_bills)].copy()
    sche["warehouse_name"] = sche["warehouse_name"].astype(str).str.strip()

    first_info = journey_df.set_index("bill_code")[["first_1A", "first_1a_arrival_time"]]
    sche["first_1A"] = sche["bill_code"].map(first_info["first_1A"])
    sche["first_1a_arrival_time"] = sche["bill_code"].map(first_info["first_1a_arrival_time"])

    sche["row_id"] = sche.groupby("bill_code").cumcount()

    is_first_arrival = (
        (sche["warehouse_name"] == sche["first_1A"])
        & (sche["io_status"] == "IN")
        & (sche["io_time"] == sche["first_1a_arrival_time"])
    )
    first_arr_row = (
        sche[is_first_arrival]
        .groupby("bill_code")["row_id"]
        .min()
        .rename("first_arr_row")
    )
    sche = sche.merge(first_arr_row, on="bill_code", how="inner")

    sche_filtered = sche[sche["row_id"] <= sche["first_arr_row"]].drop(
        columns=["row_id", "first_arr_row", "first_1A", "first_1a_arrival_time"]
    ).copy()

    print("  Xây dựng trajectory theo danh sách sự kiện...")
    trajectory_df = (
        sche_filtered.groupby("bill_code")
        .agg({
            "warehouse_name": list,
            "io_status": list,
            "io_time": list,
        })
        .reset_index()
    )
    arrival_map = journey_df.set_index("bill_code")["first_1a_arrival_time"]
    trajectory_df["arrival_time"] = trajectory_df["bill_code"].map(arrival_map)
    trajectory_df = trajectory_df.dropna(subset=["arrival_time"])

    trajectory_df = trajectory_df[trajectory_df["io_status"].apply(len) >= 2].copy()
    print(f"  Trajectory DF có {len(trajectory_df):,} bills đủ điều kiện.")

    print("\n[build_e2e_training_data] Tạo stage features...")
    stage_features = create_stage_features(trajectory_df)
    stage_features["io_status_source"] = stage_features["io_status_source"].astype(str)

    print("\n[build_e2e_training_data] Ghép nhãn hành trình và bill.csv...")
    labels_subset = journey_df[["bill_code", "first_1A", "last_1A", "time_to_first_1a_hours", "first_wh_out_time"]].copy()
    merged = stage_features.merge(labels_subset, on="bill_code", how="inner")

    full_df = bill_df.merge(merged, on="bill_code", how="inner")

    print("\n[build_e2e_training_data] Tạo đặc trưng nâng cao (engineer_bill_features)...")
    df = engineer_bill_features(full_df)
    df = df.sort_values("arrival_time").reset_index(drop=True)
    df = df.dropna(subset=["arrival_offset_hours"]).copy()

    print(f"\n[build_e2e_training_data] Tạo PIT Snapshot {snapshot_hour}:00 & Real-Time scans...")
    df = generate_pit_snapshot(df, snapshot_hour=snapshot_hour)

    df["time_to_first_1a_hours"] = df["arrival_offset_hours"]

    print(f"  Tổng số mẫu snapshot hoàn chỉnh: {df.shape}")
    return df


def prepare_e2e_splits(
    df: pd.DataFrame,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    print(f"\n[prepare_e2e_splits] Chia train/val/test ({train_split*100:.0f}/{val_split*100:.0f}/{(1-train_split-val_split)*100:.0f})...")
    bill_arrival = df.groupby("bill_code")["arrival_time"].first().sort_values()
    n_bills = len(bill_arrival)
    train_idx = int(n_bills * train_split)
    val_idx = int(n_bills * (train_split + val_split))

    train_bills = set(bill_arrival.index[:train_idx])
    val_bills = set(bill_arrival.index[train_idx:val_idx])
    test_bills = set(bill_arrival.index[val_idx:])

    train = df[df["bill_code"].isin(train_bills)].copy()
    val = df[df["bill_code"].isin(val_bills)].copy()
    test = df[df["bill_code"].isin(test_bills)].copy()

    print(f"  Tổng bills: {n_bills:,} | Tổng snapshots: {len(df):,}")
    print(f"  Train : {len(train_bills):,} bills ({len(train):,} snapshots)")
    print(f"  Val   : {len(val_bills):,} bills ({len(val):,} snapshots)")
    print(f"  Test  : {len(test_bills):,} bills ({len(test):,} snapshots)")

    print("\n[prepare_e2e_splits] Loại bỏ outlier...")
    lower_bound, upper_bound = fit_outlier_bounds(train)
    train = apply_outlier_bounds(train, lower_bound, upper_bound)
    val = apply_outlier_bounds(val, lower_bound, upper_bound)
    test = apply_outlier_bounds(test, lower_bound, upper_bound)

    print("\n[prepare_e2e_splits] Gộp category hiếm...")
    train, val, test, rare_keep_sets = fit_and_apply_rare_categories(
        train, val, test,
        cols=HIGH_CARD_COLS,
        min_count=RARE_CATEGORY_MIN_COUNT,
    )

    outlier_bounds = {
        "lower": lower_bound,
        "upper": upper_bound,
    }

    return train, val, test, rare_keep_sets, outlier_bounds
