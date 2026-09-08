import numpy as np
import pandas as pd

FEATURE_COLS = [
    "warehouse_source",
    "io_status_source",
    "io_time_source",
    "snapshot_hour_numeric",
    "hours_since_last_scan",
    "stage_depth",
    "VD_type",
    "origin_province",
    "origin_district",
    "destination_province",
    "destination_district",
    "service",
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
    "actual_weight",
    "log_actual_weight",
    "weight_bin",
    "service_VD_type",
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
    "weight_bin",
    "service_VD_type",
]

HIGH_CARD_COLS = [
    "province_route",
    "district_route",
    "origin_district",
    "destination_district",
    "service_VD_type",
]

RARE_CATEGORY_MIN_COUNT = 50
ARRIVAL_OFFSET_MAX_HOURS = 72.0


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "bill_creation_date" in df.columns and df["bill_creation_date"].notna().any():
        c = pd.to_datetime(df["bill_creation_date"], errors="coerce")
        df["creation_hour"] = c.dt.hour + c.dt.minute / 60.0
        df["creation_day_of_week"] = c.dt.dayofweek
        df["creation_month"] = c.dt.month
        df["creation_day"] = c.dt.day
    else:
        df["creation_hour"] = np.nan
        df["creation_day_of_week"] = np.nan
        df["creation_month"] = np.nan
        df["creation_day"] = np.nan

    if "estimated_delivery_date" in df.columns and df["estimated_delivery_date"].notna().any():
        e = pd.to_datetime(df["estimated_delivery_date"], errors="coerce")
        df["estimated_day_of_week"] = e.dt.dayofweek
        df["estimated_month"] = e.dt.month
        df["estimated_day"] = e.dt.day
    else:
        df["estimated_day_of_week"] = np.nan
        df["estimated_month"] = np.nan
        df["estimated_day"] = np.nan

    if "estimated_delivery_date" in df.columns and "bill_creation_date" in df.columns:
        e = pd.to_datetime(df["estimated_delivery_date"], errors="coerce")
        c = pd.to_datetime(df["bill_creation_date"], errors="coerce")
        df["estimated_delivery_days"] = (e - c).dt.total_seconds() / (24 * 3600)
    else:
        df["estimated_delivery_days"] = np.nan

    df["estimated_delivery_days_cat"] = pd.cut(
        df["estimated_delivery_days"],
        bins=[-np.inf, 0, 1, 2, 3, 5, 7, np.inf],
        labels=["<=0", "1", "2", "3", "4-5", "6-7", ">7"],
    )

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

    df["creation_hour_sin"] = np.sin(2 * np.pi * df["creation_hour"] / 24)
    df["creation_hour_cos"] = np.cos(2 * np.pi * df["creation_hour"] / 24)
    df["creation_dow_sin"] = np.sin(2 * np.pi * df["creation_day_of_week"] / 7)
    df["creation_dow_cos"] = np.cos(2 * np.pi * df["creation_day_of_week"] / 7)
    df["creation_month_sin"] = np.sin(2 * np.pi * df["creation_month"] / 12)
    df["creation_month_cos"] = np.cos(2 * np.pi * df["creation_month"] / 12)

    op = (
        df["origin_province"].fillna("UNKNOWN").astype(str)
        if "origin_province" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )
    dp = (
        df["destination_province"].fillna("UNKNOWN").astype(str)
        if "destination_province" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )
    od = (
        df["origin_district"].fillna("UNKNOWN").astype(str)
        if "origin_district" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )
    dd = (
        df["destination_district"].fillna("UNKNOWN").astype(str)
        if "destination_district" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )

    df["province_route"] = op + " -> " + dp
    df["district_route"] = od + " -> " + dd

    w = pd.to_numeric(df.get("actual_weight", np.nan), errors="coerce").clip(lower=0)
    df["actual_weight"] = w
    df["log_actual_weight"] = np.log1p(w)
    df["weight_bin"] = pd.cut(
        w,
        bins=[-np.inf, 0.5, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["<=0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">50"],
    )

    svc = (
        df["service"].fillna("UNKNOWN").astype(str)
        if "service" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )
    vd = (
        df["VD_type"].fillna("UNKNOWN").astype(str)
        if "VD_type" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )
    df["service_VD_type"] = svc + "_" + vd

    return df


def fit_rare_categories(series: pd.Series, min_count: int = RARE_CATEGORY_MIN_COUNT) -> set:
    freq = series.value_counts()
    return set(freq[freq >= min_count].index)


def apply_rare_categories(series: pd.Series, keep_set: set, other_label: str = "OTHER") -> pd.Series:
    return series.where(series.isin(keep_set), other_label)


def fit_and_apply_rare_categories(
    train: pd.DataFrame,
    val: pd.DataFrame = None,
    test: pd.DataFrame = None,
    cols: list[str] = None,
    min_count: int = RARE_CATEGORY_MIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set]]:
    if cols is None:
        cols = HIGH_CARD_COLS

    rare_keep_sets = {}
    for col in cols:
        if col not in train.columns:
            continue
        keep_set = fit_rare_categories(train[col], min_count=min_count)
        train[col] = apply_rare_categories(train[col], keep_set)
        if val is not None and col in val.columns:
            val[col] = apply_rare_categories(val[col], keep_set)
        if test is not None and col in test.columns:
            test[col] = apply_rare_categories(test[col], keep_set)
        rare_keep_sets[col] = keep_set

    return train, val, test, rare_keep_sets


def prepare_training_dataset(
    bill_schedule_df: pd.DataFrame,
    bill_df: pd.DataFrame,
    warehouse_1a_df: pd.DataFrame,
    target_warehouse: str,
    snapshot_hour: int = 20,
) -> pd.DataFrame:
    dest = target_warehouse
    set_1a = set(warehouse_1a_df["name"].dropna().str.strip())

    sche = bill_schedule_df.copy()
    sche["io_time"] = pd.to_datetime(sche["io_time"])
    sche = sche.sort_values(["bill_code", "io_time"]).reset_index(drop=True)

    arrival_mask = (sche["warehouse_name"] == dest) & (sche["io_status"] == "IN")
    valid_bills = sche.loc[arrival_mask, "bill_code"].unique()
    if len(valid_bills) == 0:
        raise ValueError(f"Không tìm thấy bill nào có check-in 'IN' tại kho '{dest}'.")

    sche = sche[sche["bill_code"].isin(valid_bills)].copy()
    sche["row_id"] = sche.groupby("bill_code").cumcount()

    arrival_rows = sche[(sche["warehouse_name"] == dest) & (sche["io_status"] == "IN")]
    first_selected = arrival_rows.groupby("bill_code")["row_id"].min().rename("first_row")
    sche = sche.merge(first_selected, on="bill_code", how="inner")

    sche_filtered = sche[sche["row_id"] <= sche["first_row"]].drop(columns=["row_id", "first_row"]).copy()

    other_1a = set_1a - {dest}
    bad_bills = sche_filtered.loc[sche_filtered["warehouse_name"].isin(other_1a), "bill_code"].unique()
    sche_filtered = sche_filtered[~sche_filtered["bill_code"].isin(bad_bills)].copy()

    if sche_filtered.empty:
        raise ValueError(f"Sau khi lọc, không còn bill nào hợp lệ cho kho '{dest}'.")

    traj = (
        sche_filtered.groupby("bill_code")
        .agg({
            "warehouse_name": list,
            "io_status": list,
            "io_time": list,
        })
        .reset_index()
    )

    arrival_map = arrival_rows.groupby("bill_code")["io_time"].first()
    traj["arrival_time"] = traj["bill_code"].map(arrival_map)
    traj = traj[traj["io_status"].apply(len) >= 2].copy()

    traj["warehouse_name"] = traj["warehouse_name"].str[:-1]
    traj["io_status"] = traj["io_status"].str[:-1]
    traj["io_time"] = traj["io_time"].str[:-1]

    stages = traj.explode(["warehouse_name", "io_status", "io_time"]).dropna(
        subset=["io_status", "warehouse_name", "io_time"]
    )

    stages["stage_depth"] = stages.groupby("bill_code").cumcount()
    stages = stages.rename(
        columns={
            "warehouse_name": "warehouse_source",
            "io_status": "io_status_source",
        }
    )

    stages["io_time_source_datetime"] = pd.to_datetime(stages["io_time"])
    t = stages["io_time_source_datetime"]
    stages["io_time_source"] = t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0

    df = bill_df.merge(stages, on="bill_code", how="inner")
    df = _engineer_features(df)

    scan_points = df[["bill_code", "io_time_source_datetime"]].rename(
        columns={"io_time_source_datetime": "snapshot_time"}
    )

    bounds = (
        df.groupby("bill_code")
        .agg(
            start_time=("io_time_source_datetime", "min"),
            arrival_time=("arrival_time", "first"),
        )
        .reset_index()
    )

    bounds["days"] = (bounds["arrival_time"].dt.normalize() - bounds["start_time"].dt.normalize()).dt.days
    max_d = int(bounds["days"].max()) if not bounds.empty and bounds["days"].max() > 0 else 0

    snap_dfs = []
    for d in range(max_d + 1):
        sub = bounds[bounds["days"] >= d].copy()
        sub["snapshot_time"] = sub["start_time"].dt.normalize() + pd.Timedelta(days=d, hours=snapshot_hour)
        sub = sub[(sub["snapshot_time"] >= sub["start_time"]) & (sub["snapshot_time"] < sub["arrival_time"])]
        if not sub.empty:
            snap_dfs.append(sub[["bill_code", "snapshot_time"]])

    snap_df_20h = pd.concat(snap_dfs, ignore_index=True) if snap_dfs else pd.DataFrame(columns=["bill_code", "snapshot_time"])

    grid_df = (
        pd.concat([scan_points, snap_df_20h], ignore_index=True)
        .drop_duplicates(subset=["bill_code", "snapshot_time"])
        .sort_values("snapshot_time")
        .reset_index(drop=True)
    )

    df = df.sort_values("io_time_source_datetime").reset_index(drop=True)

    df = pd.merge_asof(
        grid_df,
        df,
        left_on="snapshot_time",
        right_on="io_time_source_datetime",
        by="bill_code",
        direction="backward",
    )

    df = df.dropna(subset=["io_time_source_datetime", "arrival_time"]).copy()

    df["arrival_offset_hours"] = (df["arrival_time"] - df["snapshot_time"]).dt.total_seconds() / 3600.0
    df["hours_since_last_scan"] = (df["snapshot_time"] - df["io_time_source_datetime"]).dt.total_seconds() / 3600.0

    st = df["snapshot_time"]
    df["snapshot_hour_numeric"] = st.dt.hour + st.dt.minute / 60.0 + st.dt.second / 3600.0

    df["is_missing_out"] = 0
    mask_out = df["io_status_source"] == "OUT"
    if mask_out.sum() > 0:
        drop_idx = df[mask_out].sample(frac=0.3, random_state=42).index
        df.loc[drop_idx, "io_status_source"] = "MISSING"
        df.loc[drop_idx, "io_time_source"] = np.nan
        df.loc[drop_idx, "is_missing_out"] = 1

    return df


def prepare_prediction_features(
    df: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)
    category_mappings = bundle.get("category_mappings", {})
    rare_keep_sets = bundle.get("rare_keep_sets", {})

    result = df.copy()

    if "bill_code" not in result.columns:
        result["bill_code"] = [f"bill_{i}" for i in range(len(result))]

    if "warehouse_name" in result.columns:
        result["warehouse_source"] = result["warehouse_name"]
    elif "warehouse_source" not in result.columns:
        result["warehouse_source"] = np.nan

    if "io_status" in result.columns:
        result["io_status_source"] = result["io_status"]
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

    result = _engineer_features(result)

    if "snapshot_time" in result.columns and result["snapshot_time"].notna().any():
        snap_time = pd.to_datetime(result["snapshot_time"], errors="coerce")
    elif "io_time" in result.columns and result["io_time"].notna().any():
        snap_time = pd.to_datetime(result["io_time"], errors="coerce")
    else:
        snap_time = pd.Series([pd.Timestamp.now()] * len(result), index=result.index)

    result["snapshot_hour_numeric"] = (
        snap_time.dt.hour + snap_time.dt.minute / 60.0 + snap_time.dt.second / 3600.0
    )

    if "io_time" in result.columns and "snapshot_time" in result.columns:
        io_t = pd.to_datetime(result["io_time"], errors="coerce")
        result["hours_since_last_scan"] = (snap_time - io_t).dt.total_seconds() / 3600.0
        result["hours_since_last_scan"] = result["hours_since_last_scan"].clip(lower=0.0)
    else:
        result["hours_since_last_scan"] = 0.0

    for c in feature_cols:
        if c not in result.columns:
            result[c] = np.nan

    for col, keep_set in rare_keep_sets.items():
        if col in result.columns:
            result[col] = apply_rare_categories(result[col], keep_set)

    if "io_status_source" in result.columns:
        result["io_status_source"] = result["io_status_source"].astype(str)
    if "warehouse_source" in result.columns:
        result["warehouse_source"] = result["warehouse_source"].astype(str)

    for col, cats in category_mappings.items():
        if col in result.columns:
            result[col] = pd.Categorical(result[col], categories=cats)

    return result