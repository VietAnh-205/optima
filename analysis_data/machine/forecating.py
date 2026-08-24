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
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
# ================================================================
# CONFIGURATION
# ================================================================
selected_set = {'Kho Sóng Thần'}
# selected_set = {'Kho Văn Giang'}
OUTPUT_DIR = "output"
# Đường dẫn dữ liệu — thay đổi theo môi trường
DATA_DIR =  os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WAREHOUSE_1A_PATH = os.path.join(DATA_DIR, "warehouse_1A.csv")
BILL_PATH = os.path.join(DATA_DIR, "bill.csv")
BILL_SCHEDULE_PATH = os.path.join(DATA_DIR, "bill_schedule.csv")
# Snapshot parameters cho evaluation
SNAPSHOT_HOUR = 20
SNAPSHOT_MINUTE = 0
CHUNK_SIZE = 5000
# Train/test split ratio
TRAIN_RATIO = 0.9
# Sampling size cho test (nếu test quá lớn)
SAMPLE_SIZE = 100000
# Ngưỡng tần suất tối thiểu để giữ 1 category riêng cho các interaction/route
# feature cardinality cao (full_route, district_route, route_service, route_VD_type);
# category có tần suất thấp hơn sẽ bị gộp thành "OTHER"
RARE_CATEGORY_MIN_COUNT = 50


# ================================================================
# 1. ĐỌC DỮ LIỆU
# ================================================================

print("=" * 60)
print("1. ĐỌC DỮ LIỆU")
print("=" * 60)


def load_bill_schedule(path, chunksize=500_000):
    """Đọc bill_schedule.csv theo chunk (dữ liệu lớn), parse io_time, sort theo bill_code+io_time."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk['io_time'] = pd.to_datetime(chunk['io_time'])
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df.sort_values(['bill_code', 'io_time']).reset_index(drop=True)


wh1a = pd.read_csv(WAREHOUSE_1A_PATH)
set_1a = set(wh1a['name'].dropna().str.strip())

bill_full = pd.read_csv(BILL_PATH)

# Đọc bill_schedule đúng 1 lần, dùng lại xuyên suốt pipeline (trước đây đọc trùng 2 lần)
sche = load_bill_schedule(BILL_SCHEDULE_PATH)

print(f"  bill_schedule shape: {sche.shape}")


# ================================================================
# 3. LỌC BILLS ĐI QUA KHO 1A ĐƯỢC CHỌN
# ================================================================

print()
print("=" * 60)
print("3. LỌC BILLS ĐI QUA KHO 1A ĐƯỢC CHỌN")
print("=" * 60)

# Chỉ lấy bills phải đi qua kho 1A đã chọn
valid_bills = sche.loc[
    sche['warehouse_name'].isin(selected_set),
    'bill_code'
].unique()

sche_filtered = sche[sche['bill_code'].isin(valid_bills)].copy()

# Xóa bỏ dòng dữ liệu sau khi kho 1A được chọn xuất hiện
sche['row_id'] = sche.groupby('bill_code').cumcount()

first_selected = (
    sche[sche['warehouse_name'].isin(selected_set)]
    .groupby('bill_code')['row_id']
    .min()
    .rename('first_selected_row')
)

sche = sche.merge(first_selected, on='bill_code', how='left')

sche_filtered = sche[
    sche['row_id'] <= sche['first_selected_row']
].copy()

sche_filtered = sche_filtered.drop(columns=['row_id', 'first_selected_row'])

# Chỉ giữ bills chưa qua kho 1A nào khác
set_1a_without_selected_set = set_1a - selected_set

bills_to_remove = sche_filtered.loc[
    sche_filtered['warehouse_name'].isin(set_1a_without_selected_set),
    'bill_code'
].unique()

sche_filtered = sche_filtered[
    ~sche_filtered['bill_code'].isin(bills_to_remove)
].copy()

print(f"  sche_filtered shape: {sche_filtered.shape}")

# Sort cho trajectory
sche = sche_filtered.sort_values(["bill_code", "io_time"]).copy()


# ================================================================
# 4. XÂY DỰNG TRAJECTORY
# ================================================================

print()
print("=" * 60)
print("4. XÂY DỰNG TRAJECTORY")
print("=" * 60)

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

# Thời gian đến kho 1A
destination = next(iter(selected_set))

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

# Lọc trajectory có độ dài ít nhất 2
trajectory_df = trajectory_df[
    trajectory_df["io_status"].apply(len) >= 2
].copy()

# Kho trước kho 1A (warehouse_2last)
trajectory_df["warehouse_2last"] = trajectory_df["warehouse_name"].apply(lambda x: x[-2])
warehouse_len2 = set(trajectory_df["warehouse_2last"])

warehouse_freq = (
    trajectory_df["warehouse_2last"]
    .value_counts()
    .rename_axis("warehouse_2last")
    .reset_index(name="frequency")
)

trajectory_df_full_target = trajectory_df.copy()
trajectory_df = trajectory_df_full_target.copy()

print(f"  trajectory_df shape: {trajectory_df.shape}")
print(f"  destination: {destination}")


# ================================================================
# 5. TẠO STAGE FEATURES
# ================================================================

print()
print("=" * 60)
print("5. TẠO STAGE FEATURES")
print("=" * 60)


def create_stage_features_fast(trajectory_df):
    """
    Tạo đặc trưng tại mỗi bước hành trình của đơn.
    Mỗi lần đơn check in/out tại một kho → 1 điểm dữ liệu.
    Thêm phiên bản missing data cho OUT (io_status=-1, io_time=-1).
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
        ignore_index=True
    )
    df = df.dropna(subset=["io_status"])

    # 5. Rename source features
    df = df.rename(columns={
        "warehouse_name": "warehouse_source",
        "io_status": "io_status_source"
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
    # Lưu ý: bản OUT-missing có cùng target (arrival_offset_hours) với bản gốc,
    # nên bị đánh trọng số thấp hơn khi train (xem sample_weight ở bước train)
    # thay vì nhân đôi vô điều kiện mọi phân phối.
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


stage_features = create_stage_features_fast(trajectory_df)
stage_features["io_status_source"] = stage_features["io_status_source"].astype(str)

print(f"  stage_features shape: {stage_features.shape}")


# ================================================================
# 6. LOẠI BỎ OUTLIER (IQR)
# ================================================================

print()
print("=" * 60)
print("6. LOẠI BỎ OUTLIER (IQR)")
print("=" * 60)

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

print(f"  Original rows: {len(df)}")
print(f"  Outliers: {df['is_outlier'].sum()}")

stage_features_clean = df[~df["is_outlier"]].drop(columns="is_outlier").copy()
stage_features_clean = stage_features_clean[stage_features_clean.arrival_offset_hours <= 72.0]

print(f"  Remaining rows: {len(stage_features_clean)}")


# ================================================================
# 7. MERGE BILL INFO + FEATURE ENGINEERING
# ================================================================

print()
print("=" * 60)
print("7. FEATURE ENGINEERING")
print("=" * 60)

bill_full_subset = bill_full.merge(
    stage_features_clean,
    on="bill_code",
    how="inner"
)

print(f"  bill_full_subset shape: {bill_full_subset.shape}")

# --- Datetime conversion ---
bill_full_subset["bill_creation_date"] = pd.to_datetime(
    bill_full_subset["bill_creation_date"], errors="coerce"
)
bill_full_subset["estimated_delivery_date"] = pd.to_datetime(
    bill_full_subset["estimated_delivery_date"], errors="coerce"
)

df = bill_full_subset.copy()

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
    labels=["<=0", "0-1", "1-2", "2-3", "3-5", "5-7", "7-14", ">14"]
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
    labels=["<=0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">50"]
)

# --- 7.8 Categorical interaction features ---
df["service_VD_type"] = df["service"].astype(str) + "_" + df["VD_type"].astype(str)
df["service_creation_dow"] = df["service"].astype(str) + "_" + df["creation_day_of_week"].astype(str)
df["service_creation_hour"] = df["service"].astype(str) + "_" + df["creation_hour"].round().astype(str)
df["VD_type_creation_dow"] = df["VD_type"].astype(str) + "_" + df["creation_day_of_week"].astype(str)
df["route_service"] = df["province_route"].astype(str) + "_" + df["service"].astype(str)
df["route_VD_type"] = df["province_route"].astype(str) + "_" + df["VD_type"].astype(str)

# Sort theo thời gian
df = df.sort_values("arrival_time").reset_index(drop=True)

# Xóa rows thiếu target
df = df.dropna(subset=["arrival_offset_hours"]).copy()

print(f"  Engineered df shape: {df.shape}")


# ================================================================
# 8. CHIA TRAIN / TEST
# ================================================================

print()
print("=" * 60)
print("8. CHIA TRAIN / TEST")
print("=" * 60)

# Chia theo bill_code (không chia theo dòng sự kiện) để tránh 1 bill bị
# cắt đôi giữa train/test — mỗi bill sinh nhiều dòng (nhiều stage +
# bản OUT-missing), nếu chia theo dòng thì 1 bill có thể vừa nằm trong
# train vừa nằm trong test, gây rò rỉ thông tin ở mức bill.
bill_arrival = df.groupby("bill_code")["arrival_time"].first().sort_values()
bill_split_idx = int(len(bill_arrival) * TRAIN_RATIO)
train_bills = set(bill_arrival.index[:bill_split_idx])
test_bills = set(bill_arrival.index[bill_split_idx:])

train = df[df["bill_code"].isin(train_bills)].copy()
test = df[df["bill_code"].isin(test_bills)].copy()

print(f"  Total : {len(df)}")
print(f"  Train : {len(train)}")
print(f"  Test  : {len(test)}")
print(f"  Train period: {train['bill_creation_date'].min()} -> {train['bill_creation_date'].max()}")
print(f"  Test period : {test['bill_creation_date'].min()} -> {test['bill_creation_date'].max()}")


# ================================================================
# 8b. GỘP CATEGORY HIẾM CHO CÁC INTERACTION/ROUTE FEATURE
# ================================================================
# full_route, district_route, route_service, route_VD_type là các
# chuỗi ghép categorical, cardinality rất lớn, nhiều giá trị chỉ xuất
# hiện vài lần → dễ overfit và không generalize. Gộp các giá trị hiếm
# (đếm trên TRAIN) thành "OTHER" trước khi đưa vào categorical dtype.

print()
print("=" * 60)
print("8b. GỘP CATEGORY HIẾM")
print("=" * 60)

HIGH_CARD_COLS = ["full_route", "district_route", "route_service", "route_VD_type"]


def fit_rare_category_keep_set(series, min_count):
    freq = series.value_counts()
    return set(freq[freq >= min_count].index)


def apply_rare_category_map(series, keep_set, other_label="OTHER"):
    return series.where(series.isin(keep_set), other_label)


for col in HIGH_CARD_COLS:
    n_before = train[col].nunique()
    keep_set = fit_rare_category_keep_set(train[col], RARE_CATEGORY_MIN_COUNT)
    train[col] = apply_rare_category_map(train[col], keep_set)
    test[col] = apply_rare_category_map(test[col], keep_set)
    print(f"  {col}: {n_before} -> {train[col].nunique()} categories (min_count={RARE_CATEGORY_MIN_COUNT})")


# ================================================================
# 9. HUẤN LUYỆN MÔ HÌNH LIGHTGBM
# ================================================================

print()
print("=" * 60)
print("9. HUẤN LUYỆN MÔ HÌNH LIGHTGBM")
print("=" * 60)

# --- Define features ---
feature_cols = [
    'warehouse_source', 'io_status_source', 'io_time_source',
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

categorical_features = [
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

# Make categorical variables compatible between train/test
for col in categorical_features:
    train[col] = train[col].astype("category")
    test[col] = pd.Categorical(
        test[col],
        categories=train[col].cat.categories
    )

X_train = train[feature_cols]
y_train = train["arrival_offset_hours"]

X_test = test[feature_cols]
y_test = test["arrival_offset_hours"]

# Bản OUT-missing (is_missing_out=1) có cùng target với bản gốc của cùng
# stage → đánh trọng số thấp hơn (MISSING_OUT_WEIGHT) thay vì để mô hình
# học với trọng số bằng nhau như đang nhân đôi vô điều kiện phân phối gốc.
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
    categorical_feature=categorical_features,
)

print("  Model training complete!")


# ================================================================
# 10. FEATURE IMPORTANCE
# ================================================================

print()
print("=" * 60)
print("10. FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "gain": model.booster_.feature_importance(importance_type="gain"),
    "split": model.booster_.feature_importance(importance_type="split"),
})

feature_importance["gain_pct"] = (
    feature_importance["gain"] / feature_importance["gain"].sum() * 100
)

feature_importance = feature_importance.sort_values(
    "gain", ascending=False
).reset_index(drop=True)

# Plot top features
top_n = 30
fi_top = feature_importance.head(top_n).sort_values("gain", ascending=True)

plt.figure(figsize=(10, 10))
plt.barh(fi_top["feature"], fi_top["gain"])
plt.xlabel("Feature importance (gain)")
plt.ylabel("Feature")
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()


# ================================================================
# 11. SAMPLING TEST SET (nếu quá lớn)
# ================================================================

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


# ================================================================
# 12. ĐÁNH GIÁ MÔ HÌNH THEO SNAPSHOT HÀNG NGÀY
# ================================================================

print()
print("=" * 60)
print("12. ĐÁNH GIÁ MÔ HÌNH THEO SNAPSHOT")
print("=" * 60)

# --- Prepare test data ---
test = test.copy()
test["bill_code"] = test["bill_code"].astype(str)
test["arrival_time"] = pd.to_datetime(test["arrival_time"], errors="coerce")

print(f"  Number of test bills: {test['bill_code'].nunique()}")
print(f"  Valid test arrival_time: {test['arrival_time'].notna().sum()}")

# --- Prepare trajectory data ---
traj = trajectory_df_sample[
    ["bill_code", "warehouse_name", "io_status", "io_time", "arrival_time"]
].copy()

traj["bill_code"] = traj["bill_code"].astype(str)

test_bill_codes = set(test["bill_code"].dropna().unique())
traj = traj[traj["bill_code"].isin(test_bill_codes)].copy()

print(f"  Trajectory rows before processing: {len(traj)}")
print(f"  Trajectory bills: {traj['bill_code'].nunique()}")

# Convert arrival time
traj["arrival_time"] = pd.to_datetime(traj["arrival_time"], errors="coerce")
print(f"  Valid trajectory arrival_time: {traj['arrival_time'].notna().sum()}")

# Remove final stage
traj["warehouse_name"] = traj["warehouse_name"].str[:-1]
traj["io_status"] = traj["io_status"].str[:-1]
traj["io_time"] = traj["io_time"].str[:-1]

# Explode trajectories
traj = traj.explode(
    ["warehouse_name", "io_status", "io_time"],
    ignore_index=True,
)
traj = traj.dropna(subset=["warehouse_name", "io_status", "io_time"]).copy()

print(f"  Trajectory rows after explode: {len(traj)}")

# Convert io_time
traj["io_time"] = pd.to_datetime(traj["io_time"], errors="coerce")
traj = traj.dropna(subset=["io_time"]).copy()

print(f"  Trajectory rows with valid io_time: {len(traj)}")

# Sort for merge_asof
traj = traj.sort_values(["io_time", "bill_code"]).reset_index(drop=True)
traj = traj[["bill_code", "warehouse_name", "io_status", "io_time", "arrival_time"]].copy()

# --- Determine evaluation date range ---
TEST_START_DATE = traj["io_time"].min().normalize()
LATEST_DATE = traj["arrival_time"].max().normalize()

print()
print("=" * 60)
print("EVALUATION PERIOD")
print("=" * 60)
print(f"  Evaluation start: {TEST_START_DATE}")
print(f"  Evaluation end  : {LATEST_DATE}")

evaluation_dates = pd.date_range(start=TEST_START_DATE, end=LATEST_DATE, freq="D")

snapshot_dates = pd.DataFrame({"snapshot_date": evaluation_dates})
snapshot_dates["snapshot_time"] = (
    snapshot_dates["snapshot_date"]
    + pd.Timedelta(hours=SNAPSHOT_HOUR)
    + pd.Timedelta(minutes=SNAPSHOT_MINUTE)
)

print(f"  Number of evaluation days: {len(snapshot_dates)}")
print(f"  Snapshot time: {SNAPSHOT_HOUR:02d}:{SNAPSHOT_MINUTE:02d}")

# --- Prepare bill-level features ---
bill_features = test.copy()
bill_features["bill_code"] = bill_features["bill_code"].astype(str)

if bill_features["bill_code"].duplicated().any():
    print("  WARNING: duplicated bill_code in test. Keeping last row.")
    bill_features = bill_features.drop_duplicates("bill_code", keep="last").copy()

print(f"  Unique test bills: {bill_features['bill_code'].nunique()}")

# Remove dynamic features (will be generated from snapshot)
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

# --- Chunked daily as-of evaluation ---
test_bill_codes_array = bill_features["bill_code"].drop_duplicates().values
n_bills = len(test_bill_codes_array)

print(f"  Bills to evaluate: {n_bills}")

all_results = []

print()
print("=" * 60)
print("START CHUNKED DAILY AS-OF EVALUATION")
print("=" * 60)

for start in range(0, n_bills, CHUNK_SIZE):

    end = min(start + CHUNK_SIZE, n_bills)
    chunk_codes = test_bill_codes_array[start:end]

    print(f"\n  Processing bills {start:,} - {end:,} / {n_bills:,}")

    # 11.1 Trajectory for this chunk
    traj_chunk = traj[traj["bill_code"].isin(chunk_codes)].copy()
    print(f"    trajectory rows: {len(traj_chunk)}")

    if len(traj_chunk) == 0:
        print("    No trajectory → skip")
        del traj_chunk
        gc.collect()
        continue

    # 11.2 Create bill × evaluation-day
    bill_days = pd.DataFrame({
        "bill_code": np.repeat(chunk_codes, len(snapshot_dates)),
        "snapshot_date": np.tile(snapshot_dates["snapshot_date"].values, len(chunk_codes)),
        "snapshot_time": np.tile(snapshot_dates["snapshot_time"].values, len(chunk_codes)),
    })
    print(f"    bill-day rows: {len(bill_days)}")

    # 11.3 Sort for merge_asof
    bill_days = bill_days.sort_values(
        ["snapshot_time", "bill_code"]
    ).reset_index(drop=True)

    traj_chunk = traj_chunk.sort_values(
        ["io_time", "bill_code"]
    ).reset_index(drop=True)

    # 11.4 As-of join
    current_stage = pd.merge_asof(
        bill_days,
        traj_chunk,
        by="bill_code",
        left_on="snapshot_time",
        right_on="io_time",
        direction="backward",
    )
    print(f"    after merge_asof: {len(current_stage)}")

    # 11.5 Remove bills that already arrived
    current_stage = current_stage[
        current_stage["arrival_time"] > current_stage["snapshot_time"]
    ].copy()
    print(f"    after not-arrived filter: {len(current_stage)}")

    # 11.6 Remove bills with no known stage
    current_stage = current_stage[
        current_stage["io_time"].notna()
    ].copy()
    print(f"    after known-stage filter: {len(current_stage)}")

    if len(current_stage) == 0:
        print("    No valid evaluation rows → skip")
        del traj_chunk, bill_days, current_stage
        gc.collect()
        continue

    # 11.7 Create dynamic stage features
    current_stage = current_stage.rename(columns={
        "warehouse_name": "warehouse_source",
        "io_status": "io_status_source",
    })

    t = current_stage["io_time"]
    current_stage["io_time_source"] = (
        t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    )

    # 11.8 Target
    current_stage["arrival_offset_hours"] = (
        current_stage["arrival_time"]
        - current_stage["io_time"].dt.normalize()
    ).dt.total_seconds() / 3600

    # 11.9 Merge bill-level features
    bill_features_chunk = bill_features[
        bill_features["bill_code"].isin(chunk_codes)
    ].copy()

    eval_chunk = current_stage.merge(
        bill_features_chunk,
        on="bill_code",
        how="left",
    )

    # 11.10 Categorical features
    eval_chunk["io_status_source"] = eval_chunk["io_status_source"].astype(str)
    eval_chunk["warehouse_source"] = eval_chunk["warehouse_source"].astype(str)

    for col in categorical_features:
        if col in eval_chunk.columns:
            eval_chunk[col] = pd.Categorical(
                eval_chunk[col],
                categories=train[col].cat.categories,
            )

    # 11.11 Check features
    missing_features = [
        col for col in feature_cols if col not in eval_chunk.columns
    ]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # 11.12 Prediction
    X_chunk = eval_chunk[feature_cols]
    eval_chunk["y_pred"] = model.predict(X_chunk)

    # 11.13 Keep only necessary results
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
    print(f"    predictions: {len(result_chunk)}")

    # 11.14 Free memory
    del traj_chunk, bill_days, current_stage, eval_chunk, X_chunk, result_chunk, bill_features_chunk
    gc.collect()


# ================================================================
# 13. GỘP KẾT QUẢ
# ================================================================

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


# ================================================================
# 14. KẾT QUẢ ĐÁNH GIÁ
# ================================================================

print()
print("=" * 60)
print("EVALUATION RESULT")
print("=" * 60)

print(f"Rows: {len(eval_df)}")

if len(eval_df) > 0:
    print(f"Bills: {eval_df['bill_code'].nunique()}")
    print(f"Days: {eval_df['snapshot_date'].nunique()}")
    print(f"Date range: {eval_df['snapshot_date'].min()} → {eval_df['snapshot_date'].max()}")
    print()
    print(eval_df.head())


# ================================================================
# 15. OVERALL METRICS
# ================================================================

if len(eval_df) > 0:

    y_true = eval_df["arrival_offset_hours"]
    y_pred = eval_df["y_pred"]

    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print()
    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"MAE   : {mae:.4f} hours")
    print(f"MedAE : {medae:.4f} hours")
    print(f"R2    : {r2:.4f}")


# ================================================================
# 16. METRICS BY WAREHOUSE
# ================================================================

print()
print("=" * 60)
print("METRICS BY WAREHOUSE")
print("=" * 60)

warehouse_metrics = []

for warehouse, group in eval_df.groupby("warehouse_source", dropna=False):

    group_valid = group[
        group["arrival_offset_hours"].notna()
        & group["y_pred"].notna()
    ]

    if len(group_valid) == 0:
        continue

    y_true_w = group_valid["arrival_offset_hours"]
    y_pred_w = group_valid["y_pred"]

    warehouse_metrics.append({
        "warehouse_source": warehouse,
        "n": len(group_valid),
        "MAE": mean_absolute_error(y_true_w, y_pred_w),
        "MedAE": median_absolute_error(y_true_w, y_pred_w),
        "R2": r2_score(y_true_w, y_pred_w) if len(group_valid) > 1 else np.nan,
    })

warehouse_metrics = pd.DataFrame(warehouse_metrics)
warehouse_metrics = warehouse_metrics.sort_values("MAE", ascending=True)

print(warehouse_metrics.to_string(index=False))


# ================================================================
# 17. WAREHOUSE FREQUENCY
# ================================================================

warehouse_freq = (
    trajectory_df["warehouse_2last"]
    .value_counts()
    .rename_axis("warehouse_2last")
    .reset_index(name="frequency")
)

print()
print("=" * 60)
print("WAREHOUSE FREQUENCY")
print("=" * 60)
print(warehouse_freq.head(10))

print()
print("=" * 60)
print("DONE!")
print("=" * 60)