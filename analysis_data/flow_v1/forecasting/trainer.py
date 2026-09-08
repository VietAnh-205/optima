# -*- coding: utf-8 -*-

import os
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

from .config import (
    CATEGORICAL_FEATURES,
    CHUNK_SIZE,
    FEATURE_COLS,
    HIGH_CARD_COLS,
    MIN_SAMPLES_MULTIMODAL,
    RARE_CATEGORY_MIN_COUNT,
    SNAPSHOT_HOURS,
    SNAPSHOT_HOUR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    lgbm_spec,
)
from .features import (
    align_categoricals,
    apply_outlier_bounds,
    create_stage_features,
    engineer_bill_features,
    fit_and_apply_rare_categories,
    fit_outlier_bounds,
    generate_pit_snapshot,
)
from .model_io import save_bundle, _slug


# ════════════════════════════════════════════════════════════════════════
# Load data helpers
# ════════════════════════════════════════════════════════════════════════

def _load_bill_schedule(path: str, chunksize: int = CHUNK_SIZE) -> pd.DataFrame:
    """Đọc bill_schedule.csv theo chunk, parse io_time."""
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
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    snapshot_hours: list[int] | None = None,
    output_dir: str | None = None,
) -> dict:
    """
    Chạy full pipeline: đọc data → lọc → trajectory → features → snapshot → split → train.
    """
    destination = next(iter(selected_set))
    if snapshot_hours is None:
        snapshot_hours = SNAPSHOT_HOURS

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
    sche = _load_bill_schedule(bill_schedule_path, chunksize=CHUNK_SIZE)
    print(f"  bill_schedule shape: {sche.shape}")

    # ── 2. LỌC BILLS ĐI QUA KHO 1A ĐƯỢC CHỌN (#5) ───────────────────
    print("\n[2] Lọc bills đi qua kho 1A được chọn...")

    # #5: Đồng bộ điều kiện: chỉ lấy các bản ghi check-in 'IN' tại kho đích
    arrival_mask = (sche["warehouse_name"].isin(selected_set)) & (sche["io_status"] == "IN")
    valid_bills = sche.loc[arrival_mask, "bill_code"].unique()
    if len(valid_bills) == 0:
        raise ValueError(
            f"Không tìm thấy bill nào có check-in 'IN' tại kho '{destination}'.\n"
            f"Kiểm tra lại tên kho trong warehouse_1A.csv (VD: 'Kho Sóng Thần')."
        )

    sche = sche[sche["bill_code"].isin(valid_bills)].copy()
    sche["row_id"] = sche.groupby("bill_code").cumcount()

    arrival_rows = sche[
        (sche["warehouse_name"].isin(selected_set)) & (sche["io_status"] == "IN")
    ]

    first_selected = (
        arrival_rows.groupby("bill_code")["row_id"]
        .min()
        .rename("first_selected_row")
    )

    sche = sche.merge(first_selected, on="bill_code", how="inner")

    # Chỉ giữ hành trình từ đầu đến lần đầu tiên check-in IN tại kho đích
    sche_filtered = sche[
        sche["row_id"] <= sche["first_selected_row"]
    ].drop(columns=["row_id", "first_selected_row"]).copy()

    # Loại bỏ bills đã từng qua kho 1A khác trước đó
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
            f"Sau khi lọc, không còn bill nào hợp lệ cho kho '{destination}'."
        )

    sche = sche_filtered.sort_values(["bill_code", "io_time"]).copy()

    # ── 3. XÂY DỰNG TRAJECTORY ──────────────────────────────────────
    print("\n[3] Xây dựng trajectory...")
    trajectory_df = (
        sche.groupby("bill_code")
        .agg({
            "warehouse_name": list,
            "io_status": list,
            "io_time": list,
        })
        .reset_index()
    )

    # #5: arrival_time đồng bộ lấy từ arrival_rows (lần đầu IN tại kho đích)
    arrival = arrival_rows.groupby("bill_code")["io_time"].first()
    trajectory_df["arrival_time"] = trajectory_df["bill_code"].map(arrival)
    trajectory_df = trajectory_df.dropna(subset=["arrival_time"])

    # Lọc trajectory có độ dài >= 2 (phải có ít nhất 1 scan trước kho đích)
    trajectory_df = trajectory_df[
        trajectory_df["io_status"].apply(len) >= 2
    ].copy()

    print(f"  trajectory_df shape: {trajectory_df.shape}")

    # ── 4. TẠO STAGE FEATURES (#4) ──────────────────────────────────
    print("\n[4] Tạo stage features...")
    stage_features = create_stage_features(trajectory_df)
    stage_features["io_status_source"] = stage_features["io_status_source"].astype(str)
    print(f"  stage_features shape: {stage_features.shape}")

    # ── 5. MERGE BILL INFO + FEATURE ENGINEERING ────────────────────
    print("\n[5] Feature engineering...")
    bill_full_subset = bill_full.merge(
        stage_features,
        on="bill_code",
        how="inner",
    )
    df = engineer_bill_features(bill_full_subset)
    df = df.sort_values("arrival_time").reset_index(drop=True)
    df = df.dropna(subset=["arrival_offset_hours"]).copy()
    print(f"  Engineered df shape: {df.shape}")

    # ── 6. ĐỒNG BỘ PHÂN PHỐI (SNAPSHOT 20:00 & REAL-TIME #1 & #7) ───
    print(f"\n[6] Tạo PIT Snapshot 20:00 & Real-Time scans...")
    df = generate_pit_snapshot(df, snapshot_hour=20)
    print(f"  Số lượng snapshot (train samples): {df.shape}")

    # ── 7. CHIA TRAIN / VAL / TEST (#11) ────────────────────────────
    print(f"\n[7] Chia train / val / test ({train_split*100:.0f}/{val_split*100:.0f}/{(1-train_split-val_split)*100:.0f})...")
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

    print(f"  Total bills: {n_bills} | Total rows: {len(df)}")
    print(f"  Train: {len(train_bills)} bills ({len(train)} rows)")
    print(f"  Val  : {len(val_bills)} bills ({len(val)} rows)")
    print(f"  Test : {len(test_bills)} bills ({len(test)} rows)")

    # ── 8. LOẠI BỎ OUTLIER (CHỈ FIT TRÊN TRAIN #2) ──────────────────
    print("\n[8] Loại bỏ outlier (Fit IQR trên Train, Apply cho cả 3 tập)...")
    lower_bound, upper_bound = fit_outlier_bounds(train)
    train = apply_outlier_bounds(train, lower_bound, upper_bound)
    val = apply_outlier_bounds(val, lower_bound, upper_bound)
    test = apply_outlier_bounds(test, lower_bound, upper_bound)

    # ── 9. GỘP CATEGORY HIẾM & ENCODING (#14) ───────────────────────
    print("\n[9] Gộp category hiếm...")
    train, val, test, rare_keep_sets = fit_and_apply_rare_categories(
        train, val, test,
        cols=HIGH_CARD_COLS,
        min_count=RARE_CATEGORY_MIN_COUNT,
    )

    category_mappings = {}
    for col in CATEGORICAL_FEATURES:
        train[col] = train[col].astype("category")
        category_mappings[col] = train[col].cat.categories.tolist()
        val[col] = pd.Categorical(val[col], categories=train[col].cat.categories)
        test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

    # ── 10. HUẤN LUYỆN MÔ HÌNH (DBSCAN + RF + LIGHTGBM #8, #9, #10) ───
    print("\n[10] Huấn luyện mô hình (Multi-modal/Single-modal)...")
    segment_models = {}
    segments = train["warehouse_source"].dropna().unique().tolist()
    print(f"  Số lượng segment (warehouse_source) sẽ train: {len(segments)}")

    bill_counts = train.groupby("bill_code").size()
    train["sample_weight"] = 1.0 / np.sqrt(train["bill_code"].map(bill_counts))

    for seg in segments:
        mask_train = train["warehouse_source"] == seg
        mask_val = val["warehouse_source"] == seg

        X_tr = train[mask_train][FEATURE_COLS]
        y_tr = train[mask_train]["arrival_offset_hours"].values
        w_tr = train[mask_train]["sample_weight"].values

        X_va = val[mask_val][FEATURE_COLS]
        y_va = val[mask_val]["arrival_offset_hours"].values

        n_samples = len(X_tr)
        if n_samples < 5:
            print(f"    - {seg}: Bỏ qua do quá ít dữ liệu ({n_samples} dòng)")
            continue

        print(f"\n    === SEGMENT: {seg} (n={n_samples}) ===")

        # #8: Kích hoạt Multi-modal chỉ khi n >= MIN_SAMPLES_MULTIMODAL (200)
        n_clusters = 1
        if n_samples >= MIN_SAMPLES_MULTIMODAL:
            # Subsample nếu n_samples lớn để tránh MemoryError / O(N^2) của DBSCAN
            if n_samples > 15_000:
                rng = np.random.default_rng(42)
                sub_idx = rng.choice(n_samples, size=15_000, replace=False)
                y_sub = y_tr[sub_idx]
            else:
                y_sub = y_tr

            dbscan = DBSCAN(eps=0.5, min_samples=max(10, int(len(y_sub) * 0.03)))
            c_sub = dbscan.fit_predict(y_sub.reshape(-1, 1))

            valid_mask = c_sub != -1
            if valid_mask.any():
                cluster_means = pd.Series(y_sub[valid_mask]).groupby(c_sub[valid_mask]).mean()
                sorted_clusters = cluster_means.sort_values().index.tolist()
                cluster_centers = [cluster_means[k] for k in sorted_clusters]

                if len(cluster_centers) >= 2:
                    centers_arr = np.array(cluster_centers)
                    dists = np.abs(y_tr[:, None] - centers_arr[None, :])
                    c_tr = np.argmin(dists, axis=1)
                    n_clusters = len(cluster_centers)

        if n_clusters >= 2:
            print(f"      -> Multi-modal: {n_clusters} clusters")

            X_tr_clf = X_tr.copy()
            for c in CATEGORICAL_FEATURES:
                if c in X_tr_clf.columns:
                    if hasattr(X_tr_clf[c], "cat") and X_tr_clf[c].cat is not None:
                        X_tr_clf[c] = X_tr_clf[c].cat.codes
                    else:
                        X_tr_clf[c] = pd.Categorical(X_tr_clf[c]).codes
            X_tr_clf = X_tr_clf.fillna(0)

            clf = RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            clf.fit(X_tr_clf, c_tr)

            regressors = {}
            for cid in np.unique(c_tr):
                mask_c = c_tr == cid
                X_c = X_tr[mask_c]
                y_c = y_tr[mask_c]
                w_c = w_tr[mask_c]

                params = lgbm_spec(len(X_c))
                reg = LGBMRegressor(**params)

                # #8: Early stopping nội bộ nếu cụm đủ mẫu
                n_c = len(X_c)
                if n_c >= 50:
                    cut = int(n_c * 0.85)
                    reg.fit(
                        X_c.iloc[:cut], y_c[:cut], sample_weight=w_c[:cut],
                        eval_set=[(X_c.iloc[cut:], y_c[cut:])],
                        callbacks=[early_stopping(stopping_rounds=30, verbose=False)],
                        categorical_feature=CATEGORICAL_FEATURES,
                    )
                else:
                    reg.fit(X_c, y_c, sample_weight=w_c, categorical_feature=CATEGORICAL_FEATURES)

                regressors[cid] = reg

            segment_models[seg] = {
                "type": "multi",
                "classifier": clf,
                "regressors": regressors,
            }
        else:
            print(f"      -> Single-modal (1 cụm)")
            params = lgbm_spec(len(X_tr))
            reg = LGBMRegressor(**params)
            fit_kwargs = {
                "sample_weight": w_tr,
                "categorical_feature": CATEGORICAL_FEATURES,
            }
            if len(X_va) > 0:
                fit_kwargs["eval_set"] = [(X_va, y_va)]
                fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=50, verbose=False)]
            reg.fit(X_tr, y_tr, **fit_kwargs)

            segment_models[seg] = {
                "type": "single",
                "regressor": reg,
            }

    # #9: Huấn luyện Fallback Model OTHER với eval_set trên toàn bộ Val
    print("\n  Train mô hình OTHER (Fallback toàn cục)...")
    X_tr_all = train[FEATURE_COLS]
    y_tr_all = train["arrival_offset_hours"].values
    w_tr_all = train["sample_weight"].values

    X_va_all = val[FEATURE_COLS]
    y_va_all = val["arrival_offset_hours"].values

    params = lgbm_spec(len(X_tr_all))
    model_other = LGBMRegressor(**params)
    fit_kwargs = {
        "sample_weight": w_tr_all,
        "categorical_feature": CATEGORICAL_FEATURES,
    }
    if len(X_va_all) > 0:
        fit_kwargs["eval_set"] = [(X_va_all, y_va_all)]
        fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=50, verbose=False)]
    model_other.fit(X_tr_all, y_tr_all, **fit_kwargs)

    segment_models["OTHER"] = {
        "type": "single",
        "regressor": model_other,
    }

    print("  Model training complete!")

    # ── 11. ĐÁNH GIÁ TRÊN TOÀN BỘ TẬP TEST (#3) ─────────────────────
    print("\n[11] Đánh giá metrics trên toàn bộ tập Test (Không lọc test_preds > 0)...")
    X_test = test[FEATURE_COLS]
    y_test = test["arrival_offset_hours"].values

    test_preds = np.zeros(len(test))
    for seg in test["warehouse_source"].unique():
        mask = test["warehouse_source"] == seg
        if mask.sum() > 0:
            model_info = segment_models.get(seg, segment_models["OTHER"])
            X_t = X_test[mask]

            if model_info["type"] == "multi":
                X_t_clf = X_t.copy()
                for c in CATEGORICAL_FEATURES:
                    if c in X_t_clf.columns:
                        if hasattr(X_t_clf[c], "cat") and X_t_clf[c].cat is not None:
                            X_t_clf[c] = X_t_clf[c].cat.codes
                        else:
                            X_t_clf[c] = pd.Categorical(X_t_clf[c]).codes
                X_t_clf = X_t_clf.fillna(0)

                c_pred = model_info["classifier"].predict(X_t_clf)
                yp = np.zeros(len(X_t))
                for cid in np.unique(c_pred):
                    idx_c = np.where(c_pred == cid)[0]
                    if cid in model_info["regressors"]:
                        yp[idx_c] = model_info["regressors"][cid].predict(X_t.iloc[idx_c])
                    else:
                        yp[idx_c] = np.mean([model_info["regressors"][k].predict(X_t.iloc[idx_c]) for k in model_info["regressors"]], axis=0)
                test_preds[mask] = yp
            else:
                test_preds[mask] = model_info["regressor"].predict(X_t)

    # #3: Tính metrics trên toàn bộ y_test không lọc
    if len(y_test) > 0:
        train_metrics = {
            "test_mae": float(mean_absolute_error(y_test, test_preds)),
            "test_medae": float(median_absolute_error(y_test, test_preds)),
            "test_r2": float(r2_score(y_test, test_preds)),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
        }
    else:
        train_metrics = {
            "test_mae": np.nan, "test_medae": np.nan, "test_r2": np.nan,
            "train_size": len(train), "val_size": len(val), "test_size": 0,
        }

    print(f"  Test  MAE  : {train_metrics['test_mae']:.4f} hours")
    print(f"  Test  MedAE: {train_metrics['test_medae']:.4f} hours")
    print(f"  Test  R²   : {train_metrics['test_r2']:.4f}")

    # ── 12. BUILD BUNDLE & SAVE ─────────────────────────────────────
    bundle = {
        "segment_models": segment_models,
        "selected_set": selected_set,
        "warehouse_name": destination,
        "slug": _slug(destination),
        "snapshot_hours": snapshot_hours,
        "feature_cols": FEATURE_COLS,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_mappings": category_mappings,
        "rare_keep_sets": rare_keep_sets,
        "outlier_bounds": {
            "lower": lower_bound,
            "upper": upper_bound,
        },
        "train_metrics": train_metrics,
        "trained_at": datetime.now().isoformat(),
    }

    if output_dir is not None:
        save_bundle(bundle, output_dir)

    print("\n" + "=" * 60)
    print(f"DONE — {destination}")
    print("=" * 60)

    return bundle
