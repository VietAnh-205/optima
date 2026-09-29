# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_data.flow_v1.forecasting.config import (
    FEATURE_COLS,
    CATEGORICAL_FEATURES,
    MIN_SAMPLES_MULTIMODAL,
    lgbm_spec,
)
from analysis_data.flow_v1.forecasting.features import (
    prepare_prediction_features,
    align_categoricals,
    apply_rare_categories,
    engineer_bill_features,
)


class ArrivalRegressor:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.segment_models = {}
        self.feature_cols = list(FEATURE_COLS)
        if "predicted_first_1A" not in self.feature_cols:
            self.feature_cols.append("predicted_first_1A")

        self.categorical_features = list(CATEGORICAL_FEATURES)
        if "predicted_first_1A" not in self.categorical_features:
            self.categorical_features.append("predicted_first_1A")

        self.category_mappings = {}
        self.rare_keep_sets = {}
        self.outlier_bounds = {}
        self.is_fitted = False
        self.metrics = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        rare_keep_sets: dict = None,
        outlier_bounds: dict = None,
    ):
        print("\n" + "=" * 60)
        print("HUẤN LUYỆN ARRIVAL TIME REGRESSOR (SEGMENT + MULTI-MODAL)")
        print("=" * 60)

        train = train_df.copy()
        val = val_df.copy() if val_df is not None and not val_df.empty else pd.DataFrame()
        test = test_df.copy() if test_df is not None and not test_df.empty else pd.DataFrame()

        self.rare_keep_sets = rare_keep_sets or {}
        self.outlier_bounds = outlier_bounds or {}

        for target_col in ["time_to_first_1a_hours", "arrival_offset_hours"]:
            if target_col in train.columns:
                target = target_col
                break
        else:
            raise KeyError("train_df phải chứa cột 'time_to_first_1a_hours' hoặc 'arrival_offset_hours'")

        train["target_hours"] = train[target]
        if not val.empty:
            val["target_hours"] = val[target]
        if not test.empty:
            test["target_hours"] = test[target]

        for d in [train, val, test]:
            if not d.empty and "predicted_first_1A" not in d.columns:
                if "first_1A" in d.columns:
                    d["predicted_first_1A"] = d["first_1A"]
                else:
                    d["predicted_first_1A"] = "UNKNOWN"

        self.category_mappings = {}
        for col in self.categorical_features:
            if col not in train.columns:
                train[col] = "OTHER"
            train[col] = train[col].astype(object).fillna("OTHER").astype(str).astype("category")
            self.category_mappings[col] = train[col].cat.categories.tolist()

            cats_set = set(self.category_mappings[col])
            if not val.empty:
                if col not in val.columns:
                    val[col] = "OTHER"
                val_series = val[col].astype(object).fillna("OTHER").astype(str)
                val_series = val_series.where(val_series.isin(cats_set), "OTHER")
                val[col] = pd.Categorical(val_series, categories=self.category_mappings[col])

            if not test.empty:
                if col not in test.columns:
                    test[col] = "OTHER"
                test_series = test[col].astype(object).fillna("OTHER").astype(str)
                test_series = test_series.where(test_series.isin(cats_set), "OTHER")
                test[col] = pd.Categorical(test_series, categories=self.category_mappings[col])

        bill_counts = train.groupby("bill_code").size()
        train["sample_weight"] = 1.0 / np.sqrt(train["bill_code"].map(bill_counts))

        self.segment_models = {}
        segments = train["warehouse_source"].dropna().unique().tolist()
        print(f"  Số lượng phân khúc warehouse_source cần huấn luyện: {len(segments)}")

        for seg in segments:
            mask_train = (train["warehouse_source"] == seg).values
            mask_val = (val["warehouse_source"] == seg).values if not val.empty else np.array([], dtype=bool)

            X_tr = train[mask_train][self.feature_cols]
            y_tr = train[mask_train]["target_hours"].values
            w_tr = train[mask_train]["sample_weight"].values

            X_va = val[mask_val][self.feature_cols] if not val.empty and mask_val.sum() > 0 else pd.DataFrame()
            y_va = val[mask_val]["target_hours"].values if not val.empty and mask_val.sum() > 0 else np.array([])

            n_samples = len(X_tr)
            if n_samples < 5:
                continue

            print(f"\n    === SEGMENT: {seg} (n={n_samples:,}) ===")

            n_clusters = 1
            if n_samples >= MIN_SAMPLES_MULTIMODAL:
                if n_samples > 15_000:
                    rng = np.random.default_rng(self.random_state)
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
                for c in X_tr_clf.columns:
                    if hasattr(X_tr_clf[c], "cat") and X_tr_clf[c].cat is not None:
                        X_tr_clf[c] = X_tr_clf[c].cat.codes
                    elif X_tr_clf[c].dtype == "object":
                        X_tr_clf[c] = pd.Categorical(X_tr_clf[c].astype(str)).codes
                X_tr_clf = X_tr_clf.fillna(0)

                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=8, min_samples_leaf=5,
                    class_weight="balanced", random_state=self.random_state, n_jobs=-1
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

                    n_c = len(X_c)
                    if n_c >= 50:
                        cut = int(n_c * 0.85)
                        reg.fit(
                            X_c.iloc[:cut], y_c[:cut], sample_weight=w_c[:cut],
                            eval_set=[(X_c.iloc[cut:], y_c[cut:])],
                            callbacks=[early_stopping(stopping_rounds=30, verbose=False)],
                            categorical_feature=self.categorical_features,
                        )
                    else:
                        reg.fit(X_c, y_c, sample_weight=w_c, categorical_feature=self.categorical_features)

                    regressors[cid] = reg

                self.segment_models[seg] = {
                    "type": "multi",
                    "classifier": clf,
                    "regressors": regressors,
                }
            else:
                print(f"      -> Single-modal (1 cluster)")
                params = lgbm_spec(len(X_tr))
                reg = LGBMRegressor(**params)
                fit_kwargs = {
                    "sample_weight": w_tr,
                    "categorical_feature": self.categorical_features,
                }
                if len(X_va) > 0:
                    fit_kwargs["eval_set"] = [(X_va, y_va)]
                    fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=50, verbose=False)]
                reg.fit(X_tr, y_tr, **fit_kwargs)

                self.segment_models[seg] = {
                    "type": "single",
                    "regressor": reg,
                }

        print("\n  Huấn luyện mô hình Fallback toàn cục (OTHER)...")
        X_tr_all = train[self.feature_cols]
        y_tr_all = train["target_hours"].values
        w_tr_all = train["sample_weight"].values

        params_other = lgbm_spec(len(X_tr_all))
        model_other = LGBMRegressor(**params_other)
        fit_kwargs_other = {
            "sample_weight": w_tr_all,
            "categorical_feature": self.categorical_features,
        }
        if not val.empty and len(val) > 0:
            fit_kwargs_other["eval_set"] = [(val[self.feature_cols], val["target_hours"].values)]
            fit_kwargs_other["callbacks"] = [early_stopping(stopping_rounds=50, verbose=False)]

        model_other.fit(X_tr_all, y_tr_all, **fit_kwargs_other)
        self.segment_models["OTHER"] = {
            "type": "single",
            "regressor": model_other,
        }

        self.is_fitted = True
        print("  ArrivalRegressor hoàn tất huấn luyện!")

        if not test.empty and len(test) > 0:
            self.metrics = self.evaluate(test)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("ArrivalRegressor chưa được huấn luyện. Hãy gọi fit() trước.")

        res = df.copy()
        for col in res.select_dtypes(include=["category"]).columns:
            res[col] = res[col].astype(str)

        raw_warehouse = (
            res["warehouse_name"].astype(str)
            if "warehouse_name" in res.columns
            else res.get("warehouse_source", pd.Series(["OTHER"] * len(res), dtype=str))
        )
        raw_io_status = (
            res["io_status"].astype(str)
            if "io_status" in res.columns
            else res.get("io_status_source", pd.Series(["IN"] * len(res), dtype=str))
        )

        pseudo_bundle = {
            "feature_cols": self.feature_cols,
            "categorical_features": self.categorical_features,
            "category_mappings": self.category_mappings,
            "rare_keep_sets": self.rare_keep_sets,
        }
        X_ready = prepare_prediction_features(res, pseudo_bundle)

        for col in self.feature_cols:
            if col not in X_ready.columns:
                X_ready[col] = np.nan

        X_ready = align_categoricals(X_ready, self.category_mappings, cols=self.categorical_features)

        X = X_ready[self.feature_cols]
        preds = np.zeros(len(X))

        routing_src = raw_warehouse.astype(object).fillna("OTHER").values
        sources = np.unique(routing_src)

        for src in sources:
            mask = routing_src == src
            if mask.sum() == 0:
                continue

            model_info = self.segment_models.get(src, self.segment_models.get("OTHER"))
            if model_info is None:
                continue

            X_t = X[mask]

            if model_info["type"] == "multi":
                X_t_clf = X_t.copy()
                for c in X_t_clf.columns:
                    if hasattr(X_t_clf[c], "cat") and X_t_clf[c].cat is not None:
                        X_t_clf[c] = X_t_clf[c].cat.codes
                    elif X_t_clf[c].dtype == "object":
                        X_t_clf[c] = pd.Categorical(X_t_clf[c].astype(str)).codes
                X_t_clf = X_t_clf.fillna(0)

                c_pred = model_info["classifier"].predict(X_t_clf)
                yp = np.zeros(len(X_t))
                for cid in np.unique(c_pred):
                    idx_c = np.where(c_pred == cid)[0]
                    if cid in model_info["regressors"]:
                        yp[idx_c] = model_info["regressors"][cid].predict(X_t.iloc[idx_c])
                    else:
                        yp[idx_c] = np.mean([
                            model_info["regressors"][k].predict(X_t.iloc[idx_c])
                            for k in model_info["regressors"]
                        ], axis=0)
                preds[mask] = yp
            elif model_info["type"] == "single":
                preds[mask] = model_info["regressor"].predict(X_t)
            else:
                preds[mask] = model_info.predict(X_t)

        clipped_hours = np.clip(preds, 0.0, None)

        if "snapshot_time" in X_ready.columns and X_ready["snapshot_time"].notna().any():
            base_time = pd.to_datetime(X_ready["snapshot_time"], errors="coerce")
        elif "io_time" in res.columns:
            base_time = pd.to_datetime(res["io_time"], errors="coerce")
        elif "bill_creation_date" in res.columns:
            base_time = pd.to_datetime(res["bill_creation_date"], errors="coerce")
        else:
            base_time = pd.Series([pd.Timestamp.now()] * len(res))

        predicted_arrival_time = base_time + pd.to_timedelta(clipped_hours, unit="h")

        bill_codes = res["bill_code"].values if "bill_code" in res.columns else [f"bill_{i}" for i in range(len(res))]
        out = pd.DataFrame({
            "bill_code": bill_codes,
            "warehouse_source": raw_warehouse.values,
            "io_status_source": raw_io_status.values,
            "predicted_hours_to_first_1A": np.round(clipped_hours, 2),
            "predicted_arrival_time": predicted_arrival_time,
        })
        return out

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        for target_col in ["time_to_first_1a_hours", "arrival_offset_hours"]:
            if target_col in test_df.columns:
                target = target_col
                break
        else:
            print("  [evaluate] test_df không chứa cột target thời gian.")
            return {}

        valid = test_df.dropna(subset=[target]).copy()
        if valid.empty:
            return {}

        if "predicted_first_1A" not in valid.columns:
            if "first_1A" in valid.columns:
                valid["predicted_first_1A"] = valid["first_1A"]
            else:
                valid["predicted_first_1A"] = "UNKNOWN"

        preds = self.predict(valid)
        y_true = valid[target].values
        y_pred = preds["predicted_hours_to_first_1A"].values

        metrics = {
            "test_mae": float(mean_absolute_error(y_true, y_pred)),
            "test_medae": float(median_absolute_error(y_true, y_pred)),
            "test_r2": float(r2_score(y_true, y_pred)),
            "n_samples": len(valid),
        }

        print("\n" + "-" * 50)
        print("KẾT QUẢ ĐÁNH GIÁ ARRIVAL TIME REGRESSOR:")
        print(f"  Số mẫu đánh giá : {metrics['n_samples']:,}")
        print(f"  Test MAE        : {metrics['test_mae']:.4f} giờ")
        print(f"  Test MedAE      : {metrics['test_medae']:.4f} giờ")
        print(f"  Test R²         : {metrics['test_r2']:.4f}")
        print("-" * 50)

        return metrics
