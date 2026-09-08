import os
import re
import math
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from preprocessor import (
    FEATURE_COLS,
    CATEGORICAL_FEATURES,
    HIGH_CARD_COLS,
    RARE_CATEGORY_MIN_COUNT,
    ARRIVAL_OFFSET_MAX_HOURS,
    fit_and_apply_rare_categories,
    prepare_prediction_features,
)


def slugify_name(name: str) -> str:
    s = name.lower().strip()
    vn_map = str.maketrans(
        "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ",
        "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd",
    )
    s = s.translate(vn_map)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def get_lgbm_params(n_samples: int) -> dict:
    n = max(int(n_samples), 1)
    depth = max(4, min(10, int(math.log2(max(n, 2)))))
    leaves = max(15, min(63, int(n / 100)))
    leaf = max(1, min(50, int(n * 0.02)))
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
        "verbose": -1,
    }


class ArrivalForecaster:
    FEATURE_COLS = FEATURE_COLS
    CAT_COLS = CATEGORICAL_FEATURES
    HIGH_CARD_COLS = HIGH_CARD_COLS
    DEFAULT_SAVE_DIR = "saved_models"

    def __init__(self, target_warehouse: str = None):
        self.target_warehouse = target_warehouse
        self.bundle = {}
        self.is_fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        target_warehouse: str = None,
        train_split: float = 0.75,
        val_split: float = 0.15,
        min_samples_multimodal: int = 200,
        save_dir: str = None,
        chronological: bool = True,
    ) -> "ArrivalForecaster":
        if target_warehouse:
            self.target_warehouse = target_warehouse
        dest = self.target_warehouse or "Global_1A"

        req_cols = ["bill_code", "warehouse_source", "io_status_source", "arrival_offset_hours"]
        for c in req_cols:
            if c not in train_df.columns:
                raise ValueError(f"Thiếu cột bắt buộc trong train_df: '{c}'")

        df = train_df.dropna(subset=["arrival_offset_hours"]).copy()

        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = np.nan

        if chronological and "arrival_time" in df.columns:
            bill_arrival = df.groupby("bill_code")["arrival_time"].first().sort_values()
            bills = list(bill_arrival.index)
        else:
            bills = list(df["bill_code"].unique())
            rng = np.random.default_rng(42)
            rng.shuffle(bills)

        n_b = len(bills)
        train_idx = int(n_b * train_split)
        val_idx = int(n_b * (train_split + val_split))

        tr_b = set(bills[:train_idx])
        va_b = set(bills[train_idx:val_idx])
        te_b = set(bills[val_idx:])

        train = df[df["bill_code"].isin(tr_b)].copy()
        val = df[df["bill_code"].isin(va_b)].copy()
        test = df[df["bill_code"].isin(te_b)].copy()

        valid_tr = train.dropna(subset=["warehouse_source", "io_status_source", "arrival_offset_hours"])
        grp = valid_tr.groupby(["warehouse_source", "io_status_source"])["arrival_offset_hours"]
        q1 = grp.quantile(0.25)
        q3 = grp.quantile(0.75)
        iqr = q3 - q1
        lo_bound = q1 - 1.5 * iqr
        up_bound = q3 + 1.5 * iqr

        def apply_outlier(d):
            if d.empty:
                return d
            k = list(zip(d["warehouse_source"], d["io_status_source"]))
            lo = pd.Series(k, index=d.index).map(lo_bound).fillna(0.0)
            hi = pd.Series(k, index=d.index).map(up_bound).fillna(ARRIVAL_OFFSET_MAX_HOURS)
            keep = (
                (d["arrival_offset_hours"] >= lo)
                & (d["arrival_offset_hours"] <= hi)
                & (d["arrival_offset_hours"] <= ARRIVAL_OFFSET_MAX_HOURS)
                & (d["arrival_offset_hours"] >= 0.0)
            )
            return d[keep].copy()

        train = apply_outlier(train)
        val = apply_outlier(val)
        test = apply_outlier(test)

        train, val, test, rare_keep_sets = fit_and_apply_rare_categories(
            train, val, test, cols=self.HIGH_CARD_COLS, min_count=RARE_CATEGORY_MIN_COUNT
        )

        cat_mappings = {}
        for c in self.CAT_COLS:
            train[c] = train[c].astype("category")
            cat_mappings[c] = train[c].cat.categories.tolist()
            val[c] = pd.Categorical(val[c], categories=train[c].cat.categories)
            test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

        segment_models = {}
        segments = train["warehouse_source"].dropna().unique().tolist()
        bill_cnts = train.groupby("bill_code").size()
        train["sample_weight"] = 1.0 / np.sqrt(train["bill_code"].map(bill_cnts))

        for seg in segments:
            tr_m = train["warehouse_source"] == seg
            va_m = val["warehouse_source"] == seg

            X_tr = train[tr_m][self.FEATURE_COLS]
            y_tr = train[tr_m]["arrival_offset_hours"].values
            w_tr = train[tr_m]["sample_weight"].values

            X_va = val[va_m][self.FEATURE_COLS]
            y_va = val[va_m]["arrival_offset_hours"].values

            n_samples = len(X_tr)
            if n_samples < 5:
                continue

            n_clusters = 1
            if n_samples >= min_samples_multimodal:
                if n_samples > 15_000:
                    rng = np.random.default_rng(42)
                    sub_idx = rng.choice(n_samples, size=15_000, replace=False)
                    y_sub = y_tr[sub_idx]
                else:
                    y_sub = y_tr

                db = DBSCAN(eps=0.5, min_samples=max(10, int(len(y_sub) * 0.03))).fit(y_sub.reshape(-1, 1))
                valid_mask = db.labels_ != -1
                if valid_mask.any():
                    cluster_means = pd.Series(y_sub[valid_mask]).groupby(db.labels_[valid_mask]).mean()
                    sorted_clusters = cluster_means.sort_values().index.tolist()
                    cluster_centers = [cluster_means[k] for k in sorted_clusters]
                    if len(cluster_centers) >= 2:
                        centers_arr = np.array(cluster_centers)
                        dists = np.abs(y_tr[:, None] - centers_arr[None, :])
                        c_tr = np.argmin(dists, axis=1)
                        n_clusters = len(cluster_centers)

            if n_clusters >= 2:
                X_tr_clf = X_tr.copy()
                for c in self.CAT_COLS:
                    if hasattr(X_tr_clf[c], "cat") and X_tr_clf[c].cat is not None:
                        X_tr_clf[c] = X_tr_clf[c].cat.codes
                    else:
                        X_tr_clf[c] = pd.Categorical(X_tr_clf[c]).codes
                X_tr_clf = X_tr_clf.fillna(0)

                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=8, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1
                ).fit(X_tr_clf, c_tr)

                regressors = {}
                for cid in np.unique(c_tr):
                    mc = c_tr == cid
                    Xc, yc, wc = X_tr[mc], y_tr[mc], w_tr[mc]
                    reg = LGBMRegressor(**get_lgbm_params(len(Xc)))
                    if len(Xc) >= 50:
                        cut = int(len(Xc) * 0.85)
                        reg.fit(
                            Xc.iloc[:cut], yc[:cut], sample_weight=wc[:cut],
                            eval_set=[(Xc.iloc[cut:], yc[cut:])],
                            callbacks=[early_stopping(30, verbose=False)],
                            categorical_feature=self.CAT_COLS,
                        )
                    else:
                        reg.fit(Xc, yc, sample_weight=wc, categorical_feature=self.CAT_COLS)
                    regressors[cid] = reg

                segment_models[seg] = {"type": "multi", "classifier": clf, "regressors": regressors}
            else:
                reg = LGBMRegressor(**get_lgbm_params(len(X_tr)))
                kw = {"sample_weight": w_tr, "categorical_feature": self.CAT_COLS}
                if len(X_va) > 0:
                    kw["eval_set"] = [(X_va, y_va)]
                    kw["callbacks"] = [early_stopping(50, verbose=False)]
                reg.fit(X_tr, y_tr, **kw)
                segment_models[seg] = {"type": "single", "regressor": reg}

        m_other = LGBMRegressor(**get_lgbm_params(len(train)))
        kw_o = {"sample_weight": train["sample_weight"].values, "categorical_feature": self.CAT_COLS}
        if len(val) > 0:
            kw_o["eval_set"] = [(val[self.FEATURE_COLS], val["arrival_offset_hours"].values)]
            kw_o["callbacks"] = [early_stopping(50, verbose=False)]
        m_other.fit(train[self.FEATURE_COLS], train["arrival_offset_hours"].values, **kw_o)
        segment_models["OTHER"] = {"type": "single", "regressor": m_other}

        y_test = test["arrival_offset_hours"].values
        X_test = test[self.FEATURE_COLS]
        test_preds = np.zeros(len(test))

        for seg in test["warehouse_source"].unique():
            m = test["warehouse_source"] == seg
            if m.sum() > 0:
                mi = segment_models.get(seg, segment_models["OTHER"])
                Xt = X_test[m]
                if mi["type"] == "multi":
                    Xt_c = Xt.copy()
                    for c in self.CAT_COLS:
                        if hasattr(Xt_c[c], "cat") and Xt_c[c].cat is not None:
                            Xt_c[c] = Xt_c[c].cat.codes
                        else:
                            Xt_c[c] = pd.Categorical(Xt_c[c]).codes
                    Xt_c = Xt_c.fillna(0)

                    cp = mi["classifier"].predict(Xt_c)
                    yp = np.zeros(len(Xt))
                    for cid in np.unique(cp):
                        idx = np.where(cp == cid)[0]
                        if cid in mi["regressors"]:
                            yp[idx] = mi["regressors"][cid].predict(Xt.iloc[idx])
                        else:
                            yp[idx] = np.mean([mi["regressors"][k].predict(Xt.iloc[idx]) for k in mi["regressors"]], axis=0)
                    test_preds[m] = yp
                else:
                    test_preds[m] = mi["regressor"].predict(Xt)

        metrics = {
            "test_mae": float(mean_absolute_error(y_test, test_preds)) if len(y_test) else np.nan,
            "test_medae": float(median_absolute_error(y_test, test_preds)) if len(y_test) else np.nan,
            "test_r2": float(r2_score(y_test, test_preds)) if len(y_test) else np.nan,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
        }

        slug = slugify_name(dest)
        self.bundle = {
            "warehouse_name": dest,
            "slug": slug,
            "segment_models": segment_models,
            "category_mappings": cat_mappings,
            "rare_keep_sets": rare_keep_sets,
            "outlier_bounds": {"lower": lo_bound, "upper": up_bound},
            "feature_cols": self.FEATURE_COLS,
            "categorical_features": self.CAT_COLS,
            "train_metrics": metrics,
            "trained_at": datetime.now().isoformat(),
        }
        self.is_fitted = True

        out_dir = save_dir or self.DEFAULT_SAVE_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{slug}.pkl")
        self.save(out_path)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.bundle:
            raise ValueError("Mô hình chưa được fit hoặc load!")

        raw_wh = (
            df["warehouse_name"].astype(str)
            if "warehouse_name" in df.columns
            else df.get("warehouse_source", pd.Series(["UNKNOWN"] * len(df))).astype(str)
        )
        raw_st = (
            df["io_status"].astype(str)
            if "io_status" in df.columns
            else df.get("io_status_source", pd.Series(["UNKNOWN"] * len(df))).astype(str)
        )

        result = prepare_prediction_features(df, self.bundle)

        feature_cols = self.bundle.get("feature_cols", self.FEATURE_COLS)
        cat_cols = self.bundle.get("categorical_features", self.CAT_COLS)
        seg_models = self.bundle["segment_models"]

        X = result[feature_cols]
        preds = np.zeros(len(X))

        routing = raw_wh.fillna("OTHER").values
        for src in np.unique(routing):
            m = routing == src
            if m.sum() > 0:
                mi = seg_models.get(src)
                if mi is None:
                    mi = seg_models.get("OTHER")
                if mi is None and seg_models:
                    mi = next(iter(seg_models.values()))

                Xt = X[m]
                if mi and mi.get("type") == "multi":
                    Xt_c = Xt.copy()
                    for c in cat_cols:
                        if hasattr(Xt_c[c], "cat") and Xt_c[c].cat is not None:
                            Xt_c[c] = Xt_c[c].cat.codes
                        else:
                            Xt_c[c] = pd.Categorical(Xt_c[c]).codes
                    Xt_c = Xt_c.fillna(0)

                    cp = mi["classifier"].predict(Xt_c)
                    yp = np.zeros(len(Xt))
                    for cid in np.unique(cp):
                        idx = np.where(cp == cid)[0]
                        if cid in mi["regressors"]:
                            yp[idx] = mi["regressors"][cid].predict(Xt.iloc[idx])
                        else:
                            yp[idx] = np.mean([mi["regressors"][k].predict(Xt.iloc[idx]) for k in mi["regressors"]], axis=0)
                    preds[m] = yp
                elif mi and mi.get("type") == "single":
                    preds[m] = mi["regressor"].predict(Xt)
                elif mi and hasattr(mi, "predict"):
                    preds[m] = mi.predict(Xt)

        result["predicted_arrival_offset_hours"] = np.clip(preds, 0.0, None)
        result["warehouse_source"] = raw_wh.values
        result["io_status_source"] = raw_st.values

        if "snapshot_time" in result.columns and result["snapshot_time"].notna().any():
            base_time = pd.to_datetime(result["snapshot_time"], errors="coerce")
        elif "io_time" in df.columns and pd.to_datetime(df["io_time"], errors="coerce").notna().any():
            base_time = pd.to_datetime(df["io_time"], errors="coerce")
        else:
            base_time = pd.Series([pd.Timestamp.now()] * len(result), index=result.index)

        result["predicted_arrival_time"] = base_time + pd.to_timedelta(result["predicted_arrival_offset_hours"], unit="h")

        output_cols = [
            "bill_code",
            "warehouse_source",
            "io_status_source",
            "predicted_arrival_offset_hours",
            "predicted_arrival_time",
        ]
        return result[output_cols].copy()

    def predict_one(self, bill_info: dict) -> dict:
        df = pd.DataFrame([bill_info])
        return self.predict(df).iloc[0].to_dict()

    def save(self, filepath: str):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        joblib.dump(self.bundle, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str) -> "ArrivalForecaster":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file model: {filepath}")
        bundle = joblib.load(filepath)
        inst = cls(target_warehouse=bundle.get("warehouse_name"))
        inst.bundle = bundle
        inst.is_fitted = True
        return inst

    @classmethod
    def list_available_models(cls, models_dir: str = DEFAULT_SAVE_DIR, load_metrics: bool = False) -> list[dict]:
        results = []
        if not os.path.isdir(models_dir):
            return results

        summary_csv = os.path.join(models_dir, "training_summary.csv")
        if os.path.exists(summary_csv):
            try:
                sdf = pd.read_csv(summary_csv)
                for _, row in sdf.iterrows():
                    wh = str(row.get("warehouse", ""))
                    slug = slugify_name(wh)
                    pkl_name = f"{slug}.pkl"
                    pkl_path = os.path.join(models_dir, pkl_name)
                    if not os.path.exists(pkl_path):
                        cand = os.path.join(models_dir, f"{wh.lower().replace(' ', '_')}.pkl")
                        if os.path.exists(cand):
                            pkl_path = cand

                    if os.path.exists(pkl_path):
                        results.append({
                            "warehouse_name": wh,
                            "slug": slug,
                            "path": pkl_path,
                            "train_metrics": {
                                "test_mae": row.get("test_mae"),
                                "test_medae": row.get("test_medae"),
                                "test_r2": row.get("test_r2"),
                                "samples": row.get("samples"),
                            },
                        })
                if results:
                    return results
            except Exception:
                pass

        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pkl"):
                pkl_path = os.path.join(models_dir, f)
                slug = f.replace(".pkl", "")
                wh_name = slug.replace("_", " ").title()
                entry = {
                    "warehouse_name": wh_name,
                    "slug": slug,
                    "path": pkl_path,
                    "trained_at": "N/A",
                    "train_metrics": {},
                }
                if load_metrics:
                    try:
                        b = joblib.load(pkl_path)
                        entry["warehouse_name"] = b.get("warehouse_name", wh_name)
                        entry["slug"] = b.get("slug", slug)
                        entry["trained_at"] = b.get("trained_at", "N/A")
                        entry["train_metrics"] = b.get("train_metrics", {})
                    except Exception:
                        pass
                results.append(entry)
        return results

    def __repr__(self) -> str:
        wh = self.bundle.get("warehouse_name", self.target_warehouse or "Unset")
        metrics = self.bundle.get("train_metrics", {})
        mae_str = f", test_MAE={metrics['test_mae']:.2f}h" if "test_mae" in metrics else ""
        return f"ArrivalForecaster(warehouse={wh!r}{mae_str}, is_fitted={self.is_fitted})"
