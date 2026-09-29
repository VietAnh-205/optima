# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_data.flow_v1.forecasting.features import engineer_bill_features


class SingleClassModel:
    def __init__(self, single_class):
        self.single_class = single_class
        self.classes_ = np.array([single_class])

    def predict(self, X):
        return np.array([self.single_class] * len(X))

    def predict_proba(self, X):
        return np.ones((len(X), 1))


class WarehouseClassifier:

    CLASSIFIER_FEATURES = [
        "VD_type",
        "origin_province",
        "origin_district",
        "destination_province",
        "destination_district",
        "service",
        "actual_weight",
        "log_actual_weight",
        "weight_bin",
        "province_route",
        "district_route",
        "estimated_delivery_days",
        "estimated_delivery_days_cat",
        "creation_hour_sin",
        "creation_hour_cos",
        "creation_dow_sin",
        "creation_dow_cos",
        "creation_month_sin",
        "creation_month_cos",
        "creation_delivery_dow_diff",
        "creation_delivery_dow_diff_circular",
        "same_creation_delivery_dow",
        "same_creation_delivery_month",
        "service_VD_type",
    ]

    CATEGORICAL_FEATURES = [
        "VD_type",
        "origin_province",
        "origin_district",
        "destination_province",
        "destination_district",
        "service",
        "weight_bin",
        "province_route",
        "district_route",
        "estimated_delivery_days_cat",
        "service_VD_type",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.first_1a_model = None
        self.last_1a_model = None
        self.classes_ = None
        self.category_mappings = {}
        self.is_fitted = False
        self.metrics = {}

    def _prepare_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        res = df.copy()
        for col in res.select_dtypes(include=["category"]).columns:
            res[col] = res[col].astype(str)

        missing_engineered = [c for c in ["province_route", "creation_hour_sin", "weight_bin"] if c not in res.columns]
        if missing_engineered:
            res = engineer_bill_features(res)

        for col in self.CLASSIFIER_FEATURES:
            if col not in res.columns:
                res[col] = np.nan

        if is_train:
            self.category_mappings = {}
            for col in self.CATEGORICAL_FEATURES:
                res[col] = res[col].astype(object).fillna("MISSING").astype(str).astype("category")
                self.category_mappings[col] = res[col].cat.categories.tolist()
        else:
            for col in self.CATEGORICAL_FEATURES:
                res[col] = res[col].astype(object).fillna("MISSING").astype(str)
                categories = self.category_mappings.get(col, [])
                res[col] = pd.Categorical(res[col], categories=categories)

        return res

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        all_warehouses: list[str] = None,
    ):
        print("\n" + "=" * 60)
        print("HUẤN LUYỆN WAREHOUSE CLASSIFIER (CASCADE)")
        print("=" * 60)

        train_bills = train_df.drop_duplicates(subset=["bill_code"]).copy()
        print(f"  Số bills tập train: {len(train_bills):,}")

        if all_warehouses is not None:
            self.classes_ = sorted(list(set(w.strip() for w in all_warehouses if pd.notna(w))))
        else:
            classes_set = set(train_bills["first_1A"].dropna().unique()).union(
                set(train_bills["last_1A"].dropna().unique())
            )
            self.classes_ = sorted(list(classes_set))

        print(f"  Số lượng nhãn kho 1A (num_classes): {len(self.classes_)}")

        X_tr = self._prepare_features(train_bills, is_train=True)
        y_tr_first = train_bills["first_1A"].astype(str).values
        y_tr_last = train_bills["last_1A"].astype(str).values

        has_val = val_df is not None and not val_df.empty
        if has_val:
            val_bills = val_df.drop_duplicates(subset=["bill_code"]).copy()
            X_va = self._prepare_features(val_bills, is_train=False)
            y_va_first = val_bills["first_1A"].astype(str).values
            y_va_last = val_bills["last_1A"].astype(str).values
            print(f"  Số bills tập val: {len(val_bills):,}")

        print("\n[1/2] Huấn luyện Model A (Dự đoán first_1A)...")
        n_samples = len(X_tr)
        n_est = 300 if n_samples < 10000 else 600
        lr = 0.05 if n_samples < 10000 else 0.03
        leaf_samples = max(1, min(20, int(n_samples * 0.02)))

        unique_first_classes = np.unique(y_tr_first)
        num_classes_a = len(unique_first_classes)

        if num_classes_a >= 2:
            self.first_1a_model = LGBMClassifier(
                objective="multiclass",
                num_class=num_classes_a,
                n_estimators=n_est,
                learning_rate=lr,
                num_leaves=31,
                max_depth=6,
                min_child_samples=leaf_samples,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

            fit_kwargs_a = {
                "categorical_feature": self.CATEGORICAL_FEATURES,
            }
            if has_val:
                val_mask_a = np.isin(y_va_first, unique_first_classes)
                if val_mask_a.sum() > 0:
                    fit_kwargs_a["eval_set"] = [(X_va[val_mask_a][self.CLASSIFIER_FEATURES], y_va_first[val_mask_a])]
                    fit_kwargs_a["callbacks"] = [early_stopping(stopping_rounds=30, verbose=False)]

            self.first_1a_model.fit(
                X_tr[self.CLASSIFIER_FEATURES],
                y_tr_first,
                **fit_kwargs_a,
            )
        else:
            self.first_1a_model = SingleClassModel(unique_first_classes[0])

        print("  Model A hoàn tất.")

        first_1a_pred_tr = self.first_1a_model.predict(X_tr[self.CLASSIFIER_FEATURES])

        print("\n[2/2] Huấn luyện Model B (Dự đoán last_1A - Cascade)...")
        features_b = self.CLASSIFIER_FEATURES + ["predicted_first_1A"]
        categoricals_b = self.CATEGORICAL_FEATURES + ["predicted_first_1A"]

        X_tr_b = X_tr.copy()
        X_tr_b["predicted_first_1A"] = pd.Categorical(first_1a_pred_tr, categories=self.classes_)

        unique_last_classes = np.unique(y_tr_last)
        num_classes_b = len(unique_last_classes)

        if num_classes_b >= 2:
            self.last_1a_model = LGBMClassifier(
                objective="multiclass",
                num_class=num_classes_b,
                n_estimators=n_est,
                learning_rate=lr,
                num_leaves=31,
                max_depth=6,
                min_child_samples=leaf_samples,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

            fit_kwargs_b = {
                "categorical_feature": categoricals_b,
            }
            if has_val:
                first_1a_pred_va = self.first_1a_model.predict(X_va[self.CLASSIFIER_FEATURES])
                X_va_b = X_va.copy()
                X_va_b["predicted_first_1A"] = pd.Categorical(first_1a_pred_va, categories=self.classes_)

                val_mask_b = np.isin(y_va_last, unique_last_classes)
                if val_mask_b.sum() > 0:
                    fit_kwargs_b["eval_set"] = [(X_va_b[val_mask_b][features_b], y_va_last[val_mask_b])]
                    fit_kwargs_b["callbacks"] = [early_stopping(stopping_rounds=30, verbose=False)]

            self.last_1a_model.fit(
                X_tr_b[features_b],
                y_tr_last,
                **fit_kwargs_b,
            )
        else:
            self.last_1a_model = SingleClassModel(unique_last_classes[0])

        print("  Model B hoàn tất.")

        self.is_fitted = True

        if test_df is not None and not test_df.empty:
            self.metrics = self.evaluate(test_df)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("WarehouseClassifier chưa được huấn luyện. Hãy gọi fit() trước.")

        X = self._prepare_features(df, is_train=False)

        first_probs = self.first_1a_model.predict_proba(X[self.CLASSIFIER_FEATURES])
        model_a_classes = self.first_1a_model.classes_

        top1_first_idx = np.argmax(first_probs, axis=1)
        pred_first = model_a_classes[top1_first_idx]
        conf_first = np.max(first_probs, axis=1)

        top3_first = []
        for row in first_probs:
            sorted_indices = np.argsort(-row)[:3]
            top3_first.append([model_a_classes[idx] for idx in sorted_indices])

        features_b = self.CLASSIFIER_FEATURES + ["predicted_first_1A"]
        X_b = X.copy()
        X_b["predicted_first_1A"] = pd.Categorical(pred_first, categories=self.classes_)

        last_probs = self.last_1a_model.predict_proba(X_b[features_b])
        model_b_classes = self.last_1a_model.classes_

        top1_last_idx = np.argmax(last_probs, axis=1)
        pred_last = model_b_classes[top1_last_idx]
        conf_last = np.max(last_probs, axis=1)

        top3_last = []
        for row in last_probs:
            sorted_indices = np.argsort(-row)[:3]
            top3_last.append([model_b_classes[idx] for idx in sorted_indices])

        bill_codes = df["bill_code"].values if "bill_code" in df.columns else [f"bill_{i}" for i in range(len(df))]
        out = pd.DataFrame({
            "bill_code": bill_codes,
            "predicted_first_1A": pred_first,
            "first_1A_confidence": np.round(conf_first, 4),
            "top3_first_1A": top3_first,
            "predicted_last_1A": pred_last,
            "last_1A_confidence": np.round(conf_last, 4),
            "top3_last_1A": top3_last,
        })
        return out

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        test_bills = test_df.drop_duplicates(subset=["bill_code"]).copy()
        if "first_1A" not in test_bills.columns or "last_1A" not in test_bills.columns:
            print("  [evaluate] test_df không chứa nhãn ground truth first_1A hoặc last_1A.")
            return {}

        preds = self.predict(test_bills)
        y_first = test_bills["first_1A"].astype(str).values
        y_last = test_bills["last_1A"].astype(str).values

        pred_first = preds["predicted_first_1A"].values
        pred_last = preds["predicted_last_1A"].values

        top3_first_correct = [
            truth in top3 for truth, top3 in zip(y_first, preds["top3_first_1A"])
        ]
        top3_last_correct = [
            truth in top3 for truth, top3 in zip(y_last, preds["top3_last_1A"])
        ]

        metrics = {
            "first_1A_top1_acc": float(accuracy_score(y_first, pred_first)),
            "first_1A_top3_acc": float(np.mean(top3_first_correct)),
            "first_1A_weighted_f1": float(f1_score(y_first, pred_first, average="weighted", zero_division=0)),
            "first_1A_macro_f1": float(f1_score(y_first, pred_first, average="macro", zero_division=0)),
            "last_1A_top1_acc": float(accuracy_score(y_last, pred_last)),
            "last_1A_top3_acc": float(np.mean(top3_last_correct)),
            "last_1A_weighted_f1": float(f1_score(y_last, pred_last, average="weighted", zero_division=0)),
            "last_1A_macro_f1": float(f1_score(y_last, pred_last, average="macro", zero_division=0)),
            "combined_both_correct_acc": float(np.mean((y_first == pred_first) & (y_last == pred_last))),
            "n_test_bills": len(test_bills),
        }

        print("\n" + "-" * 50)
        print("KẾT QUẢ ĐÁNH GIÁ WAREHOUSE CLASSIFIER:")
        print(f"  Test bills: {metrics['n_test_bills']:,}")
        print(f"  first_1A Top-1 Accuracy: {metrics['first_1A_top1_acc']*100:.2f}%")
        print(f"  first_1A Top-3 Accuracy: {metrics['first_1A_top3_acc']*100:.2f}%")
        print(f"  first_1A Weighted F1   : {metrics['first_1A_weighted_f1']:.4f}")
        print(f"  last_1A  Top-1 Accuracy: {metrics['last_1A_top1_acc']*100:.2f}%")
        print(f"  last_1A  Top-3 Accuracy: {metrics['last_1A_top3_acc']*100:.2f}%")
        print(f"  last_1A  Weighted F1   : {metrics['last_1A_weighted_f1']:.4f}")
        print(f"  ĐÚNG CẢ HAI KHO (Combined): {metrics['combined_both_correct_acc']*100:.2f}%")
        print("-" * 50)

        return metrics
