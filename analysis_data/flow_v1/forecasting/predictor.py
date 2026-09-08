# -*- coding: utf-8 -*-
"""
forecasting/predictor.py
========================
Sử dụng:
    from forecasting import ForecastingPredictor

    predictor = ForecastingPredictor("saved_forecasting_models/kho_song_than/model_bundle.pkl")
    results = predictor.predict(df)
    print(results[["bill_code", "predicted_arrival_offset_hours"]])
"""

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURE_COLS
from .features import prepare_prediction_features
from .model_io import load_bundle


class ForecastingPredictor:
    """
    Dự đoán thời gian đến kho 1A cho đơn hàng.

    Attributes
    ----------
    bundle : dict
        Model bundle đã load từ file .pkl
    """

    def __init__(self, model_path: str):
        """Load model bundle từ file .pkl.

        Parameters
        ----------
        model_path : str
            Đường dẫn tới file model_bundle.pkl
        """
        self.bundle = load_bundle(model_path)
        self.segment_models = self.bundle.get("segment_models", {})
        if not self.segment_models:
            self.segment_models = {"OTHER": self.bundle.get("model")}
        self._feature_cols = self.bundle.get("feature_cols", FEATURE_COLS)

    @property
    def warehouse_name(self) -> str:
        """Tên kho 1A mà model này dự đoán."""
        selected_set = self.bundle.get("selected_set", set())
        return next(iter(selected_set), "Unknown")

    @property
    def train_metrics(self) -> dict:
        """Metrics từ lúc train."""
        return self.bundle.get("train_metrics", {})

    @property
    def trained_at(self) -> str:
        """Thời điểm train."""
        return self.bundle.get("trained_at", "N/A")

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict batch — dự đoán cho nhiều đơn hàng cùng lúc.

        Parameters
        ----------
        df : DataFrame
            Thông tin đơn hàng và trạng thái hiện tại trên 1 dòng.
            Tất cả cột đều là optional, nhưng nên có:
            warehouse_name, io_status, io_time, bill_code, VD_type,
            bill_creation_date, estimated_delivery_date, origin_province,
            origin_district, destination_province, destination_district,
            service, actual_weight

        Returns
        -------
        DataFrame gồm:
            bill_code                        — mã đơn (tự sinh nếu thiếu)
            warehouse_source                 — kho hiện tại
            io_status_source                 — trạng thái IN/OUT tại kho hiện tại
            predicted_arrival_offset_hours   — số giờ dự đoán (được clip >= 0)
            predicted_arrival_time           — datetime dự kiến đến kho 1A (nếu có io_time)
        """
        # 1. Lưu giá trị gốc để hiển thị (tránh mất giá trị lạ sau khi align)
        raw_warehouse = (
            df["warehouse_name"].astype(str)
            if "warehouse_name" in df.columns
            else df.get("warehouse_source", pd.Series(dtype=str))
        )
        raw_io_status = (
            df["io_status"].astype(str)
            if "io_status" in df.columns
            else df.get("io_status_source", pd.Series(dtype=str))
        )

        # 2. Chuẩn bị features
        result = prepare_prediction_features(df, self.bundle)

        # 3. Predict bằng mô hình tương ứng cho từng warehouse_source
        X = result[self._feature_cols]
        preds = np.zeros(len(X))

        routing_src = raw_warehouse.fillna("OTHER").values
        sources = np.unique(routing_src)
        for src in sources:
            mask = routing_src == src
            if mask.sum() > 0:
                model_info = self.segment_models.get(src)
                if model_info is None:
                    model_info = self.segment_models.get("OTHER")
                if model_info is None and self.segment_models:
                    first_key = next(iter(self.segment_models))
                    model_info = self.segment_models[first_key]
                if model_info is None:
                    model_info = self.bundle.get("model")
                if model_info is None:
                    continue

                X_t = X[mask]

                if isinstance(model_info, dict) and model_info.get("type") == "multi":
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
                            yp[idx_c] = np.mean([
                                model_info["regressors"][k].predict(X_t.iloc[idx_c])
                                for k in model_info["regressors"]
                            ], axis=0)
                    preds[mask] = yp
                elif isinstance(model_info, dict) and model_info.get("type") == "single":
                    preds[mask] = model_info["regressor"].predict(X_t)
                else:
                    preds[mask] = model_info.predict(X_t)

        # #3: Clip kết quả >= 0 khi trả về cho người dùng
        result["predicted_arrival_offset_hours"] = np.clip(preds, 0.0, None)

        # 4. Restore giá trị gốc cho output
        result["warehouse_source"] = raw_warehouse.values
        result["io_status_source"] = raw_io_status.values

        # 5. Tính predicted_arrival_time nếu có io_time / snapshot_time
        if "snapshot_time" in result.columns and result["snapshot_time"].notna().any():
            base_time = pd.to_datetime(result["snapshot_time"], errors="coerce")
        elif "io_time" in df.columns:
            base_time = pd.to_datetime(df["io_time"], errors="coerce")
        else:
            base_time = pd.Series([pd.Timestamp.now()] * len(result))

        result["predicted_arrival_time"] = (
            base_time + pd.to_timedelta(result["predicted_arrival_offset_hours"], unit="h")
        )

        # 6. Trả về kết quả gọn
        output_cols = [
            "bill_code",
            "warehouse_source",
            "io_status_source",
            "predicted_arrival_offset_hours",
            "predicted_arrival_time",
        ]
        output_cols = [c for c in output_cols if c in result.columns]

        return result[output_cols].copy()

    def predict_one(
        self,
        bill_info: dict,
    ) -> dict:
        """
        Predict cho 1 đơn hàng duy nhất.
        """
        df = pd.DataFrame([bill_info])
        result = self.predict(df)
        return result.iloc[0].to_dict()

    def __repr__(self) -> str:
        metrics = self.train_metrics
        mae_str = f", test_MAE={metrics['test_mae']:.2f}h" if "test_mae" in metrics else ""
        return (
            f"ForecastingPredictor("
            f"warehouse={self.warehouse_name!r}"
            f"{mae_str}"
            f", trained_at={self.trained_at!r})"
        )
