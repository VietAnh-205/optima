# -*- coding: utf-8 -*-
"""
forecasting/predictor.py
=========================
⭐ Hàm predict chính — ForecastingPredictor class.

Sử dụng:
    from forecasting import ForecastingPredictor

    predictor = ForecastingPredictor("saved_forecasting_models/kho_song_than/model_bundle.pkl")
    results = predictor.predict(df)
    print(results[["bill_code", "predicted_arrival_offset_hours"]])
"""

import pandas as pd

from .config import FEATURE_COLS
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
        self._model = self.bundle["model"]
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
            predicted_arrival_offset_hours   — số giờ dự đoán
            predicted_arrival_time           — datetime dự kiến đến kho 1A (nếu có io_time)
        """
        # 1. Lưu giá trị gốc để hiển thị (trước khi align_categoricals làm mất giá trị lạ)
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

        # 2. Chuẩn bị features (align_categoricals chạy ở đây)
        result = prepare_prediction_features(df, self.bundle)

        # 3. Predict (dùng categoricals đã align — KHÔNG thay đổi X trước bước này)
        X = result[self._feature_cols]
        result["predicted_arrival_offset_hours"] = self._model.predict(X)

        # 4. Restore giá trị gốc cho output SAU KHI predict xong
        result["warehouse_source"] = raw_warehouse.values
        result["io_status_source"] = raw_io_status.values


        # 3. Tính predicted_arrival_time nếu có io_time
        if "io_time" in df.columns:
            io_time = pd.to_datetime(df["io_time"], errors="coerce")
            io_date = io_time.dt.normalize()
            result["predicted_arrival_time"] = (
                io_date
                + pd.to_timedelta(result["predicted_arrival_offset_hours"], unit="h")
            )
        else:
            result["predicted_arrival_time"] = pd.NaT

        # 4. Trả về kết quả gọn
        output_cols = [
            "bill_code",
            "warehouse_source",
            "io_status_source",
            "predicted_arrival_offset_hours",
            "predicted_arrival_time",
        ]
        
        # Bỏ các cột không có trong DataFrame
        output_cols = [c for c in output_cols if c in result.columns]

        return result[output_cols].copy()

    def predict_one(
        self,
        bill_info: dict,
    ) -> dict:
        """
        Predict cho 1 đơn hàng duy nhất.

        Parameters
        ----------
        bill_info : dict
            Thông tin đơn hàng và trạng thái hiện tại, VD:
            {"warehouse_name": "Kho HN", "io_status": "IN", "io_time": "2026-01-15 08:00:00",
             "bill_code": "B_123", "VD_type": "BillWebOnline", ...}

        Returns
        -------
        dict với các key dự đoán
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
