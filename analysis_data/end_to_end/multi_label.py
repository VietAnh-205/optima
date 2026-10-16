# -*- coding: utf-8 -*-

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_data.end_to_end.e2e_preprocessor import (
    load_raw_data,
    build_e2e_training_data,
    prepare_e2e_splits,
)
from analysis_data.end_to_end.warehouse_classifier import WarehouseClassifier
from analysis_data.end_to_end.arrival_regressor import ArrivalRegressor


def load_data(bill_path: str, bill_schedule_path: str, warehouse_1a_path: str):
    return load_raw_data(
        bill_path=bill_path,
        bill_schedule_path=bill_schedule_path,
        warehouse_1a_path=warehouse_1a_path,
    )


class EndToEndPipeline:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier = WarehouseClassifier(random_state=random_state)
        self.regressor = ArrivalRegressor(random_state=random_state)
        self.is_fitted = False
        self.metrics = {}
        self.trained_at = None

    def train(
        self,
        data_dir: str = None,
        bill_path: str = None,
        bill_schedule_path: str = None,
        warehouse_1a_path: str = None,
        save_path: str = None,
        train_split: float = 0.75,
        val_split: float = 0.15,
        snapshot_hour: int = 20,
    ) -> dict:
        print("=" * 70)
        print("BẮT ĐẦU HUẤN LUYỆN END-TO-END MULTI-LABEL PIPELINE")
        print("=" * 70)

        sche, bill, wh1a = load_raw_data(
            data_dir=data_dir,
            bill_path=bill_path,
            bill_schedule_path=bill_schedule_path,
            warehouse_1a_path=warehouse_1a_path,
        )

        all_1a_warehouses = wh1a["name"].dropna().str.strip().unique().tolist()

        df = build_e2e_training_data(
            bill_schedule_df=sche,
            bill_df=bill,
            warehouse_1a_df=wh1a,
            snapshot_hour=snapshot_hour,
        )

        del sche, bill
        import gc
        gc.collect()

        train_df, val_df, test_df, rare_keep_sets, outlier_bounds = prepare_e2e_splits(
            df=df,
            train_split=train_split,
            val_split=val_split,
        )

        self.classifier.fit(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            all_warehouses=all_1a_warehouses,
        )

        self.regressor.fit(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            rare_keep_sets=rare_keep_sets,
            outlier_bounds=outlier_bounds,
        )

        self.is_fitted = True
        self.trained_at = datetime.now().isoformat()

        self.metrics = self.evaluate(test_df)

        if save_path is not None:
            self.save(save_path)

        print("\n" + "=" * 70)
        print("HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
        print("=" * 70)
        return self.metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được huấn luyện. Hãy gọi train() hoặc load() trước.")

        clf_preds = self.classifier.predict(df)

        df_reg = df.copy()
        df_reg["predicted_first_1A"] = clf_preds["predicted_first_1A"].values

        reg_preds = self.regressor.predict(df_reg)

        result = pd.DataFrame({
            "bill_code": clf_preds["bill_code"].values,
            "first_1A": clf_preds["predicted_first_1A"].values,
            "first_1A_confidence": clf_preds["first_1A_confidence"].values,
            "top3_first_1A": clf_preds["top3_first_1A"].values,
            "last_1A": clf_preds["predicted_last_1A"].values,
            "last_1A_confidence": clf_preds["last_1A_confidence"].values,
            "top3_last_1A": clf_preds["top3_last_1A"].values,
            "hours_to_first_1A": reg_preds["predicted_hours_to_first_1A"].values,
            "predicted_arrival_time": reg_preds["predicted_arrival_time"].values,
        }, index=df.index)

        if "warehouse_source" in reg_preds.columns:
            result["current_warehouse"] = reg_preds["warehouse_source"].values
        if "io_status_source" in reg_preds.columns:
            result["current_io_status"] = reg_preds["io_status_source"].values

        return result

    def predict_operational_window(
        self,
        df: pd.DataFrame,
        cutoff_time,
        window_start,
        window_end,
        route_pair: str | None = None,
    ) -> pd.DataFrame:
        """
        Dự báo chuyên biệt cho Vận Hành Theo Ca (Operational Shift Planning):
        Chuẩn hóa Giao thức Điểm Cắt Nghiệp Vụ (Cut-off Point Protocol):
          1. Lọc mốc Cut-off: Chỉ xét các snapshot phát sinh trước hoặc tại cutoff_time.
          2. Khử trùng lặp đa snapshot: Chỉ lấy bản ghi mới nhất của mỗi vận đơn (bill_code).
          3. Lọc trạng thái luân chuyển (In-transit): Loại bỏ các đơn đã cập bến đích trước cutoff_time.
          4. Dự báo qua pipeline End-to-End (Classifier -> Regressor).
          5. Lọc các đơn có predicted_arrival_time rơi vào cửa sổ ca [window_start, window_end).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được huấn luyện. Hãy gọi train() hoặc load() trước.")

        df_work = df.copy()
        if "snapshot_time" in df_work.columns:
            df_work["snapshot_time"] = pd.to_datetime(df_work["snapshot_time"], errors="coerce")
        else:
            df_work["snapshot_time"] = pd.Timestamp.now()

        cutoff_ts = pd.to_datetime(cutoff_time)
        ws_ts = pd.to_datetime(window_start)
        we_ts = pd.to_datetime(window_end)

        # 1. Chốt mốc cutoff
        sub_cut = df_work[df_work["snapshot_time"] <= cutoff_ts]
        if sub_cut.empty:
            return pd.DataFrame()

        # 2. Lấy snapshot mới nhất của từng đơn
        sub_cut = sub_cut.sort_values("snapshot_time")
        latest = sub_cut.drop_duplicates("bill_code", keep="last").copy()

        # 3. Lọc các đơn còn đang luân chuyển trên đường (chưa đến đích trước cutoff)
        if "arrival_time" in latest.columns:
            latest["arrival_time"] = pd.to_datetime(latest["arrival_time"], errors="coerce")
            in_transit = latest[latest["arrival_time"] >= cutoff_ts].copy()
        else:
            in_transit = latest.copy()

        if in_transit.empty:
            return pd.DataFrame()

        # 4. Dự báo qua pipeline
        preds = self.predict(in_transit)
        in_transit["pred_first_1A"] = preds["first_1A"].values
        in_transit["pred_last_1A"] = preds["last_1A"].values
        in_transit["predicted_arrival_time"] = pd.to_datetime(preds["predicted_arrival_time"], errors="coerce")
        in_transit["pred_pair"] = in_transit["pred_first_1A"].astype(str) + " -> " + in_transit["pred_last_1A"].astype(str)

        # 5. Lọc đơn dự báo rơi vào khung ca [window_start, window_end)
        mask_window = (in_transit["predicted_arrival_time"] >= ws_ts) & (in_transit["predicted_arrival_time"] < we_ts)
        if route_pair is not None and route_pair != "ALL_ROUTES":
            mask_window = mask_window & (in_transit["pred_pair"] == route_pair)

        return in_transit[mask_window].copy()

    def predict_one(self, bill_info: dict) -> dict:
        df = pd.DataFrame([bill_info])
        pred_df = self.predict(df)
        return pred_df.iloc[0].to_dict()

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        print("\n" + "=" * 60)
        print("ĐÁNH GIÁ TỔNG THỂ PIPELINE END-TO-END TRÊN TẬP TEST")
        print("=" * 60)

        clf_metrics = self.classifier.evaluate(test_df)
        reg_metrics = self.regressor.evaluate(test_df)

        combined = {
            "classifier": clf_metrics,
            "regressor": reg_metrics,
            "evaluated_at": datetime.now().isoformat(),
        }
        return combined

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "classifier": self.classifier,
            "regressor": self.regressor,
            "metrics": self.metrics,
            "trained_at": self.trained_at,
            "version": "1.0.0",
        }
        joblib.dump(bundle, filepath, compress=3)
        print(f"\n[EndToEndPipeline] Đã lưu mô hình thành công tại: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EndToEndPipeline":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")

        bundle = joblib.load(filepath)
        if not isinstance(bundle, dict) or "classifier" not in bundle or "regressor" not in bundle:
            raise ValueError("File bundle không hợp lệ cho EndToEndPipeline!")

        pipeline = cls()
        pipeline.classifier = bundle["classifier"]
        pipeline.regressor = bundle["regressor"]
        pipeline.metrics = bundle.get("metrics", {})
        pipeline.trained_at = bundle.get("trained_at", "N/A")
        pipeline.is_fitted = True

        print(f"[EndToEndPipeline] Đã load pipeline thành công (trained_at: {pipeline.trained_at})")
        return pipeline
