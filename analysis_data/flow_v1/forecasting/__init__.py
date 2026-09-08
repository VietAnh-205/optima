# -*- coding: utf-8 -*-
"""

Sử dụng nhanh:
    from forecasting import ForecastingPredictor

    predictor = ForecastingPredictor("model_bundle.pkl")
    results = predictor.predict(bill_df, schedule_df)
"""

from .model_io import list_available_models, load_bundle, save_bundle, _slug
from .predictor import ForecastingPredictor
from .trainer import train_pipeline

__all__ = [
    "ForecastingPredictor",
    "train_pipeline",
    "load_bundle",
    "save_bundle",
    "list_available_models",
    "_slug",
]
