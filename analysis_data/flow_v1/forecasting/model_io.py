# -*- coding: utf-8 -*-


import os
import re

import joblib


def _slug(name: str) -> str:
    """Chuyển tên kho thành slug an toàn cho filesystem.

    VD: 'Kho Sóng Thần' → 'kho_song_than'
    """
    s = name.lower().strip()
    # Bỏ dấu tiếng Việt cơ bản
    _vn_map = str.maketrans(
        "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ",
        "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd",
    )
    s = s.translate(_vn_map)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def save_bundle(bundle: dict, output_dir: str) -> str:
    """Lưu model bundle ra file .pkl.

    Parameters
    ----------
    bundle : dict
        Model bundle dict (chứa model, feature_cols, category_mappings, ...).
    output_dir : str
        Thư mục gốc lưu models (VD: saved_forecasting_models/).
        Sẽ tạo thư mục con theo tên kho.

    Returns
    -------
    str : Đường dẫn file .pkl đã lưu
    """
    selected_set = bundle.get("selected_set", set())
    warehouse_name = next(iter(selected_set), "unknown")
    slug = _slug(warehouse_name)

    model_dir = os.path.join(output_dir, slug)
    os.makedirs(model_dir, exist_ok=True)

    pkl_path = os.path.join(model_dir, "model_bundle.pkl")
    joblib.dump(bundle, pkl_path, compress=3)

    print(f"  [model_io] Saved: {pkl_path}")
    return pkl_path


def load_bundle(path: str) -> dict:
    """Load model bundle từ file .pkl.

    Parameters
    ----------
    path : str
        Đường dẫn tới file model_bundle.pkl

    Returns
    -------
    dict : Model bundle
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError(f"Expected dict, got {type(bundle).__name__}")

    if "segment_models" not in bundle and "model" not in bundle:
        raise ValueError("Model bundle must contain either 'segment_models' or 'model'")

    required_keys = ["feature_cols", "categorical_features"]
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {missing}")

    return bundle


def list_available_models(models_dir: str) -> list[dict]:
    """Quét thư mục models, liệt kê tất cả model đã train.

    Parameters
    ----------
    models_dir : str
        Đường dẫn thư mục chứa các thư mục kho con

    Returns
    -------
    list[dict] : Mỗi dict chứa {warehouse_name, path, trained_at}
    """
    results = []
    if not os.path.isdir(models_dir):
        return results

    for slug in sorted(os.listdir(models_dir)):
        pkl_path = os.path.join(models_dir, slug, "model_bundle.pkl")
        if os.path.exists(pkl_path):
            try:
                bundle = joblib.load(pkl_path)
                selected_set = bundle.get("selected_set", set())
                warehouse_name = next(iter(selected_set), slug)
                results.append({
                    "warehouse_name": warehouse_name,
                    "path": pkl_path,
                    "trained_at": bundle.get("trained_at", "N/A"),
                    "train_metrics": bundle.get("train_metrics", {}),
                })
            except Exception:
                continue

    return results
