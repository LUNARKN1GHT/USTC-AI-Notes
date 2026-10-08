"""运行教师参考实现的独立验证脚本。"""

from pathlib import Path

import numpy as np

from data_utils_参考答案 import (
    FEATURE_COLUMNS,
    clean_data,
    find_most_unusual_sample,
    inspect_data,
    load_data,
    standardize_features,
    summarize_by_label,
)

ROOT = Path(__file__).parent

raw = load_data(ROOT / "data" / "samples.csv")
assert inspect_data(raw) == {
    "rows": 16,
    "columns": 5,
    "missing_cells": 2,
    "duplicate_rows": 1,
}

cleaned = clean_data(raw)
assert cleaned.shape == (11, 5)
assert cleaned["sample_id"].is_unique
assert cleaned["label"].value_counts().to_dict() == {"B": 5, "A": 3, "C": 3}
assert not cleaned.isna().any().any()

standardized = standardize_features(cleaned)
z_columns = [f"z_{column}" for column in FEATURE_COLUMNS]
assert standardized.shape == (11, 8)
assert np.allclose(standardized[z_columns].mean().to_numpy(), 0.0)
assert np.allclose(standardized[z_columns].std(ddof=0).to_numpy(), 1.0)

summary = summarize_by_label(cleaned)
assert summary["label"].tolist() == ["A", "B", "C"]
assert summary["sample_count"].tolist() == [3, 5, 3]
assert find_most_unusual_sample(cleaned) == "S001"

print("✅ 教师参考实现全部验证通过")
