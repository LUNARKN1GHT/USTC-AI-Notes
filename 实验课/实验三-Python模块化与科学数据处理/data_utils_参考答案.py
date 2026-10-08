"""实验三数据处理模块的教师参考实现。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ("feature_1", "feature_2", "feature_3")
REQUIRED_COLUMNS = ("sample_id", "label", *FEATURE_COLUMNS)
VALID_LABELS = {"A", "B", "C"}


def load_data(path: str | Path) -> pd.DataFrame:
    """从 CSV 文件读取原始样本表。"""
    return pd.read_csv(path)


def inspect_data(data: pd.DataFrame) -> dict[str, int]:
    """返回行数、列数、缺失单元格数和重复行数。"""
    return {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """按照实验规则清洗数据，并返回索引连续的新 DataFrame。"""
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"缺少必需列：{names}")

    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates()

    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=list(REQUIRED_COLUMNS))
    cleaned = cleaned[cleaned["label"].isin(VALID_LABELS)]
    cleaned = cleaned[(cleaned[list(FEATURE_COLUMNS)] > 0).all(axis=1)]
    return cleaned.reset_index(drop=True)


def standardize_features(data: pd.DataFrame) -> pd.DataFrame:
    """使用总体标准差标准化特征，并添加三个 z_ 开头的新列。"""
    result = data.copy()
    values = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0)

    if np.any(standard_deviations == 0):
        raise ValueError("特征的标准差不能为 0")

    standardized = (values - means) / standard_deviations
    standardized_columns = [f"z_{column}" for column in FEATURE_COLUMNS]
    result.loc[:, standardized_columns] = standardized
    return result


def summarize_by_label(data: pd.DataFrame) -> pd.DataFrame:
    """按标签统计样本数和三个原始特征的均值。"""
    summary = (
        data.groupby("label", as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            mean_feature_1=("feature_1", "mean"),
            mean_feature_2=("feature_2", "mean"),
            mean_feature_3=("feature_3", "mean"),
        )
        .sort_values("label")
    )
    return summary.reset_index(drop=True)


def find_most_unusual_sample(data: pd.DataFrame) -> str:
    """返回标准化特征平方和最大的样本编号；并列时保留先出现者。"""
    standardized = standardize_features(data)
    standardized_columns = [f"z_{column}" for column in FEATURE_COLUMNS]
    scores = np.square(standardized[standardized_columns].to_numpy()).sum(axis=1)
    position = int(np.argmax(scores))
    return str(standardized.iloc[position]["sample_id"])
