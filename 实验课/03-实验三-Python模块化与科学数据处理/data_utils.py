"""实验三的数据读取、清洗、变换与统计函数。

请完成所有 TODO，并保持已经给出的函数名、参数和返回类型不变。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = ("feature_1", "feature_2", "feature_3")
REQUIRED_COLUMNS = ("sample_id", "label", *FEATURE_COLUMNS)
VALID_LABELS = {"A", "B", "C"}


def load_data(path: str | Path) -> pd.DataFrame:
    """从 CSV 文件读取原始样本表。"""
    # TODO 1：读取 CSV 并返回 DataFrame。
    raise NotImplementedError


def inspect_data(data: pd.DataFrame) -> dict[str, int]:
    """返回行数、列数、缺失单元格数和重复行数。"""
    # TODO 2：不要修改传入的 DataFrame。
    raise NotImplementedError


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """按照实验规则清洗数据，并返回索引连续的新 DataFrame。

    清洗规则：
    1. 缺少必需列时抛出 ValueError；
    2. 删除完全重复的记录；
    3. 将三个特征列转换为数值，无法转换的值记为缺失；
    4. 删除必需字段中含缺失值的记录；
    5. 只保留标签 A、B、C；
    6. 只保留三个特征值都大于 0 的记录。
    """
    # TODO 3：先复制数据，避免修改调用者持有的原始表。
    raise NotImplementedError


def standardize_features(data: pd.DataFrame) -> pd.DataFrame:
    """使用总体标准差标准化特征，并添加三个 z_ 开头的新列。"""
    # TODO 4：使用 NumPy 向量化完成，不逐行循环。
    raise NotImplementedError


def summarize_by_label(data: pd.DataFrame) -> pd.DataFrame:
    """按标签统计样本数和三个原始特征的均值。"""
    # TODO 5：结果按标签升序排列，标签保留为普通列。
    raise NotImplementedError


def find_most_unusual_sample(data: pd.DataFrame) -> str:
    """返回标准化特征平方和最大的样本编号；并列时保留先出现者。"""
    # TODO 6：可以调用 standardize_features()，不要复制其实现。
    raise NotImplementedError
