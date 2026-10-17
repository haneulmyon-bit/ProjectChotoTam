"""Кодирование признаков и масштабирование.

encode_features - детерминированное кодирование без «обучения» на данных
(категории фиксированы), его можно применять ко всему датасету до сплита.
build_preprocessor -- масштабирование, которое обязательно обучать только на train.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .data import drop_unused_columns


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Цифровизация признаков (п. 5 блокнота). Целевая переменная остается в таблице."""
    work = drop_unused_columns(df).copy()

    # 'No internet service' / 'No phone service' по смыслу равны 'No'
    work = work.replace({"No internet service": "No", "No phone service": "No"})

    # Бинарные Yes/No -> 1/0
    for c in work.columns:
        if set(work[c].dropna().unique()) <= {"Yes", "No"}:
            work[c] = work[c].map({"Yes": 1, "No": 0})
    work["Gender"] = work["Gender"].map({"Male": 1, "Female": 0})

    # Contract -- порядковый признак
    work["Contract"] = work["Contract"].map(cfg.CONTRACT_ORDER)

    # Номинальные -> one-hot (drop_first против идеальной коллинеарности)
    return pd.get_dummies(work, columns=cfg.ONE_HOT_COLS, drop_first=True, dtype=int)


def split_features_target(work: pd.DataFrame):
    """Делит закодированную таблицу на X и y."""
    return work.drop(columns=[cfg.TARGET]), work[cfg.TARGET]


def build_preprocessor(columns, numeric_cols=None, scaler=None) -> ColumnTransformer:
    """Масштабирует числовые признаки, остальные пропускает без изменений.

    Используйте внутри Pipeline или вызывайте fit только на X_train.
    """
    numeric_cols = numeric_cols or cfg.NUMERIC_COLS
    num = [c for c in numeric_cols if c in columns]
    ct = ColumnTransformer(
        [("num", scaler or StandardScaler(), num)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")
