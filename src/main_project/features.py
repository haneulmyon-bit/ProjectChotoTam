"""Анализ признаков: корреляции и мультиколлинеарность (VIF).

Позже сюда же добавятся функции важности признаков (SHAP, permutation importance).
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from . import config as cfg


def correlated_pairs(X: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Пары признаков с |r| > threshold, по убыванию |r|."""
    corr = X.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().rename("r").reset_index()
    pairs.columns = ["feature_1", "feature_2", "r"]
    pairs = pairs[pairs.r.abs() > threshold]
    return pairs.sort_values("r", key=abs, ascending=False).reset_index(drop=True)


def calc_vif(X: pd.DataFrame) -> pd.Series:
    """VIF для каждого признака, по убыванию."""
    Xc = add_constant(X.astype(float))
    vif = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.Series(vif, index=X.columns).sort_values(ascending=False)


def drop_high_vif(X: pd.DataFrame, threshold: float = cfg.VIF_THRESHOLD):
    """Итеративно удаляет признак с максимальным VIF, пока он > threshold.

    Вызывать на X_train. Возвращает (X_reduced, список (признак, VIF)).
    """
    X_red = X.copy()
    dropped = []
    while X_red.shape[1] > 1:
        v = calc_vif(X_red)
        if v.iloc[0] <= threshold:
            break
        dropped.append((v.index[0], round(v.iloc[0], 1)))
        X_red = X_red.drop(columns=v.index[0])
    return X_red, dropped
