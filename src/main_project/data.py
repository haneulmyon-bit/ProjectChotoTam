"""Загрузка и очистка данных"""
import numpy as np
import pandas as pd

from . import config as cfg


def load_raw(path=None) -> pd.DataFrame:
    """Читает исходный Excel и убирает пробелы в названиях колонок."""
    df = pd.read_excel(path or cfg.DATA_RAW)
    df.columns = df.columns.str.strip()
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка: скрытые пропуски, Total Charges -> число, дубликаты.
    """
    df = df.copy()

    # «Скрытые» пропуски: пустые строки / пробелы среди текстовых колонок
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = (df[c].astype(str).str.strip()
                 .replace({"": np.nan, "nan": np.nan, "None": np.nan}))

    # Total Charges -> число; пустые у клиентов с tenure = 0 (счетов еще не было) -> 0
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    no_bill_yet = df["Total Charges"].isna() & (df["Tenure Months"] == 0)
    df.loc[no_bill_yet, "Total Charges"] = 0.0

    return df.drop_duplicates().reset_index(drop=True)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет служебные колонки и признаки с утечкой целевой переменной."""
    return df.drop(columns=cfg.DROP_COLS + cfg.LEAKAGE, errors="ignore")


def outlier_summary(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Сводка по выбросам: правило 1.5*IQR и |z| > 3."""
    cols = cols or cfg.OUTLIER_COLS
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (df[c] - df[c].mean()) / df[c].std()
        rows.append({"feature": c, "lower": round(lo, 2), "upper": round(hi, 2),
                     "n_IQR_outliers": int(((df[c] < lo) | (df[c] > hi)).sum()),
                     "n_|z|>3": int((z.abs() > 3).sum())})
    return pd.DataFrame(rows)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Уменьшает размер данных в памяти (int -> минимальный int, float -> float32)."""
    df = df.copy()
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def build_data_dictionary(raw_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    """Таблица соответствия: исходные признаки -> финальные, с комментарием о трансформации."""
    binary_yes_no = ["Senior Citizen", "Partner", "Dependents", "Phone Service",
                     "Paperless Billing", cfg.TARGET]
    addon_like = cfg.ADDONS + ["Multiple Lines"]

    rows = []
    for col in raw_df.columns:
        in_final = col in final_df.columns
        new_type = str(final_df[col].dtype) if in_final else "Удален (или исключен)"

        comment = ""
        if not in_final:
            if col in cfg.DROP_COLS:
                comment = "Исключен (высокая кардинальность / неинформативен / дублирует координаты)"
            elif col in cfg.LEAKAGE:
                comment = "Исключен (утечка целевой переменной / доступно только постфактум)"
            else:
                comment = "Удален в ходе VIF-фильтрации из-за высокой мультиколлинеарности"
        else:
            if col == "Gender":
                comment = "Кодирование: 1 - Male, 0 - Female"
            elif col in binary_yes_no:
                comment = "Кодирование: 1 - Yes, 0 - No"
            elif col == "Contract":
                comment = "Порядковое кодирование: 0 - Month-to-month, 1 - One year, 2 - Two year"
            elif col in addon_like:
                comment = "Кодирование: 1 - Yes, 0 - No/No service (объединены)"
            elif col in ["Latitude", "Longitude"]:
                comment = "Сохранен как вещественный признак, тип приведен к float32"

        rows.append({
            "Название атрибута (исходное)": col,
            "Описание": cfg.FEATURE_DESCRIPTIONS.get(col, "-"),
            "Исходный формат": str(raw_df[col].dtype),
            "Финальный формат": new_type,
            "Комментарий / Трансформация": comment,
        })

    # Признаки, созданные one-hot кодированием
    for col in final_df.columns:
        if col not in raw_df.columns:
            src = "Internet Service" if "Internet" in col else "Payment Method"
            rows.append({
                "Название атрибута (исходное)": col,
                "Описание": f"Бинарный флаг для категории '{col}' (создан из {src})",
                "Исходный формат": "-",
                "Финальный формат": str(final_df[col].dtype),
                "Комментарий / Трансформация": "One-Hot кодирование (1 - да, 0 - нет)",
            })
    return pd.DataFrame(rows)
