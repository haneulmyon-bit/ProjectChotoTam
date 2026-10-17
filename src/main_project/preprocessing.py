"""Блокнот, п. 5: цифровизация (кодирование) признаков.

Вход: data/interim/telco_clean.pkl. Результат: data/interim/telco_encoded.pkl
"""
# ----- связка между файлами (добавлено при переносе) -----
import pandas as pd
from main_project import config as cfg
from main_project.config import LEAKAGE

df = pd.read_pickle(cfg.DATA_INTERIM / "telco_clean.pkl")

# ===== КОД ИЗ БЛОКНОТА (без изменений, кроме строк с пометкой «ИЗМЕНЕНО») =====

print("\n" + "=" * 70, "\n5. ЦИФРОВИЗАЦИЯ\n", "=" * 70)
work = df.drop(columns=["CustomerID", "Count", "Country", "State", "City",
                        "Zip Code", "Lat Long"] + LEAKAGE, errors="ignore").copy()

# 'No internet service' / 'No phone service' по смыслу равны 'No'
work = work.replace({"No internet service": "No", "No phone service": "No"})

# Бинарные Yes/No -> 1/0
for c in work.columns:
    if set(work[c].dropna().unique()) <= {"Yes", "No"}:
        work[c] = work[c].map({"Yes": 1, "No": 0})
work["Gender"] = work["Gender"].map({"Male": 1, "Female": 0})

# Contract - порядковый признак
work["Contract"] = work["Contract"].map(
    {"Month-to-month": 0, "One year": 1, "Two year": 2})

# Номинальные -> one-hot (drop_first, чтобы не создавать идеальную коллинеарность)
work = pd.get_dummies(work, columns=["Internet Service", "Payment Method"],
                      drop_first=True, dtype=int)
print("Типы после кодирования:\n", work.dtypes.value_counts())
print("Остались нечисловые столбцы:", list(work.select_dtypes("object").columns))

# ----- связка между файлами (добавлено при переносе) -----
work.to_pickle(cfg.DATA_INTERIM / "telco_encoded.pkl")
