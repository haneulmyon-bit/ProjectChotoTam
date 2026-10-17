"""Блокнот, пп. 0-4: загрузка, пропуски, ошибки, выбросы, дубликаты.

Результат: data/interim/telco_clean.pkl
"""
# ----- связка между файлами (добавлено при переносе) -----
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main_project import config as cfg
from main_project.config import TARGET

cfg.FIGURES.mkdir(parents=True, exist_ok=True)
cfg.DATA_INTERIM.mkdir(parents=True, exist_ok=True)

# ===== КОД ИЗ БЛОКНОТА =====

# --- 0. Просмотр исходных данных ---

df = pd.read_excel(str(cfg.DATA_RAW))
df.columns = df.columns.str.strip()
print("=" * 70, "\n0. ОБЩАЯ ИНФОРМАЦИЯ\n", "=" * 70)

# Краткое описание датасета
df.info()

# Выводит сводную статистику по столбцам
print(df.describe(include="all").T)

# --- 1. Пропущенные значения ---

# У числовых типов данных нет NaN значений, поэтому их не нужно обрабатывать
print("\n" + "=" * 70, "\n1. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ\n", "=" * 70)

# «Скрытые» пропуски: пустые строки / пробелы среди типа «object»
obj_cols = df.select_dtypes("object").columns
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip().replace(
        {"": np.nan, "nan": np.nan, "None": np.nan}
    )

miss = pd.DataFrame({"n_missing": df.isna().sum(),
                     "pct": (df.isna().mean() * 100).round(2)})
print("Сводка по пропущенным значениям (кол-во пропусков и их доля от всех значений):")
print(miss[miss.n_missing > 0])

# Привидение к числу «Total Charges» (пустые станут NaN)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
bad_tc = df[df["Total Charges"].isna()]
print(f"\nTotal Charges пустой у {len(bad_tc)} строк; Tenure Months у них:",
      bad_tc["Tenure Months"].unique())

# Это клиенты с tenure = 0 (ещё не было счетов) -> логично заполнить 0
df.loc[df["Total Charges"].isna() & (df["Tenure Months"] == 0), "Total Charges"] = 0.0

# Churn Reason пуст у не ушедших клиентов - это структурный пропуск, не ошибка
print("\nChurn Reason пуст у ушедших клиентов:",
      df.loc[df[TARGET] == 1, "Churn Reason"].isna().sum(), "(ожидается 0)")

print("=" * 70)
df.info()

# --- 2. Явные и логические ошибки ---
print("\n" + "=" * 70, "\n2. ЯВНЫЕ ОШИБКИ И ОПИСКИ\n", "=" * 70)
cat_cols = [c for c in df.select_dtypes("object").columns
            if c not in ["CustomerID", "Lat Long", "City", "Churn Reason"]]
for c in cat_cols:
    print(f"{c}: {sorted(df[c].dropna().unique())}")

# 2.1 Единообразие написания категорий (регистр, пробелы)
isRegistryError = False
for c in cat_cols:
    raw = df[c].dropna()
    if raw.str.lower().nunique() != raw.nunique():
        isRegistryError = True
        print("\nВНИМАНИЕ: разный регистр в", c)
if not isRegistryError:
    print("\nВ категориальных признаках не найдено ошибок регистра")

# 2.2 Логические проверки
checks = {
    "Tenure Months < 0": (df["Tenure Months"] < 0).sum(),
    "Monthly Charges <= 0": (df["Monthly Charges"] <= 0).sum(),
    "Total Charges < 0": (df["Total Charges"] < 0).sum(),
    "Churn Label != Churn Value":
    (df["Churn Label"].map({"Yes": 1, "No": 0}) != df[TARGET]).sum(),
    "Latitude вне [-90, 90]": (~df["Latitude"].between(-90, 90)).sum(),
    "Longitude вне [-180, 180]": (~df["Longitude"].between(-180, 180)).sum(),
    "Churn Score вне [0, 100]": (~df["Churn Score"].between(0, 100)).sum(),
}

# Согласованность Total Charges ≈ Tenure * Monthly (допуск 50%)
approx = df["Tenure Months"] * df["Monthly Charges"]
diff_mask = (
    (df["Total Charges"] - approx).abs() > 0.5 * approx.clip(lower=1)
) & (df["Tenure Months"] > 0)
checks["Total Charges сильно != Tenure*Monthly (> 50%)"] = diff_mask.sum()

# Согласованность зависимых категорий
addons = ["Online Security", "Online Backup", "Device Protection",
          "Tech Support", "Streaming TV", "Streaming Movies"]
checks["Addon != 'No internet service' при Internet Service = No"] = sum(
    ((df["Internet Service"] == "No") & (df[a] != "No internet service")).sum()
    for a in addons
)
checks["Multiple Lines != 'No phone service' при Phone Service = No"] = (
    (df["Phone Service"] == "No") & (df["Multiple Lines"] != "No phone service")).sum()

# Lat Long согласуется с Latitude / Longitude
ll = df["Lat Long"].str.split(",", expand=True).astype(float)
lat_mismatch = (ll[0] - df["Latitude"]).abs() > 1e-4
lon_mismatch = (ll[1] - df["Longitude"]).abs() > 1e-4
checks["Lat Long != Latitude/Longitude"] = (lat_mismatch | lon_mismatch).sum()
print()
print(pd.Series(checks).to_string())

# Константные столбцы (не несут информации)
const = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
print("\nКонстантные столбцы:", const)

# --- 3. Выбросы ---


print("\n" + "=" * 70, "\n3. ВЫБРОСЫ (правило 1.5*IQR и |z|>3)\n", "=" * 70)
num_cols = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
rows = []
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    # z показывает, на сколько стандартных отклонений значение отстоит от среднего
    z = (df[c] - df[c].mean()) / df[c].std()
    rows.append({"feature": c, "lower": round(lo, 2), "upper": round(hi, 2),
                 "n_IQR_outliers": int(((df[c] < lo) | (df[c] > hi)).sum()),
                 "n_|z|>3": int((z.abs() > 3).sum())})
print(pd.DataFrame(rows).to_string(index=False))

fig, axes = plt.subplots(1, len(num_cols), figsize=(4 * len(num_cols), 4))
for ax, c in zip(axes, num_cols):
    sns.boxplot(y=df[c], ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.savefig(cfg.FIGURES / "boxplots.png", dpi=120)
plt.show()
plt.close()

# --- 4. Дубликаты ---
print("\n" + "=" * 70, "\n4. ДУБЛИКАТЫ\n", "=" * 70)
print("Полностью дублирующихся строк:", df.duplicated().sum())
print("Дубликатов CustomerID:", df["CustomerID"].duplicated().sum())
cols_wo_id = [c for c in df.columns if c != "CustomerID"]
print(
    "Дубликатов по всем признакам кроме CustomerID:",
    df.duplicated(subset=cols_wo_id).sum()
)
df = df.drop_duplicates().reset_index(drop=True)

# ----- связка между файлами (добавлено при переносе) -----
df.to_pickle(cfg.DATA_INTERIM / "telco_clean.pkl")
