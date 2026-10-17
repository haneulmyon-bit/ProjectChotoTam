"""Блокнот, пп. 6-7 и итог: мультиколлинеарность, оптимизация типов,
два готовых датасета.

Вход: data/interim/telco_encoded.pkl
Результат: data/processed/telco_encoded_full.csv, telco_encoded_vif_reduced.csv
"""
# ----- связка между файлами (добавлено при переносе) -----
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main_project import config as cfg
from main_project.config import TARGET
cfg.FIGURES.mkdir(parents=True, exist_ok=True)
cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

work = pd.read_pickle(cfg.DATA_INTERIM / "telco_encoded.pkl")

# ===== КОД ИЗ БЛОКНОТА =====

# --- 6. Мультиколлинеарность ---


print("\n" + "=" * 70, "\n6. МУЛЬТИКОЛЛИНЕАРНОСТЬ\n", "=" * 70)
X = work.drop(columns=[TARGET])

corr = X.corr()
plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True)
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.savefig(cfg.FIGURES / "corr_matrix.png", dpi=120)
plt.close()

pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack().rename("r").reset_index())
pairs.columns = ["feature_1", "feature_2", "r"]
print("Пары с |r| > 0.7:")
print(pairs[pairs.r.abs() > 0.7]
      .sort_values("r", key=abs, ascending=False)
      .to_string(index=False))


def calc_vif(frame):
    Xc = add_constant(frame.astype(float))
    return pd.Series([variance_inflation_factor(Xc.values, i)
                      for i in range(1, Xc.shape[1])],
                     index=frame.columns).sort_values(ascending=False)


vif = calc_vif(X)
print("\nVIF (>10 - сильная мультиколлинеарность):")
print(vif.round(2).to_string())

# Итеративное удаление признаков с VIF > 10
X_red = X.copy()
dropped = []
while True:
    v = calc_vif(X_red)
    if v.iloc[0] <= 10:
        break
    dropped.append((v.index[0], round(v.iloc[0], 1)))
    X_red = X_red.drop(columns=v.index[0])
print("\nУдалены при VIF > 10:", dropped)
print("Итоговые VIF:\n", calc_vif(X_red).round(2).to_string())

# --- 7. Изменение размерности типа данных ---
# Объединяем отобранные после VIF признаки и целевую переменную
reduced_df = pd.concat([X_red, work[TARGET]], axis=1)

# Оптимизация типов данных для уменьшения размера в памяти
for col in reduced_df.select_dtypes(include=['integer']).columns:
    reduced_df[col] = pd.to_numeric(reduced_df[col], downcast="integer")

for col in reduced_df.select_dtypes(include=['float']).columns:
    reduced_df[col] = pd.to_numeric(reduced_df[col], downcast="float")

# Сохранение оптимизированных датасетов
work.to_csv(cfg.DATA_PROCESSED / "telco_encoded_full.csv", index=False)
reduced_df.to_csv(cfg.DATA_PROCESSED / "telco_encoded_vif_reduced.csv", index=False)

print("Размерность итогового датасета:", reduced_df.shape)
reduced_df.info()
print(
    "\nБаланс классов целевой переменной:\n",
    work[TARGET].value_counts(normalize=True).round(3)
)

# --- Обработанные датасеты для дальнейшей работы ---
# Сохраняем датасеты БЕЗ масштабирования,
# чтобы избежать утечки данных (Data Leakage) при последующей валидации моделей.
# MinMaxScaler применяется строго к тренировочным выборкам после разбиения.

target_column = "Churn Value"

# 1. Полный датасет (все закодированные признаки, включая коллинеарные)
full_df = work.copy()

# 2. Редуцированный датасет (после удаления мультиколлинеарности через VIF)
vif_reduced_df = reduced_df.copy()

# Сохраняем итоговые чистые датасеты в исходном масштабе
full_df.to_csv(cfg.DATA_PROCESSED / "telco_encoded_full.csv", index=False)
vif_reduced_df.to_csv(cfg.DATA_PROCESSED / "telco_encoded_vif_reduced.csv", index=False)

print("=== Датасеты для передачи коллегам успешно сохранены! ===")
print(f"1. Полный датасет (Full): {full_df.shape} (для древесных моделей)")
print(f"2. Редуцированный датасет (VIF-Reduced): "
      f"{vif_reduced_df.shape} (для линейных моделей и SPlit/SPXY)")
print(
    f"Баланс классов целевой переменной: "
    f"{full_df[target_column].value_counts(normalize=True).round(3).to_dict()}"
)

# ----- связка между файлами (добавлено при переносе) -----
reduced_df.to_pickle(cfg.DATA_INTERIM / "reduced_df.pkl")
