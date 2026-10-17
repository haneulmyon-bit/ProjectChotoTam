"""Блокнот: спецификация признаков (Data Dictionary).

Вход: data/interim/reduced_df.pkl. Результат: docs/data_dictionary.csv
"""
# ----- связка между файлами (добавлено при переносе) -----
import pandas as pd
from main_project import config as cfg
cfg.DOCS.mkdir(parents=True, exist_ok=True)

reduced_df = pd.read_pickle(cfg.DATA_INTERIM / "reduced_df.pkl")

# ===== КОД ИЗ БЛОКНОТА =====


raw_df = pd.read_excel(str(cfg.DATA_RAW))
raw_df.columns = raw_df.columns.str.strip()

# Спецификация для исходного датасета
original_features_info = {
    "CustomerID": "Уникальный идентификатор клиента",
    "Count": "Технический счетчик (всегда равен 1)",
    "Country": "Страна проживания клиента",
    "State": "Штат проживания клиента",
    "City": "Город проживания клиента",
    "Zip Code": "Почтовый индекс",
    "Lat Long": "Координаты (строка)",
    "Latitude": "Широта местоположения",
    "Longitude": "Долгота местоположения",
    "Gender": "Пол клиента",
    "Senior Citizen": "Является ли клиент пожилым гражданином (Yes/No)",
    "Partner": "Наличие супруга/партнера (Yes/No)",
    "Dependents": "Наличие иждивенцев (Yes/No)",
    "Tenure Months": "Количество месяцев, в течение которых клиент пользуется услугами",
    "Phone Service": "Подключена ли телефонная связь (Yes/No)",
    "Multiple Lines": (
        "Подключено ли несколько телефонных линий (Yes/No/No phone service)"
        ),
    "Internet Service": "Тип интернет-провайдера (DSL/Fiber optic/No)",
    "Online Security": "Услуга онлайн-безопасности (Yes/No/No internet service)",
    "Online Backup": "Услуга резервного копирования (Yes/No/No internet service)",
    "Device Protection": "Страховка оборудования (Yes/No/No internet service)",
    "Tech Support": "Услуга технической поддержки (Yes/No/No internet service)",
    "Streaming TV": "Услуга стримингового телевидения (Yes/No/No internet service)",
    "Streaming Movies": "Услуга стриминга кино (Yes/No/No internet service)",
    "Contract": "Тип контракта (Month-to-month/One year/Two year)",
    "Paperless Billing": "Электронный биллинг (Yes/No)",
    "Payment Method": "Способ оплаты услуг",
    "Monthly Charges": "Ежемесячный платеж клиента",
    "Total Charges": "Общая сумма платежей за все время",
    "Churn Label": "Текстовая метка оттока (Yes/No)",
    "Churn Value": "Бинарная целевая переменная оттока (1 - ушел, 0 - остался)",
    "Churn Score": "Показатель вероятности оттока от внутренней системы (0-100)",
    "CLTV": "Прогнозируемая пожизненная ценность клиента",
    "Churn Reason": "Причина ухода клиента (заполнено только для ушедших)"
}

# Строим интерактивную таблицу соответствия признаков
rows_spec = []
for col in raw_df.columns:
    desc = original_features_info.get(col, "-")
    old_type = str(raw_df[col].dtype)

    # Проверяем, остался ли признак в финальном датасете (после VIF-фильтрации)
    in_final = col in reduced_df.columns
    new_type = str(reduced_df[col].dtype) if in_final else "Удален (или исключен)"

    # Формируем комментарий по трансформации
    comment = ""
    if not in_final:
        if col in [
            "CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long"
        ]:
            comment = (
                "Исключен (высокая кардинальность / неинформативен / "
                "дублирует координаты)"
            )
        elif col in ["Churn Label", "Churn Score", "CLTV", "Churn Reason"]:
            comment = (
                "Исключен (утечка целевой переменной / доступно только постфактум)"
            )
        elif col in ["Monthly Charges", "Total Charges"]:
            comment = "Удален в ходе VIF-фильтрации из-за высокой мультиколлинеарности"
    else:
        if col == "Gender":
            comment = "Кодирование: 1 - Male, 0 - Female"
        elif col in [
            "Senior Citizen", "Partner", "Dependents",
            "Phone Service", "Paperless Billing", "Churn Value"
        ]:
            comment = "Кодирование: 1 - Yes, 0 - No"
        elif col == "Contract":
            comment = (
                "Порядковое кодирование: 0 - Month-to-month, 1 - One year, 2 - Two year"
            )
        elif col in [
            "Online Security", "Online Backup", "Device Protection", "Tech Support",
            "Streaming TV", "Streaming Movies", "Multiple Lines"
        ]:
            comment = "Кодирование: 1 - Yes, 0 - No/No service (объединены)"
        elif col in ["Latitude", "Longitude"]:
            comment = "Сохранен как вещественный признак, тип приведен к float32"

    # Дополнительно отметим OHE-колонки, которых не было в исходном датасете напрямую
    rows_spec.append({
        "Название атрибута (исходное)": col,
        "Описание": desc,
        "Исходный формат": old_type,
        "Финальный формат": new_type,
        "Комментарий / Трансформация": comment
    })

# Добавим в спецификацию созданные через One-Hot Encoding признаки
source = "Internet Service" if "Internet" in col else "Payment Method"
description = (
    f"Бинарный флаг для категории '{col}'\n"
    f"(Создан из {source})"
)

for col in reduced_df.columns:
    if col not in raw_df.columns:
        rows_spec.append({
            "Название атрибута (исходное)": f"{col}",
            "Описание": description,
            "Исходный формат": "-",
            "Финальный формат": str(reduced_df[col].dtype),
            "Комментарий / Трансформация": "One-Hot кодирование (1 - да, 0 - нет)"
        })

df_spec = pd.DataFrame(rows_spec)
pd.set_option('display.max_rows', len(df_spec))
print(df_spec.to_string())

# ----- связка между файлами (добавлено при переносе) -----
df_spec.to_csv(cfg.DOCS / "data_dictionary.csv", index=False, encoding="utf-8-sig")
