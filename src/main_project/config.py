"""Константы проекта: пути, целевая переменная, списки колонок.
"""
from pathlib import Path

# --- Пути (ROOT = корень репозитория) ---
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
DOCS = ROOT / "docs"

# --- Параметры эксперимента ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Целевая переменная и колонки ---
TARGET = "Churn Value"

# Признаки с утечкой целевой переменной (известны только постфактум)
LEAKAGE = ["Churn Label", "Churn Score", "CLTV", "Churn Reason"]

# Служебные / неинформативные колонки
DROP_COLS = ["CustomerID", "Count", "Country",
             "State", "City", "Zip Code", "Lat Long"]

# Дополнительные услуги (значения Yes / No / No internet service)
ADDONS = ["Online Security", "Online Backup", "Device Protection",
          "Tech Support", "Streaming TV", "Streaming Movies"]

# Числовые признаки, которые нужно масштабировать
NUMERIC_COLS = ["Tenure Months", "Monthly Charges",
                "Total Charges", "Latitude", "Longitude"]

# Числовые колонки для анализа выбросов
OUTLIER_COLS = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]

# Кодирование
CONTRACT_ORDER = {"Month-to-month": 0, "One year": 1, "Two year": 2}
ONE_HOT_COLS = ["Internet Service", "Payment Method"]

# Порог мультиколлинеарности
VIF_THRESHOLD = 10

# --- Описания исходных признаков (для data dictionary) ---
FEATURE_DESCRIPTIONS = {
    "CustomerID": "Уникальный идентификатор клиента",
    "Count": "Технический счетчик",
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
    "Churn Reason": "Причина ухода клиента (заполнено только для ушедших)",
}
