"""Запускает подготовку данных целиком: python scripts/prepare_data.py

Шаги выполняются строго по порядку. Результат -- два датасета в data/processed/.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STEPS = ["data", "preprocessing", "features", "data_dictionary"]

for step in STEPS:
    print(f"\n########## {step} ##########")
    runpy.run_module(f"main_project.{step}", run_name="__main__")
