# Артефакты

| Артефакт | Назначение |
|---|---|
| [`final.py`](final.py) | Единый код обучения финальных моделей и инференса без подбора гиперпараметров |
| [`model_config.json`](model_config.json) | Зафиксированные гиперпараметры, веса ансамбля и сценарий маршрута 5 |
| [`requirements.txt`](requirements.txt) | Зависимости окружения |
| [`benchmark_results.json`](benchmark_results.json) | Машиночитаемые замеры end-to-end времени, пропускной способности и пикового RSS |
| [`submission_selective.csv`](submission_selective.csv) | Лучший конкурсный прогноз на ноябрь-декабрь 2025 |
| [`submission_selective_2026.csv`](submission_selective_2026.csv) | Сценарный почасовой прогноз на весь 2026 год |
| [Chronos-2 weights](https://huggingface.co/amazon/chronos-2) | Внешний предобученный артефакт 120M параметров; автоматически кешируется при первом запуске |

LightGBM, SARIMAX и CatBoost обучаются из локальных файлов при каждом запуске. Отдельные бинарные веса не включены, чтобы исключить несовместимость версий и сохранить проверяемую воспроизводимость. План сериализации production-артефактов приведён в [`limitations_and_roadmap.md`](limitations_and_roadmap.md).
