# AI vs Human Text Detection

Полноценный локальный исследовательский проект для диссертации по распознаванию текстов `human` / `AI-generated`.

## Что реализовано
- Модульная архитектура: загрузка данных, предобработка, признаки, обучение, эксперименты, отчеты, интерпретация.
- Поддержка классов: `human`, `ai_deepseek`, `ai_gpt_fast`, `ai_gpt_reasoning`, `ai_yandex`, `artificial_humanized` (и совместимость с legacy `ai_weak`/`ai_strong`).
- Автогенерация бинарных задач:
  - `human vs ai_all`
  - `human vs ai_weak`
  - `human vs ai_strong`
- Эксперименты A–G (G запускается, если есть класс `artificial_humanized`).
- Модели: Logistic Regression, Linear SVM, Random Forest, Histogram-based Gradient Boosting.
- Гибридные признаки: ручные стилометрические + TF-IDF.
- Интерпретация: feature importance / коэффициенты и текстовый вывод по гипотезе weak vs strong.
- Streamlit-интерфейс для запуска в PyCharm.

## Структура проекта

```text
.
├── main.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── demo_data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modeling.py
│   ├── experiments.py
│   ├── interpretation.py
│   ├── reporting.py
│   └── service.py
├── data/
│   ├── raw/
│   │   ├── human/
│   │   ├── ai_deepseek/
│   │   ├── ai_gpt_fast/
│   │   ├── ai_gpt_reasoning/
│   │   ├── ai_yandex/
│   │   └── artificial_humanized/
│   └── processed/
├── models/
├── reports/
├── scripts/
│   ├── generate_demo_dataset.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py
│   ├── test_feature_extractor.py
│   ├── test_run_experiment.py
│   ├── test_data_loader.py
│   ├── test_cross_generalization.py
│   ├── test_modeling_alignment.py
│   ├── test_service.py
│   ├── test_interpretation.py
│   └── test_reporting.py
├── requirements.txt
└── .gitignore
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Быстрый старт (Windows + PyCharm)
1. Откройте проект как папку в PyCharm.
2. Выберите Python interpreter из виртуального окружения.
3. Установите зависимости из `requirements.txt`.
4. Сгенерируйте демо-данные:
   ```bash
   python scripts/generate_demo_dataset.py
   ```
5. Запустите приложение:
   ```bash
   streamlit run main.py
   ```

> Также поддерживается запуск кнопкой Run в PyCharm (`python main.py`):
> скрипт автоматически перезапустит себя через `streamlit run`.

## Запуск через терминал
### Обучение и эксперименты
```bash
python scripts/train.py
```

### Предсказание для одного текста
```bash
python scripts/predict.py --text "Ваш текст здесь"
```

## Формат данных
Расположите тексты в:
- `data/raw/human/`
- `data/raw/ai_deepseek/`
- `data/raw/ai_gpt_fast/`
- `data/raw/ai_gpt_reasoning/`
- `data/raw/ai_yandex/`
- `data/raw/artificial_humanized/`

Допустимы `txt` и `csv` (в csv нужен столбец `text`).

## Описание экспериментов
- **A**: `human vs ai_all`
- **B**: `human vs ai_weak`
- **C**: `human vs ai_strong`
- **D**: train on `human+ai_weak`, test on `human+ai_strong`
- **E**: train on `human+ai_strong`, test on `human+ai_weak`
- **F**: устойчивость на длинах текста (`<=100`, `100-300`, `300+` слов)
- **G**: `humanized_ai_test` (если есть класс `artificial_humanized`)

## Метрики и артефакты
Сохраняются в `reports/`:
- `results.csv`
- `results.json`
- `f1_comparison.png`
- `feature_importance.csv`
- `hypothesis_conclusion.txt`

## Интерпретация результатов
- Линейные модели: коэффициенты признаков.
- Деревья: feature importance.
- Отдельный автоматический вывод по гипотезе “weak детектируется легче, чем strong”.
- Вывод строится **только** по фактическим метрикам.

## Тестирование

```bash
python -m pytest tests/ -q
```

## Как добавить свои данные
1. Добавьте тексты по папкам классов.
2. Повторите `python scripts/train.py` или обучение через Streamlit.
3. Смотрите сравнение в табе "Сравнение" или в `reports/results.csv`.

## Дополнительно в интерфейсе
- Вкладка загрузки поддерживает: 
  - вставку текста с выбором класса;
  - загрузку txt/csv файла с выбором класса;
  - примеры промптов для DeepSeek / GPT Быстрая / GPT Думающая / Яндекс ИИ;
  - рекомендации по human-корпусу (классическая литература и форматы);
  - кнопка добавления расширенного тестового датасета (human / DeepSeek / GPT Fast / GPT Reasoning / Yandex AI / artificial_humanized).
- вкладка анализа содержит готовые примеры human/AI текста для быстрого теста вероятности.
