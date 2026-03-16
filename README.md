# AI vs Human Text Detection Research Prototype

Полноценный локальный исследовательский проект для диссертации по распознаванию текстов `human` / `AI-generated`.

## Что реализовано
- Модульная архитектура: загрузка данных, предобработка, признаки, обучение, эксперименты, отчеты, интерпретация.
- Поддержка классов: `human`, `ai_weak`, `ai_strong`.
- Автогенерация бинарных задач:
  - `human vs ai_all`
  - `human vs ai_weak`
  - `human vs ai_strong`
- Эксперименты A–G (G запускается, если есть класс `artificial_humanized`).
- Модели: Logistic Regression, Linear SVM, Random Forest, Gradient Boosting.
- Гибридные признаки: ручные стилометрические + TF-IDF.
- Интерпретация: feature importance / коэффициенты и текстовый вывод по гипотезе weak vs strong.
- Streamlit-интерфейс для запуска в PyCharm.

## Структура проекта

```text
.
├── main.py
├── app/
├── core/
│   ├── config.py
│   ├── data_loader.py
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
│   │   ├── ai_weak/
│   │   └── ai_strong/
│   └── processed/
├── models/
├── reports/
├── scripts/
│   ├── generate_demo_dataset.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
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
- `data/raw/ai_weak/`
- `data/raw/ai_strong/`

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

## Опциональная перплексия
В `core/features.py` есть `OptionalPerplexityEstimator`. Если `transformers` и локальная HF-модель недоступны, остальной проект работает без изменений.

## Как добавить свои данные
1. Добавьте тексты по папкам классов.
2. Повторите `python scripts/train.py` или обучение через Streamlit.
3. Смотрите сравнение в табе "Сравнение" или в `reports/results.csv`.

