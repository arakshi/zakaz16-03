from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import PathsConfig
from core.service import ProjectService

LABEL_OPTIONS = [
    "human",
    "ai_deepseek",
    "ai_gpt_fast",
    "ai_gpt_reasoning",
    "ai_yandex",
]

PROMPT_TEMPLATES = {
    "Быстрый фактологический ответ": "Сделай краткое объяснение темы: {topic}. Дай 5 пунктов и вывод.",
    "Академический абзац": "Напиши академический абзац на тему '{topic}' на русском языке, с нейтральным стилем.",
    "Сравнительный анализ": "Сравни подход A и подход B в теме '{topic}', с плюсами и минусами.",
    "Популярно-научный стиль": "Объясни тему '{topic}' для широкой аудитории понятным языком.",
}


ANALYSIS_EXAMPLES = {
    "AI пример (рассуждающий стиль)": "Если рассматривать проблему в динамике, ключевым фактором становится не сам инструмент, а контекст его внедрения: организационная культура, мотивация участников и качество обратной связи.",
    "AI пример (быстрый стиль)": "Текстовый классификатор удобно строить на сочетании TF-IDF и простых стилометрических признаков. Для проверки качества нужны отдельные сценарии экспериментов.",
    "Human пример": "Вчера я перечитал свои заметки и понял, что ошибся в формулировке гипотезы. Пришлось переписать раздел, но аргументация стала честнее и понятнее.",
}

HUMAN_TEXT_RECOMMENDATIONS = [
    "Л.Н. Толстой — 'Война и мир' (txt, docx->txt, csv[text])",
    "Ф.М. Достоевский — 'Братья Карамазовы' (txt, csv[text])",
    "А.П. Чехов — рассказы (короткие фрагменты в txt)",
    "И.С. Тургенев — 'Отцы и дети' (фрагменты 100-500 слов)",
    "Публицистика/научпоп с явным авторством (txt/csv)",
]


def get_service() -> ProjectService:
    p = PathsConfig()
    return ProjectService(
        raw_data_dir=p.data_raw,
        processed_path=p.data_processed / "merged_dataset.csv",
        models_dir=p.models_dir,
        reports_dir=p.reports_dir,
    )


def render_data_upload_tab(service: ProjectService) -> None:
    st.subheader("Сбор и загрузка датасета")
    st.caption("Можно добавить данные вручную, вставкой текста или загрузкой файлов в нужный класс.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Добавить текст вручную**")
        manual_label = st.selectbox("Класс текста", LABEL_OPTIONS, key="manual_label")
        manual_text = st.text_area("Текст для сохранения", height=170, key="manual_text")
        if st.button("Сохранить текст в data/raw", key="save_manual"):
            if not manual_text.strip():
                st.warning("Введите текст перед сохранением.")
            else:
                saved_path = service.save_text_sample(manual_text, manual_label)
                st.success(f"Сохранено: {saved_path}")

    with col2:
        st.markdown("**Загрузить файл**")
        upload_label = st.selectbox("Класс для файла", LABEL_OPTIONS, key="upload_label")
        uploaded_file = st.file_uploader("Файл txt/csv", type=["txt", "csv"], key="file_uploader")
        if st.button("Сохранить файл в data/raw", key="save_file"):
            if uploaded_file is None:
                st.warning("Сначала выберите файл.")
            else:
                saved_path = service.save_uploaded_file(uploaded_file.name, uploaded_file.getvalue(), upload_label)
                st.success(f"Файл сохранен: {saved_path}")

    st.markdown("---")
    st.markdown("**Рекомендации по human-текстам (для корпуса):**")
    for item in HUMAN_TEXT_RECOMMENDATIONS:
        st.write(f"- {item}")

    with st.expander("Примеры промптов для генерации текстов в разных ИИ"):
        st.write("Используйте шаблоны ниже, чтобы быстрее получать сопоставимые тексты в разных моделях.")
        topic = st.text_input("Тема для подстановки в шаблон", value="влияние цифровизации на образование")
        ai_target = st.selectbox("Целевой ИИ", ["DeepSeek", "GPT Быстрая", "GPT Думающая", "Яндекс ИИ"])
        template_name = st.selectbox("Шаблон промпта", list(PROMPT_TEMPLATES.keys()))
        rendered = PROMPT_TEMPLATES[template_name].format(topic=topic)
        st.code(f"[{ai_target}] {rendered}")
        st.caption("Скопируйте, отправьте в выбранный ИИ, затем вставьте ответ выше или загрузите файлом.")


    st.markdown("---")
    st.markdown("**Тестовый набор данных (быстрое наполнение):**")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("Добавляет расширенный демо-набор в классы human / ai_deepseek / ai_gpt_fast / ai_gpt_reasoning / ai_yandex.")
    with c2:
        overwrite = st.checkbox("Перезаписать demo", value=False, key="overwrite_demo")

    if st.button("Добавить тестовый датасет", key="add_demo_dataset"):
        stats = service.add_test_demo_dataset(overwrite=overwrite)
        total = sum(stats.values())
        st.success(f"Добавлено файлов: {total}")
        st.json(stats)

    st.markdown("---")
    if st.button("Собрать объединенный датасет", key="build_dataset"):
        try:
            df = service.load_and_prepare()
            st.success(f"Загружено {len(df)} текстов.")
            st.dataframe(df.head(20))
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="AI Text Detection", layout="wide")
    st.title("Детекция AI-generated текста")

    service = get_service()

    tabs = st.tabs(["1) Загрузка данных", "2) Обучение моделей", "3) Сравнение", "4) Анализ текста", "5) Отчеты"])

    with tabs[0]:
        render_data_upload_tab(service)

    with tabs[1]:
        st.subheader("Запуск обучения и экспериментов A-G")
        if st.button("Обучить и оценить"):
            with st.spinner("Обучение..."):
                artifacts = service.train_and_evaluate()
            st.success(f"Готово. Лучшая модель: {artifacts.best_model_name}")
            st.write(artifacts.hypothesis_text)
            st.dataframe(artifacts.results.sort_values(["scenario", "f1"], ascending=[True, False]))

    with tabs[2]:
        st.subheader("Сравнение результатов")
        results_path = PathsConfig().reports_dir / "results.csv"
        if results_path.exists():
            results = pd.read_csv(results_path)
            st.dataframe(results)
            img_path = PathsConfig().reports_dir / "f1_comparison.png"
            if img_path.exists():
                st.image(str(img_path), caption="F1 comparison")
        else:
            st.info("Сначала выполните обучение на вкладке 2.")

    with tabs[3]:
        st.subheader("Анализ одного текста")
        example_choice = st.selectbox("Быстрый пример текста", ["(не выбран)", *ANALYSIS_EXAMPLES.keys()])
        initial_text = ANALYSIS_EXAMPLES.get(example_choice, "")
        text = st.text_area("Вставьте текст", value=initial_text)
        st.caption("Совет: для корректной вероятности AI сначала обучите модель на сбалансированном наборе классов.")
        model_files = list(PathsConfig().models_dir.glob("best_*.joblib"))
        if model_files:
            selected_model = st.selectbox("Модель", [str(m) for m in model_files])
            if st.button("Анализировать"):
                if not text.strip():
                    st.warning("Введите текст.")
                else:
                    result = service.analyze_text(text, Path(selected_model))
                    st.metric("Вероятность AI", result["ai_probability"])
                    st.write(f"Предсказанный класс: **{result['predicted_class']}**")
                    st.caption(result["explanation"])
        else:
            st.info("Модель не найдена. Обучите её на вкладке 2.")

    with tabs[4]:
        st.subheader("Файлы отчетов")
        rep_dir = PathsConfig().reports_dir
        for file in sorted(rep_dir.glob("*")):
            st.write(f"- {file.name}")


def _is_running_with_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _run_via_streamlit_cli() -> None:
    from streamlit.web import cli as stcli

    script_path = str(Path(__file__).resolve())
    print("[INFO] main.py запущен напрямую. Перезапускаю через 'streamlit run'.")
    sys.argv = ["streamlit", "run", script_path]
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    if _is_running_with_streamlit():
        main()
    else:
        _run_via_streamlit_cli()
