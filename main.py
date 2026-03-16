from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import PathsConfig
from core.service import ProjectService


def get_service() -> ProjectService:
    p = PathsConfig()
    return ProjectService(
        raw_data_dir=p.data_raw,
        processed_path=p.data_processed / "merged_dataset.csv",
        models_dir=p.models_dir,
        reports_dir=p.reports_dir,
    )


def main() -> None:
    st.set_page_config(page_title="AI Text Detection Research", layout="wide")
    st.title("Исследовательский прототип: детекция AI-generated текста")

    service = get_service()

    tabs = st.tabs(["1) Загрузка данных", "2) Обучение моделей", "3) Сравнение", "4) Анализ текста", "5) Отчеты"])

    with tabs[0]:
        st.subheader("Загрузка данных из data/raw")
        if st.button("Собрать датасет"):
            try:
                df = service.load_and_prepare()
                st.success(f"Загружено {len(df)} текстов.")
                st.dataframe(df.head(20))
            except Exception as exc:
                st.error(str(exc))

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
        text = st.text_area("Вставьте текст")
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
    """Return True when script is executed by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _run_via_streamlit_cli() -> None:
    """Support launching from PyCharm as `python main.py` without warnings."""
    from streamlit.web import cli as stcli

    script_path = str(Path(__file__).resolve())
    print(
        "[INFO] main.py запущен напрямую. Перезапускаю приложение корректно через 'streamlit run'."
    )
    sys.argv = ["streamlit", "run", script_path]
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    if _is_running_with_streamlit():
        main()
    else:
        _run_via_streamlit_cli()
