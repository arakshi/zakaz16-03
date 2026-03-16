from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.config import PathsConfig


HUMAN_TEXTS = [
    "Сегодня я снова опоздал на автобус, поэтому пришлось идти пешком до университета. По дороге встретил старого друга и мы обсудили планы на семестр.",
    "Вечером я готовил ужин и параллельно слушал лекцию. Конспект получился сумбурным, зато идеи для исследования стали понятнее.",
]

AI_DEEPSEEK_TEXTS = [
    "Данный текст представляет собой общее описание ситуации. Следовательно, можно сделать вывод о важности анализа.",
    "Необходимо подчеркнуть, что актуальность темы определяется многими причинами. Поэтому исследование важно для практики.",
]

AI_GPT_FAST_TEXTS = [
    "Технологии ускоряют обмен знаниями, но одновременно повышают требования к критическому мышлению и проверке источников.",
    "На практике хорошие результаты дает сочетание коротких регулярных занятий и проектной работы с обратной связью.",
]

AI_GPT_REASONING_TEXTS = [
    "Если рассматривать проблему в динамике, то ключевым фактором становится не сам инструмент, а контекст его внедрения: организационная культура, мотивация участников и качество обратной связи.",
    "Наиболее устойчивые выводы получаются при сопоставлении нескольких объяснений одной и той же тенденции, а также при явной фиксации ограничений методики.",
]

AI_YANDEX_TEXTS = [
    "Цифровые сервисы обучения помогают быстрее находить материалы, однако без методической структуры они не гарантируют рост качества знаний.",
    "Эффективность образовательной платформы зависит не только от контента, но и от сценариев взаимодействия преподавателя и студента.",
]


def write_group(texts: list[str], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts, start=1):
        (folder / f"sample_{i}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    paths = PathsConfig()
    write_group(HUMAN_TEXTS, paths.data_raw / "human")
    write_group(AI_DEEPSEEK_TEXTS, paths.data_raw / "ai_deepseek")
    write_group(AI_GPT_FAST_TEXTS, paths.data_raw / "ai_gpt_fast")
    write_group(AI_GPT_REASONING_TEXTS, paths.data_raw / "ai_gpt_reasoning")
    write_group(AI_YANDEX_TEXTS, paths.data_raw / "ai_yandex")
    print("Demo dataset created in data/raw")
