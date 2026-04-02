from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

HYPOTHESIS_F1_THRESHOLD = 0.03


def extract_feature_importance(model, top_k: int = 15) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]
    names = list(pre.get_feature_names_out())

    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        imp = pd.DataFrame({"feature": names, "weight": coefs})
        imp["abs_weight"] = imp["weight"].abs()
        return imp.sort_values("abs_weight", ascending=False).head(top_k)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        imp = pd.DataFrame({"feature": names, "importance": importances})
        return imp.sort_values("importance", ascending=False).head(top_k)

    return pd.DataFrame({"feature": ["unknown"], "importance": [np.nan]})


def build_hypothesis_conclusion(results_df: pd.DataFrame) -> str:
    weak = results_df[results_df["scenario"] == "human_vs_ai_weak"]["f1"].mean()
    strong = results_df[results_df["scenario"] == "human_vs_ai_strong"]["f1"].mean()
    all_mix = results_df[results_df["scenario"] == "human_vs_ai_all"]["f1"].mean()

    if np.isnan(weak) or np.isnan(strong):
        return "Недостаточно данных для проверки гипотезы по weak/strong моделям."

    delta = weak - strong
    if delta > HYPOTHESIS_F1_THRESHOLD:
        return (
            f"Средний F1 для weak={weak:.3f}, для strong={strong:.3f} (разница {delta:.3f}). "
            "В текущем эксперименте тексты weak-моделей детектируются лучше, что поддерживает гипотезу."
        )
    if delta < -HYPOTHESIS_F1_THRESHOLD:
        return (
            f"Средний F1 для weak={weak:.3f}, для strong={strong:.3f} (разница {delta:.3f}). "
            "В текущем эксперименте strong-модели детектируются не хуже; гипотеза не подтверждается."
        )
    return (
        f"Средний F1 для weak={weak:.3f}, для strong={strong:.3f}, для ai_all={all_mix:.3f}. "
        "Разница невелика, поэтому гипотеза о заметном ухудшении детекции strong-моделей пока не подтверждена статистически значимо."
    )


def explain_single_prediction(feature_row: Dict[str, float]) -> str:
    signals: List[str] = []
    if feature_row.get("template_ratio", 0) > 0.1:
        signals.append("заметная шаблонность")
    if feature_row.get("type_token_ratio", 1) < 0.45:
        signals.append("низкое лексическое разнообразие")
    if feature_row.get("smoothness", 0) > 0.7:
        signals.append("очень гладкая структура предложений")
    if feature_row.get("punct_density_100w", 0) < 3:
        signals.append("низкая пунктуационная насыщенность")

    if not signals:
        return "Сильных аномалий по ручным признакам не обнаружено; решение основано на комбинации TF-IDF и стилометрии."
    return "Ключевые признаки: " + ", ".join(signals) + "."
