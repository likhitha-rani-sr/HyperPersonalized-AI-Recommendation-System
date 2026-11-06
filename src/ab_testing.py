"""
A/B testing utilities for HyperPersonalizedAIRecommendationSystem.

This is an importable Python module. It intentionally contains NO Streamlit UI
widgets and does not launch a dashboard when imported.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def run_ab_test(
    n_users: int = 500,
    seed: int = 42,
    control_ctr: float = 0.08,
    treatment_ctr: float = 0.12,
    control_add_to_cart_rate: float = 0.04,
    treatment_add_to_cart_rate: float = 0.07,
    save_results: bool = True,
) -> dict[str, Any]:
    """Run a SIMULATED popularity-vs-hybrid A/B test and return its results."""
    if n_users < 2:
        raise ValueError("n_users must be at least 2.")

    for name, rate in (
        ("control_ctr", control_ctr),
        ("treatment_ctr", treatment_ctr),
        ("control_add_to_cart_rate", control_add_to_cart_rate),
        ("treatment_add_to_cart_rate", treatment_add_to_cart_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    control_click = rng.binomial(1, control_ctr, size=n_users)
    treatment_click = rng.binomial(1, treatment_ctr, size=n_users)
    control_cart = rng.binomial(1, control_add_to_cart_rate, size=n_users)
    treatment_cart = rng.binomial(1, treatment_add_to_cart_rate, size=n_users)

    observations = pd.DataFrame({
        "user_number": np.arange(1, n_users + 1),
        "control_method": "Popularity baseline",
        "treatment_method": "Hybrid recommendation",
        "control_clicked": control_click,
        "treatment_clicked": treatment_click,
        "control_added_to_cart": control_cart,
        "treatment_added_to_cart": treatment_cart,
    })

    c_ctr, t_ctr = float(control_click.mean()), float(treatment_click.mean())
    c_cart, t_cart = float(control_cart.mean()), float(treatment_cart.mean())
    ctr_test = ttest_ind(treatment_click, control_click, equal_var=False)
    cart_test = ttest_ind(treatment_cart, control_cart, equal_var=False)

    def relative_lift(control: float, treatment: float) -> float | None:
        return ((treatment - control) / control * 100.0) if control > 0 else None

    summary: dict[str, Any] = {
        "experiment_name": "Simulated A/B test: Popularity vs Hybrid",
        "simulation_only": True,
        "note": (
            "Simulated observations based on configured probabilities; these are "
            "not real user behaviour or production experiment results."
        ),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": int(seed),
        "n_users_per_group": int(n_users),
        "control": {
            "method": "Popularity baseline",
            "ctr": c_ctr,
            "add_to_cart_rate": c_cart,
        },
        "treatment": {
            "method": "Hybrid recommendation",
            "ctr": t_ctr,
            "add_to_cart_rate": t_cart,
        },
        "ctr_absolute_lift_percentage_points": (t_ctr - c_ctr) * 100.0,
        "ctr_relative_lift_percent": relative_lift(c_ctr, t_ctr),
        "add_to_cart_absolute_lift_percentage_points": (t_cart - c_cart) * 100.0,
        "add_to_cart_relative_lift_percent": relative_lift(c_cart, t_cart),
        "ctr_p_value": float(ctr_test.pvalue) if np.isfinite(ctr_test.pvalue) else None,
        "add_to_cart_p_value": (
            float(cart_test.pvalue) if np.isfinite(cart_test.pvalue) else None
        ),
    }

    csv_path = summary_path = None
    if save_results:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = MODELS_DIR / f"ab_test_observations_{stamp}.csv"
        json_file = MODELS_DIR / f"ab_test_summary_{stamp}.json"
        observations.to_csv(csv_file, index=False)
        json_file.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        csv_path, summary_path = str(csv_file), str(json_file)

    return {
        "summary": summary,
        "observations": observations,
        "csv_path": csv_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    # Importing this file does not create Streamlit widgets.
    result = run_ab_test()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"\nCSV: {result['csv_path']}")
    print(f"Summary JSON: {result['summary_path']}")
