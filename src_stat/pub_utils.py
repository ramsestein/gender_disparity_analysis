"""
pub_utils.py
============
Módulo de compatibilidad: re-exporta desde core/utils.py todos los
símbolos que los scripts p01-p23 esperan encontrar en 'pub_utils'.
"""

from core.utils import (
    load_data,
    cohens_d,
    cramers_v_calc,
    get_magnitude,
    NUMERIC_VARS,
    CATEGORICAL_VARS,
    BASE_PATH,
    OUTPUT_DIR,
    FIG_DIR,
)

__all__ = [
    "load_data",
    "cohens_d",
    "cramers_v_calc",
    "get_magnitude",
    "NUMERIC_VARS",
    "CATEGORICAL_VARS",
    "BASE_PATH",
    "OUTPUT_DIR",
    "FIG_DIR",
]
