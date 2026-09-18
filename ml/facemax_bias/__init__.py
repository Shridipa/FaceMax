"""FaceMaxx bias & robustness evaluation toolkit.

Public API:
    generate_report(rows)   -> BiasReport (.text dashboard, .as_dict(), .save_json)
    generate_rows(...)      -> synthetic demo rows with planted biases
    leakage_check, grouped_split, group_kfold, validate, subgroup_performance
"""
from .demo import generate_rows, demo_summary
from .distribution import distribution, flag_underrepresented, imbalance_ratio
from .labels import (
    agreement_distribution,
    agreement_score,
    ambiguous_rows,
    label_source_accuracy,
    majority_label,
    seasonal_boundary_stats,
)
from .leakage import leakage_check, grouped_split, split_by_person, split_stats
from .report import BiasReport, generate_report
from .robustness import (
    condition_stability,
    makeup_filter_risk,
    person_flip_rate,
    quality_bucket_performance,
    recommend_quality_floor,
)
from .schema import SchemaReport, validate
from .splits import group_kfold, fold_sizes
from .subgroups import fairness_gaps, subgroup_performance, worst_best, verdict
from .vocab import (
    CHROMA_ORDER,
    COLUMNS,
    CONTRAST_ORDER,
    LIGHTING_ORDER,
    MAKEUP_ORDER,
    QUALITY_ORDER,
    SEASON_ORDER,
    SKIN_DEPTH_ORDER,
    SOURCE_ORDER,
    UNDERTONE_ORDER,
    normalize_lighting,
    normalize_quality,
)

__version__ = "0.1.0"

__all__ = [
    "BiasReport",
    "generate_report",
    "generate_rows",
    "demo_summary",
    "distribution",
    "flag_underrepresented",
    "imbalance_ratio",
    "agreement_distribution",
    "agreement_score",
    "ambiguous_rows",
    "label_source_accuracy",
    "majority_label",
    "seasonal_boundary_stats",
    "leakage_check",
    "grouped_split",
    "group_kfold",
    "grouped_kfold",
    "split_by_person",
    "split_stats",
    "fold_sizes",
    "condition_stability",
    "makeup_filter_risk",
    "person_flip_rate",
    "quality_bucket_performance",
    "recommend_quality_floor",
    "validate",
    "SchemaReport",
    "subgroup_performance",
    "fairness_gaps",
    "worst_best",
    "verdict",
    "CHROMA_ORDER",
    "COLUMNS",
    "CONTRAST_ORDER",
    "LIGHTING_ORDER",
    "MAKEUP_ORDER",
    "QUALITY_ORDER",
    "SEASON_ORDER",
    "SKIN_DEPTH_ORDER",
    "SOURCE_ORDER",
    "UNDERTONE_ORDER",
    "normalize_lighting",
    "normalize_quality",
]
