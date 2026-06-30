"""Dynamic ensemble selection module (L3).

Public API
----------
- majority_vote_pseudo_labels : generate pseudo-labels from detector scores
- generate_windowed_labels     : sliding-window pseudo-label helper
- meta_des_competence          : META-DES competence estimator
- windowed_competence          : competence over sliding windows
- ensemble_q_diversity         : Q-statistic ensemble diversity
- windowed_diversity           : diversity helper for a single window
- MetaDESSelector              : competence+diversity subset selector
- NaiveTopKSelector            : top-k-by-competence baseline selector
- DriftDetector                : ADWIN / Page-Hinkley drift wrapper
- scan_for_drift               : offline drift-flag scanner
"""

from cdade.selection.competence import meta_des_competence, windowed_competence
from cdade.selection.diversity import ensemble_q_diversity, windowed_diversity
from cdade.selection.drift_detector import DriftDetector, scan_for_drift
from cdade.selection.pseudo_label import generate_windowed_labels, majority_vote_pseudo_labels
from cdade.selection.selector import MetaDESSelector, NaiveTopKSelector

__all__ = [
    "majority_vote_pseudo_labels",
    "generate_windowed_labels",
    "meta_des_competence",
    "windowed_competence",
    "ensemble_q_diversity",
    "windowed_diversity",
    "MetaDESSelector",
    "NaiveTopKSelector",
    "DriftDetector",
    "scan_for_drift",
]
