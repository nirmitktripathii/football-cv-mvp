"""Vendored subset of TrackEval (MIT, see LICENSE): the HOTA, CLEAR and Identity metrics only.

Source: https://github.com/JonathonLuiten/TrackEval at commit 12c8791b303e0a0b50f753af204249e622d0281a
(2022-11-29, the latest). Copied: trackeval/_timing.py, trackeval/utils.py and trackeval/metrics/
{_base_metric,hota,clear,identity}.py. The only change is replacing the NumPy aliases removed in
NumPy 1.24 (``np.float`` -> ``float`` in metrics/hota.py lines 31, 37, 38, 42, 43; ``np.int`` -> ``int``
in metrics/identity.py lines 83-85). Upstream's own ``metrics/__init__`` is replaced by one that imports
these three metrics. The dataset loaders are not included; ``football_cv.evaluation.mot`` builds the
per-sequence data dict that the metrics' ``eval_sequence`` expects.
"""
from . import metrics
