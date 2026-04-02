def _missing_op(*args, **kwargs):
    raise ImportError(
        'roiaware_pool3d extension is not available in current environment.'
    )


try:
    from .points_in_boxes import (points_in_boxes_batch, points_in_boxes_cpu,
                                  points_in_boxes_gpu)
except Exception:
    points_in_boxes_batch = _missing_op
    points_in_boxes_cpu = _missing_op
    points_in_boxes_gpu = _missing_op

try:
    from .roiaware_pool3d import RoIAwarePool3d
except Exception:
    class RoIAwarePool3d:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _missing_op()

__all__ = [
    'RoIAwarePool3d', 'points_in_boxes_gpu', 'points_in_boxes_cpu',
    'points_in_boxes_batch'
]
