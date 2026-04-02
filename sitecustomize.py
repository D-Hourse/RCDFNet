"""Runtime compatibility shims for RCDFNet-main.

This module is auto-imported by Python at startup (when present on ``sys.path``).
It provides a light fallback for environments where ``mmcv-full`` is unavailable
and ``mmcv._ext`` cannot be imported.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Any


def _install_mmcv_ext_stub() -> None:
    """Install a lazy stub for ``mmcv._ext`` when compiled ops are missing.

    The stub only exists to unblock import-time symbol checks.
    Any actual call into an op will raise an informative ImportError.
    """

    if 'mmcv._ext' in sys.modules:
        return

    try:
        importlib.import_module('mmcv._ext')
        return
    except Exception:
        pass

    ext_mod = types.ModuleType('mmcv._ext')
    ext_mod.__file__ = '<mmcv._ext_stub>'

    def _missing_op(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise ImportError(
            'mmcv._ext is not available in this environment. '
            'Install mmcv-full with matching torch/cuda, or run in the RCFusion environment.'
        )

    def _getattr(name: str) -> Any:
        if name.startswith('__'):
            raise AttributeError(name)
        return _missing_op

    ext_mod.__getattr__ = _getattr  # type: ignore[attr-defined]
    sys.modules['mmcv._ext'] = ext_mod


def _install_mmcv_log_filter() -> None:
    """Filter noisy MMCV init logs globally.

    This suppresses repetitive parameter-shape and unchanged-init messages such as:
    - "xxx - torch.Size(...):"
    - "The value is the same before and after calling `init_weights` ..."
    """

    class _MMCVInitLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
            if not record.name.startswith('mmcv'):
                return True

            msg = record.getMessage()
            if 'The value is the same before and after calling `init_weights`' in msg:
                return False
            if 'Initialized by user-defined `init_weights`' in msg:
                return False
            if ' - torch.Size(' in msg and msg.rstrip().endswith(':'):
                return False
            return True

    f = _MMCVInitLogFilter()
    logging.getLogger('mmcv').addFilter(f)
    logging.getLogger('mmcv.cnn').addFilter(f)
    logging.getLogger().addFilter(f)


_install_mmcv_ext_stub()
_install_mmcv_log_filter()
