import logging

import comfy.ops

from .ops import easy_mixed_precision_ops


def _pick_operations(weight_dtype, compute_dtype, *args, **kwargs):
    logging.info(f"Using easy_mixed_precision_ops({compute_dtype})")
    return easy_mixed_precision_ops(compute_dtype)


comfy.ops.pick_operations = _pick_operations

NODE_CLASS_MAPPINGS = {}
