# ruff: noqa

"""
Provides entry points for model compression workflows.

Includes oneshot compression, training, and pre and post-processing utilities
for model optimization tasks.
"""

from .oneshot import Oneshot, oneshot
from .utils import post_process, pre_process

try:
    from .model_free import model_free_ptq
except ImportError:
    model_free_ptq = None
