# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, TypeAlias

from ..toolkit.compat import secure_torch, SDPBackend, sdpa_kernel
from .module import (
    SpatialSubnet,
    TemporalSubnet,
    SpatioTemporalNet,
    GeGLU,
    SwiGLU,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    DataFidelityLoss,
)
from .network import Model, Config

secure_torch()

__all__ = [
    'sdpa_kernel',
    'SDPBackend',
    'Model',
    'Config',
    'SpatialSubnet',
    'TemporalSubnet',
    'SpatioTemporalNet',
    'GeGLU',
    'SwiGLU',
    'MultipleQuantileLoss',
    'StandardNormalLoss',
    'StudentsTLoss',
    'DataFidelityLoss',
]

def _default_sdpa_backends() -> List:
    backends = []
    for name in ('FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH'):
        if hasattr(SDPBackend, name):
            backends.append(getattr(SDPBackend, name))
    return backends

ZLoss: TypeAlias = StandardNormalLoss
TLoss: TypeAlias = StudentsTLoss