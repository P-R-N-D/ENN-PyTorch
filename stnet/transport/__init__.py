# -*- coding: utf-8 -*-
from __future__ import annotations

import sys

from ..distributed import Endpoint
from .. import distributed as socket

sys.modules.setdefault(f"{__name__}.socket", socket)

__all__ = [
    "Endpoint",
    "socket",
]
