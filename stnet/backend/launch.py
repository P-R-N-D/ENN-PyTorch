"""Legacy orchestration module stub.

The orchestration helpers previously exposed under :mod:`stnet.backend.launch`
were relocated to :mod:`stnet.api.run`.  Importing the old module path now
raises an informative :class:`ModuleNotFoundError` that directs callers to the
new location so downstream projects can update their imports immediately.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "stnet.backend.launch has moved to stnet.api.run; update imports to use "
    "`from stnet.api.run import train, predict, launch`."
)
