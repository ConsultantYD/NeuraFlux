"""Backward-compatible alias for the canonical DQN/DDQN estimator implementation.

The project historically carried two near-identical implementations in `dqn.py` and `ddqn.py`.
`neuraflux.agency.dqn.DDQNPREstimator` is the canonical implementation; this module is kept as an
import alias to minimize disruption for downstream users and legacy code paths.
"""

from __future__ import annotations

from neuraflux.agency.dqn import DDQNPREstimator

__all__ = ["DDQNPREstimator"]
