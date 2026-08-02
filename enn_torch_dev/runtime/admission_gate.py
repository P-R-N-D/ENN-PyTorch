from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .admission import (
    PrePassAdmissionAssessment,
    PrePassAdmissionPolicy,
    PrePassAdmissionStatus,
    assess_prepass_admission,
)
from .calibration import ObservedCostProfile
from .faults import ResourceSample
from .pressure import ResourceCapacity


@runtime_checkable
class ResourceSampleProvider(Protocol):
    """Provide one execution-immediate resource sample for admission."""

    def sample(self, phase: str) -> ResourceSample:
        ...


class AdmissionUnknownAction(Enum):
    """Execution behavior when a pre-pass assessment is unknown."""

    BLOCK = "block"
    ALLOW = "allow"


@dataclass(frozen=True, slots=True)
class AdmissionSplitPolicy:
    """Bound recursive admission-driven splitting for rejected candidates."""

    max_split_depth: int = 3
    min_items: int = 1
    max_split_parts: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.max_split_depth, int) or isinstance(
            self.max_split_depth, bool
        ):
            raise TypeError("AdmissionSplitPolicy.max_split_depth must be an integer.")
        if self.max_split_depth < 0:
            raise ValueError(
                "AdmissionSplitPolicy.max_split_depth must be non-negative."
            )
        if not isinstance(self.min_items, int) or isinstance(self.min_items, bool):
            raise TypeError("AdmissionSplitPolicy.min_items must be an integer.")
        if self.min_items <= 0:
            raise ValueError("AdmissionSplitPolicy.min_items must be positive.")
        if not isinstance(self.max_split_parts, int) or isinstance(
            self.max_split_parts, bool
        ):
            raise TypeError("AdmissionSplitPolicy.max_split_parts must be an integer.")
        if self.max_split_parts < 2:
            raise ValueError(
                "AdmissionSplitPolicy.max_split_parts must be at least 2."
            )


class PrePassAdmissionBlocked(RuntimeError):
    """Raised when an opt-in admission gate blocks one execution attempt."""

    def __init__(self, assessment: PrePassAdmissionAssessment) -> None:
        if not isinstance(assessment, PrePassAdmissionAssessment):
            raise TypeError(
                "PrePassAdmissionBlocked.assessment must be a "
                "PrePassAdmissionAssessment."
            )
        self.assessment = assessment
        super().__init__(
            "Pre-pass admission blocked execution: "
            f"status={assessment.status.value}, "
            f"rejected_dimensions={assessment.rejected_dimensions!r}, "
            f"unknown_dimensions={assessment.unknown_dimensions!r}."
        )


class PrePassAdmissionGate:
    """Sample, assess, and optionally block one candidate execution attempt."""

    __slots__ = (
        "capacity",
        "observed_profile",
        "sample_provider",
        "policy",
        "unknown_action",
    )

    def __init__(
        self,
        capacity: ResourceCapacity,
        observed_profile: ObservedCostProfile,
        sample_provider: ResourceSampleProvider,
        *,
        policy: PrePassAdmissionPolicy | None = None,
        unknown_action: AdmissionUnknownAction = AdmissionUnknownAction.BLOCK,
    ) -> None:
        if not isinstance(capacity, ResourceCapacity):
            raise TypeError("PrePassAdmissionGate.capacity must be a ResourceCapacity.")
        if not isinstance(observed_profile, ObservedCostProfile):
            raise TypeError(
                "PrePassAdmissionGate.observed_profile must be an ObservedCostProfile."
            )
        if not isinstance(sample_provider, ResourceSampleProvider):
            raise TypeError(
                "PrePassAdmissionGate.sample_provider must provide "
                "sample(str) -> ResourceSample."
            )
        if policy is not None and not isinstance(policy, PrePassAdmissionPolicy):
            raise TypeError(
                "PrePassAdmissionGate.policy must be a PrePassAdmissionPolicy or None."
            )
        if not isinstance(unknown_action, AdmissionUnknownAction):
            raise TypeError(
                "PrePassAdmissionGate.unknown_action must be an "
                "AdmissionUnknownAction."
            )

        self.capacity = capacity
        self.observed_profile = observed_profile
        self.sample_provider = sample_provider
        self.policy = policy
        self.unknown_action = unknown_action

    def check(self, batch_size: int) -> PrePassAdmissionAssessment:
        """Assess one attempt and raise when the configured gate blocks it."""

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("PrePassAdmissionGate.batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("PrePassAdmissionGate.batch_size must be positive.")

        baseline = self.sample_provider.sample("before_admission")
        if not isinstance(baseline, ResourceSample):
            raise TypeError(
                "ResourceSampleProvider.sample() must return a ResourceSample."
            )

        assessment = assess_prepass_admission(
            self.capacity,
            baseline,
            self.observed_profile,
            batch_size=batch_size,
            policy=self.policy,
        )
        if assessment.status is PrePassAdmissionStatus.REJECT:
            raise PrePassAdmissionBlocked(assessment)
        if (
            assessment.status is PrePassAdmissionStatus.UNKNOWN
            and self.unknown_action is AdmissionUnknownAction.BLOCK
        ):
            raise PrePassAdmissionBlocked(assessment)
        return assessment
