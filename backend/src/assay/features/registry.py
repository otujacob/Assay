"""Versioned feature registry (PRD 3.2, FR-06). Each feature has a definition, owner and version;
every feature vector records the definition versions and the as-of time it was computed at."""

from __future__ import annotations

from dataclasses import dataclass

FEATURE_SET_VERSION = "fs-2"
CHANNEL_ORDER = ("card_present", "card_not_present", "app", "web", "open_banking")


@dataclass(frozen=True)
class FeatureDef:
    name: str
    version: int
    owner: str
    description: str


def _f(name: str, desc: str, version: int = 1, owner: str = "ml") -> FeatureDef:
    return FeatureDef(name, version, owner, desc)


REGISTRY: tuple[FeatureDef, ...] = (
    _f("amount", "Transaction amount"),
    _f("log_amount", "log(1 + amount)"),
    _f("hour", "Hour of day (UTC) of the transaction"),
    _f("channel_code", "Index into CHANNEL_ORDER; -1 if the channel is not in the list"),
    _f("velocity_1h", "Customer's prior transactions in the hour before this one"),
    _f("velocity_24h", "Customer's prior transactions in the 24 hours before this one"),
    _f("velocity_30d", "Customer's prior transactions in the 30 days before this one"),
    _f("amount_ratio_baseline", "Amount / mean of the customer's prior amounts (1.0 if fewer than 3)"),
    _f("new_beneficiary", "1 if a beneficiary is present and this customer never paid it before"),
    _f("new_device", "1 if a device is present, the customer has history, and never used it before"),
    _f("country_mismatch", "1 if country differs from the customer's most common prior country"),
    _f("beneficiary_shared_customers", "Distinct other customers who paid this beneficiary in the prior 30 days", version=2),
    _f("missing_fields", "Count of absent optional fields (device, ip, country, merchant)"),
)

# Why windowed (velocity_30d, beneficiary_shared_customers): an all-time count grows just because
# the dataset ages, so a later period always looks "new" next to an earlier one. That
# non-stationarity made a third of ordinary traffic look unfamiliar to the novelty detector.

# Strongly related features move together. Explanations are scored at group level because
# attributions for correlated features are unstable by construction (PRD 7.4), and perturbations
# change whole groups so they never produce impossible combinations (e.g. amount without log_amount).
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "amount": ("amount", "log_amount", "amount_ratio_baseline"),
    "velocity": ("velocity_1h", "velocity_24h", "velocity_30d"),
    "timing": ("hour",),
    "channel": ("channel_code",),
    "counterparty": ("new_beneficiary", "beneficiary_shared_customers"),
    "device_location": ("new_device", "country_mismatch"),
    "completeness": ("missing_fields",),
}

# Categorical features: a value never seen in the reference window is out-of-distribution.
CATEGORICAL_FEATURES: tuple[str, ...] = ("channel_code",)


def feature_names() -> tuple[str, ...]:
    return tuple(d.name for d in REGISTRY)


def definition_versions() -> dict[str, int]:
    return {d.name: d.version for d in REGISTRY}


def group_index() -> dict[str, list[int]]:
    names = feature_names()
    return {g: [names.index(f) for f in fs] for g, fs in FEATURE_GROUPS.items()}


def validate_registry() -> None:
    names = feature_names()
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names")
    for d in REGISTRY:
        if not (d.owner and d.description and d.version >= 1):
            raise ValueError(f"incomplete definition: {d.name}")
    grouped = [f for fs in FEATURE_GROUPS.values() for f in fs]
    if sorted(grouped) != sorted(names):
        raise ValueError("every feature must be in exactly one group")
    if not set(CATEGORICAL_FEATURES) <= set(names):
        raise ValueError("unknown categorical feature")


validate_registry()
