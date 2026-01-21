import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class TimeScoreMode(Enum):
    TIERS = "tiers"
    DECAY = "decay"


@dataclass
class TierConfig:
    recent_days: int = 7
    recent_boost: float = 1.2
    moderate_days: int = 30
    moderate_boost: float = 1.1


@dataclass
class DecayConfig:
    half_life_days: float = 7.0
    min_score: float = 0.1


def _normalize_timestamp(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _calculate_age_days(timestamp: datetime, reference_time: datetime) -> float:
    normalized = _normalize_timestamp(timestamp)
    ref_normalized = _normalize_timestamp(reference_time)
    delta = ref_normalized - normalized
    return max(0.0, delta.total_seconds() / 86400)


def _calculate_tier_score(age_days: float, config: TierConfig) -> float:
    if age_days <= config.recent_days:
        return config.recent_boost
    if age_days <= config.moderate_days:
        return config.moderate_boost
    return 1.0


def _calculate_decay_score(age_days: float, config: DecayConfig) -> float:
    if config.half_life_days <= 0:
        return config.min_score
    decay_constant = math.log(2) / config.half_life_days
    raw_score = math.exp(-decay_constant * age_days)
    return max(config.min_score, raw_score)


def calculate_time_score(
    timestamp: datetime | None,
    mode: TimeScoreMode,
    config: TierConfig | DecayConfig | None = None,
    *,
    reference_time: datetime | None = None,
) -> float:
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    if timestamp is None:
        # No penalty for missing timestamps (backward compatibility with legacy memories)
        return 1.0

    if config is None:
        if mode == TimeScoreMode.TIERS:
            config = TierConfig()
        else:
            config = DecayConfig()

    age_days = _calculate_age_days(timestamp, reference_time)

    if mode == TimeScoreMode.TIERS:
        if not isinstance(config, TierConfig):
            raise TypeError(f"TIERS mode requires TierConfig, got {type(config).__name__}")
        return _calculate_tier_score(age_days, config)

    if not isinstance(config, DecayConfig):
        raise TypeError(f"DECAY mode requires DecayConfig, got {type(config).__name__}")
    return _calculate_decay_score(age_days, config)


def apply_time_boost(
    score: float,
    timestamp: datetime | None,
    mode: TimeScoreMode,
    config: TierConfig | DecayConfig | None = None,
    *,
    reference_time: datetime | None = None,
) -> float:
    multiplier = calculate_time_score(
        timestamp, mode, config, reference_time=reference_time
    )
    return score * multiplier
