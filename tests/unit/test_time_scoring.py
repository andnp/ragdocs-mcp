from datetime import datetime, timedelta, timezone

import pytest

from src.search.time_scoring import (
    DecayConfig,
    TierConfig,
    TimeScoreMode,
    apply_time_boost,
    calculate_time_score,
)


# --- Tier Mode Tests ---


class TestTierMode:
    def test_recent_document_gets_highest_boost(self):
        """
        Documents within the recent window get the highest boost.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=3)
        config = TierConfig(recent_days=7, recent_boost=1.2)

        score = calculate_time_score(timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert score == 1.2

    def test_moderate_document_gets_moderate_boost(self):
        """
        Documents between recent and moderate windows get moderate boost.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=15)
        config = TierConfig(recent_days=7, recent_boost=1.2, moderate_days=30, moderate_boost=1.1)

        score = calculate_time_score(timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert score == 1.1

    def test_old_document_gets_no_boost(self):
        """
        Documents older than moderate window get no boost (1.0).
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=60)
        config = TierConfig(recent_days=7, moderate_days=30)

        score = calculate_time_score(timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert score == 1.0

    def test_none_timestamp_returns_neutral(self):
        """
        None timestamp returns 1.0 for tier mode (no penalty).
        """
        score = calculate_time_score(None, TimeScoreMode.TIERS)

        assert score == 1.0

    def test_exact_boundary_recent(self):
        """
        Documents exactly at the recent boundary are included in recent tier.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=7)
        config = TierConfig(recent_days=7, recent_boost=1.2)

        score = calculate_time_score(timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert score == 1.2

    def test_exact_boundary_moderate(self):
        """
        Documents exactly at the moderate boundary are included in moderate tier.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=30)
        config = TierConfig(recent_days=7, moderate_days=30, moderate_boost=1.1)

        score = calculate_time_score(timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert score == 1.1


# --- Decay Mode Tests ---


class TestDecayMode:
    def test_fresh_document_gets_full_score(self):
        """
        Documents at time 0 should have score of 1.0.
        """
        now = datetime.now(timezone.utc)
        config = DecayConfig(half_life_days=7.0)

        score = calculate_time_score(now, TimeScoreMode.DECAY, config, reference_time=now)

        assert score == pytest.approx(1.0, rel=1e-6)

    def test_half_life_gives_half_score(self):
        """
        Documents at exactly half-life should have score of 0.5.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=7)
        config = DecayConfig(half_life_days=7.0, min_score=0.0)

        score = calculate_time_score(timestamp, TimeScoreMode.DECAY, config, reference_time=now)

        assert score == pytest.approx(0.5, rel=1e-6)

    def test_double_half_life_gives_quarter_score(self):
        """
        Documents at 2x half-life should have score of 0.25.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=14)
        config = DecayConfig(half_life_days=7.0, min_score=0.0)

        score = calculate_time_score(timestamp, TimeScoreMode.DECAY, config, reference_time=now)

        assert score == pytest.approx(0.25, rel=1e-6)

    def test_min_score_floor_is_respected(self):
        """
        Decay never goes below the configured minimum.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=365)
        config = DecayConfig(half_life_days=7.0, min_score=0.1)

        score = calculate_time_score(timestamp, TimeScoreMode.DECAY, config, reference_time=now)

        assert score == 0.1

    def test_none_timestamp_returns_neutral(self):
        """
        None timestamp returns 1.0 (no penalty) for backward compatibility.
        """
        score = calculate_time_score(None, TimeScoreMode.DECAY)

        assert score == 1.0

    def test_zero_half_life_returns_minimum(self):
        """
        Zero or negative half-life returns minimum score.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=1)
        config = DecayConfig(half_life_days=0.0, min_score=0.1)

        score = calculate_time_score(timestamp, TimeScoreMode.DECAY, config, reference_time=now)

        assert score == 0.1

    def test_very_short_half_life(self):
        """
        Very short half-life decays rapidly but respects floor.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=10)
        config = DecayConfig(half_life_days=1.0, min_score=0.05)

        score = calculate_time_score(timestamp, TimeScoreMode.DECAY, config, reference_time=now)

        expected = 2 ** (-10)
        assert score == pytest.approx(max(0.05, expected), rel=1e-6)


# --- Timezone Handling Tests ---


class TestTimezoneHandling:
    def test_naive_datetime_treated_as_utc(self):
        """
        Naive datetimes are treated as UTC.
        """
        now = datetime.now(timezone.utc)
        naive_timestamp = now.replace(tzinfo=None) - timedelta(days=3)
        config = TierConfig(recent_days=7, recent_boost=1.2)

        score = calculate_time_score(naive_timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert score == 1.2

    def test_different_timezone_handled_correctly(self):
        """
        Timestamps in different timezones are correctly compared.
        """
        from datetime import timezone as tz

        utc_now = datetime.now(tz.utc)
        est = tz(timedelta(hours=-5))
        est_timestamp = (utc_now - timedelta(days=3)).astimezone(est)
        config = TierConfig(recent_days=7, recent_boost=1.2)

        score = calculate_time_score(est_timestamp, TimeScoreMode.TIERS, config, reference_time=utc_now)

        assert score == 1.2


# --- apply_time_boost Tests ---


class TestApplyTimeBoost:
    def test_boost_multiplies_score_tiers(self):
        """
        apply_time_boost correctly multiplies score by tier boost.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=3)
        config = TierConfig(recent_days=7, recent_boost=1.2)

        boosted = apply_time_boost(0.5, timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert boosted == pytest.approx(0.6, rel=1e-6)

    def test_boost_multiplies_score_decay(self):
        """
        apply_time_boost correctly multiplies score by decay factor.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=7)
        config = DecayConfig(half_life_days=7.0, min_score=0.0)

        boosted = apply_time_boost(1.0, timestamp, TimeScoreMode.DECAY, config, reference_time=now)

        assert boosted == pytest.approx(0.5, rel=1e-6)

    def test_zero_score_remains_zero(self):
        """
        Boosting a zero score still results in zero.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=1)
        config = TierConfig(recent_boost=1.5)

        boosted = apply_time_boost(0.0, timestamp, TimeScoreMode.TIERS, config, reference_time=now)

        assert boosted == 0.0


# --- Config Validation Tests ---


class TestConfigValidation:
    def test_wrong_config_type_for_tiers_raises(self):
        """
        Using DecayConfig with TIERS mode raises TypeError.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=3)
        wrong_config = DecayConfig()

        with pytest.raises(TypeError, match="TIERS mode requires TierConfig"):
            calculate_time_score(timestamp, TimeScoreMode.TIERS, wrong_config, reference_time=now)

    def test_wrong_config_type_for_decay_raises(self):
        """
        Using TierConfig with DECAY mode raises TypeError.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=3)
        wrong_config = TierConfig()

        with pytest.raises(TypeError, match="DECAY mode requires DecayConfig"):
            calculate_time_score(timestamp, TimeScoreMode.DECAY, wrong_config, reference_time=now)

    def test_default_tier_config_used_when_none(self):
        """
        When config is None in TIERS mode, default TierConfig is used.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=3)

        score = calculate_time_score(timestamp, TimeScoreMode.TIERS, reference_time=now)

        assert score == 1.2

    def test_default_decay_config_used_when_none(self):
        """
        When config is None in DECAY mode, default DecayConfig is used.
        """
        now = datetime.now(timezone.utc)
        timestamp = now - timedelta(days=7)

        score = calculate_time_score(timestamp, TimeScoreMode.DECAY, reference_time=now)

        assert score == pytest.approx(0.5, rel=1e-6)


# --- Edge Cases ---


class TestEdgeCases:
    def test_future_timestamp_treated_as_fresh(self):
        """
        Future timestamps (clock skew) are treated as age 0.
        """
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=5)
        config = DecayConfig(half_life_days=7.0)

        score = calculate_time_score(future, TimeScoreMode.DECAY, config, reference_time=now)

        assert score == pytest.approx(1.0, rel=1e-6)

    def test_very_old_document_decay(self):
        """
        Very old documents decay to the floor.
        """
        now = datetime.now(timezone.utc)
        ancient = now - timedelta(days=3650)
        config = DecayConfig(half_life_days=7.0, min_score=0.05)

        score = calculate_time_score(ancient, TimeScoreMode.DECAY, config, reference_time=now)

        assert score == 0.05
