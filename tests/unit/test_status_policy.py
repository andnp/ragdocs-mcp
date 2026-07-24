from searchkernel.domain.models import RecordStatus
from searchkernel.domain.status_policy import (
    DEFAULT_STATUS_MULTIPLIER_POLICY,
    StatusMultiplierPolicy,
)


def test_default_policy_does_not_penalize_active_records():
    assert DEFAULT_STATUS_MULTIPLIER_POLICY.multiplier_for(RecordStatus.ACTIVE) == 1.0


def test_default_policy_penalizes_stale_more_than_active_but_less_than_archived():
    stale = DEFAULT_STATUS_MULTIPLIER_POLICY.multiplier_for(RecordStatus.STALE)
    archived = DEFAULT_STATUS_MULTIPLIER_POLICY.multiplier_for(RecordStatus.ARCHIVED)
    assert 0.0 < archived < stale < 1.0


def test_multiplier_for_unmapped_status_defaults_to_no_penalty():
    empty_policy = StatusMultiplierPolicy(multipliers={})
    assert empty_policy.multiplier_for(RecordStatus.ARCHIVED) == 1.0


def test_policy_is_pure_config_overridable_without_touching_code():
    custom = StatusMultiplierPolicy(multipliers={RecordStatus.STALE: 0.1})
    assert custom.multiplier_for(RecordStatus.STALE) == 0.1
