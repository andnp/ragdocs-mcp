# Queue readiness for live validation

Use `queue status --json` to inspect the daemon before measuring search
performance. The queue payload reports three different kinds of state:

- `pending_count`, `scheduled_count`, and `running_count` describe current
  work. `running_count` includes only leases with a recent heartbeat.
- `historical_failure_count` reports durable Huey failure results retained for
  operator inspection.
- `failed_count` remains an established compatibility field for that same
  failure history.

Historical failures must not by themselves block a performance measurement.
Gate live measurements on pending work, due scheduled work, and recent active
leases. Inspect historical failures separately before deciding whether to
retry or purge them. Never clear queue state as part of performance validation.
