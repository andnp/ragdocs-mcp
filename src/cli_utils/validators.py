import sys

import click


def validate_range(value: int, min_val: int, max_val: int, param_name: str) -> None:
    if value < min_val or value > max_val:
        click.echo(
            f"Error: {param_name} must be between {min_val} and {max_val}", err=True
        )
        sys.exit(1)


def validate_timestamp_range(after: int | None, before: int | None) -> None:
    if after is not None and before is not None and after >= before:
        click.echo("Error: --after must be less than --before", err=True)
        sys.exit(1)


def validate_non_negative(value: int | None, param_name: str) -> None:
    if value is not None and value < 0:
        click.echo(f"Error: {param_name} must be non-negative", err=True)
        sys.exit(1)
