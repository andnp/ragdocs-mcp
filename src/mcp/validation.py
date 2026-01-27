"""Input validation utilities for MCP tool handlers.

Provides runtime validation for MCP tool arguments to complement
schema-level validation. Ensures type safety and catches edge cases
that JSON schema validation may miss.
"""

import logging

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when MCP tool arguments fail validation."""
    pass


def validate_query(arguments: dict, param_name: str = "query") -> str:
    """Validate and extract a required query string parameter.

    Args:
        arguments: MCP tool arguments dict
        param_name: Name of the query parameter (default: "query")

    Returns:
        Validated, stripped query string

    Raises:
        ValidationError: If query is missing, empty, or not a string
    """
    query = arguments.get(param_name)

    if query is None:
        raise ValidationError(f"{param_name} parameter is required")

    if not isinstance(query, str):
        raise ValidationError(f"{param_name} must be a string, got {type(query).__name__}")

    query = query.strip()
    if not query:
        raise ValidationError(f"{param_name} cannot be empty")

    return query


def validate_integer_range(
    arguments: dict,
    param_name: str,
    default: int,
    min_val: int,
    max_val: int,
) -> int:
    """Validate and extract an integer parameter within a range.

    Args:
        arguments: MCP tool arguments dict
        param_name: Name of the parameter
        default: Default value if parameter is not provided
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Validated integer value

    Raises:
        ValidationError: If value is not an integer or out of range
    """
    value = arguments.get(param_name)

    if value is None:
        return default

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{param_name} must be an integer, got {type(value).__name__}")

    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"{param_name} must be between {min_val} and {max_val}, got {value}"
        )

    return value


def validate_float_range(
    arguments: dict,
    param_name: str,
    default: float,
    min_val: float,
    max_val: float,
) -> float:
    """Validate and extract a float parameter within a range.

    Args:
        arguments: MCP tool arguments dict
        param_name: Name of the parameter
        default: Default value if parameter is not provided
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Validated float value

    Raises:
        ValidationError: If value is not a number or out of range
    """
    value = arguments.get(param_name)

    if value is None:
        return default

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{param_name} must be a number, got {type(value).__name__}")

    float_value = float(value)

    if not (min_val <= float_value <= max_val):
        raise ValidationError(
            f"{param_name} must be between {min_val} and {max_val}, got {float_value}"
        )

    return float_value


def validate_string_list(
    arguments: dict,
    param_name: str,
    default: list[str] | None = None,
) -> list[str]:
    """Validate and extract a list of strings parameter.

    Args:
        arguments: MCP tool arguments dict
        param_name: Name of the parameter
        default: Default value if parameter is not provided (default: [])

    Returns:
        Validated list of strings

    Raises:
        ValidationError: If value is not a list or contains non-strings
    """
    if default is None:
        default = []

    value = arguments.get(param_name)

    if value is None:
        return default

    if not isinstance(value, list):
        raise ValidationError(f"{param_name} must be a list, got {type(value).__name__}")

    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValidationError(
                f"{param_name}[{i}] must be a string, got {type(item).__name__}"
            )

    return value


def validate_enum(
    arguments: dict,
    param_name: str,
    allowed_values: set[str],
    default: str | None = None,
) -> str | None:
    """Validate and extract an enum parameter.

    Args:
        arguments: MCP tool arguments dict
        param_name: Name of the parameter
        allowed_values: Set of allowed string values
        default: Default value if parameter is not provided (default: None)

    Returns:
        Validated enum value or None

    Raises:
        ValidationError: If value is not in allowed set
    """
    value = arguments.get(param_name)

    if value is None:
        return default

    if not isinstance(value, str):
        raise ValidationError(f"{param_name} must be a string, got {type(value).__name__}")

    if value not in allowed_values:
        raise ValidationError(
            f"{param_name} must be one of {sorted(allowed_values)}, got '{value}'"
        )

    return value


def validate_boolean(
    arguments: dict,
    param_name: str,
    default: bool,
) -> bool:
    """Validate and extract a boolean parameter.

    Args:
        arguments: MCP tool arguments dict
        param_name: Name of the parameter
        default: Default value if parameter is not provided

    Returns:
        Validated boolean value

    Raises:
        ValidationError: If value is not a boolean
    """
    value = arguments.get(param_name)

    if value is None:
        return default

    if not isinstance(value, bool):
        raise ValidationError(f"{param_name} must be a boolean, got {type(value).__name__}")

    return value


def validate_timestamp(
    arguments: dict,
    param_name: str,
    default: int | None = None,
) -> int | None:
    """Validate and extract a Unix timestamp parameter.

    Args:
        arguments: MCP tool arguments dict
        param_name: Name of the parameter
        default: Default value if parameter is not provided (default: None)

    Returns:
        Validated Unix timestamp (positive integer) or None

    Raises:
        ValidationError: If value is not a positive integer
    """
    value = arguments.get(param_name)

    if value is None:
        return default

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{param_name} must be an integer, got {type(value).__name__}")

    if value < 0:
        raise ValidationError(f"{param_name} must be a positive Unix timestamp, got {value}")

    return value
