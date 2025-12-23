#!/usr/bin/env bash
# Test runner with coverage

set -e

# Default to all tests
TEST_PATTERN="${1:-tests/}"

echo "Running tests: $TEST_PATTERN"

if [ "$TEST_PATTERN" = "tests/" ]; then
    # Full test suite with coverage
    uv run pytest \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-report=xml \
        -v \
        tests/

    echo ""
    echo "Coverage report generated:"
    echo "  - Terminal output above"
    echo "  - HTML: htmlcov/index.html"
    echo "  - XML: coverage.xml"
else
    # Specific tests without coverage
    uv run pytest -v "$TEST_PATTERN"
fi

echo ""
echo "Tests completed successfully!"
