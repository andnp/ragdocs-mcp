import sqlite3

from src.indices._corruption import is_corruption_error, is_transient_error


def test_database_locked_is_transient_not_corruption():
    exc = sqlite3.OperationalError("database is locked")

    assert is_transient_error(exc)
    assert not is_corruption_error(exc)


def test_malformed_database_is_corruption_not_transient():
    exc = sqlite3.DatabaseError("database disk image is malformed")

    assert is_corruption_error(exc)
    assert not is_transient_error(exc)