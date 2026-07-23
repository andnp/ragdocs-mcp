"""Unit tests for golden set schema and loader."""

import json
import tempfile
from pathlib import Path

import pytest

from searchkernel.eval.golden import GoldenEntry, GoldenSet, load_golden, save_golden


class TestGoldenEntry:
    """Tests for GoldenEntry dataclass."""

    def test_golden_entry_creation(self):
        """Test creating a golden entry."""
        entry = GoldenEntry(
            query="test query",
            relevant_ids=["id1", "id2"],
        )
        assert entry.query == "test query"
        assert entry.relevant_ids == ["id1", "id2"]

    def test_golden_entry_to_dict(self):
        """Test serialization."""
        entry = GoldenEntry(
            query="test query",
            relevant_ids=["id1", "id2"],
        )
        d = entry.to_dict()
        assert d["query"] == "test query"
        assert d["relevant_ids"] == ["id1", "id2"]

    def test_golden_entry_from_dict(self):
        """Test deserialization."""
        d = {
            "query": "test query",
            "relevant_ids": ["id1", "id2"],
        }
        entry = GoldenEntry.from_dict(d)
        assert entry.query == "test query"
        assert entry.relevant_ids == ["id1", "id2"]

    def test_golden_entry_round_trip(self):
        """Test serialization and deserialization."""
        original = GoldenEntry(
            query="original query",
            relevant_ids=["a", "b", "c"],
        )
        d = original.to_dict()
        restored = GoldenEntry.from_dict(d)
        assert restored.query == original.query
        assert restored.relevant_ids == original.relevant_ids


class TestGoldenSet:
    """Tests for GoldenSet dataclass."""

    def test_golden_set_creation(self):
        """Test creating a golden set."""
        entries = [
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b", "c"]),
        ]
        gs = GoldenSet(entries=entries)
        assert len(gs) == 2
        assert gs.entries[0].query == "q1"

    def test_golden_set_len(self):
        """Test __len__ method."""
        entries = [
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
            GoldenEntry(query="q3", relevant_ids=["c"]),
        ]
        gs = GoldenSet(entries=entries)
        assert len(gs) == 3

    def test_golden_set_iter(self):
        """Test __iter__ method."""
        entries = [
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
        ]
        gs = GoldenSet(entries=entries)

        collected = list(gs)
        assert len(collected) == 2
        assert collected[0].query == "q1"
        assert collected[1].query == "q2"

    def test_golden_set_to_dict(self):
        """Test serialization."""
        entries = [
            GoldenEntry(query="q1", relevant_ids=["a", "b"]),
        ]
        gs = GoldenSet(entries=entries)
        d = gs.to_dict()

        assert "entries" in d
        assert len(d["entries"]) == 1
        assert d["entries"][0]["query"] == "q1"

    def test_golden_set_from_dict(self):
        """Test deserialization."""
        d = {
            "entries": [
                {"query": "q1", "relevant_ids": ["a", "b"]},
                {"query": "q2", "relevant_ids": ["c"]},
            ]
        }
        gs = GoldenSet.from_dict(d)
        assert len(gs) == 2
        assert gs.entries[0].query == "q1"
        assert gs.entries[1].query == "q2"


class TestLoadGolden:
    """Tests for load_golden function."""

    def test_load_golden_valid_file(self):
        """Test loading a valid golden set file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "golden.json"
            data = {
                "entries": [
                    {"query": "test query", "relevant_ids": ["id1", "id2"]},
                ]
            }
            with open(path, "w") as f:
                json.dump(data, f)

            gs = load_golden(path)
            assert len(gs) == 1
            assert gs.entries[0].query == "test query"
            assert gs.entries[0].relevant_ids == ["id1", "id2"]

    def test_load_golden_multiple_entries(self):
        """Test loading golden set with multiple entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "golden.json"
            data = {
                "entries": [
                    {"query": "q1", "relevant_ids": ["a"]},
                    {"query": "q2", "relevant_ids": ["b", "c"]},
                    {"query": "q3", "relevant_ids": ["d"]},
                ]
            }
            with open(path, "w") as f:
                json.dump(data, f)

            gs = load_golden(path)
            assert len(gs) == 3
            assert gs.entries[1].relevant_ids == ["b", "c"]

    def test_load_golden_file_not_found(self):
        """Test loading non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"

            with pytest.raises(FileNotFoundError):
                load_golden(path)

    def test_load_golden_invalid_json(self):
        """Test loading invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            with open(path, "w") as f:
                f.write("{ invalid json")

            with pytest.raises(json.JSONDecodeError):
                load_golden(path)

    def test_load_golden_missing_entries_key(self):
        """Test loading JSON without 'entries' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            data = {"wrong_key": []}
            with open(path, "w") as f:
                json.dump(data, f)

            with pytest.raises(ValueError):
                load_golden(path)

    def test_load_golden_empty_entries(self):
        """Test loading golden set with empty entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            data = {"entries": []}
            with open(path, "w") as f:
                json.dump(data, f)

            gs = load_golden(path)
            assert len(gs) == 0


class TestSaveGolden:
    """Tests for save_golden function."""

    def test_save_golden_creates_file(self):
        """Test that save_golden creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.json"

            gs = GoldenSet(entries=[
                GoldenEntry(query="test", relevant_ids=["a", "b"]),
            ])
            save_golden(gs, path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert len(data["entries"]) == 1

    def test_save_golden_round_trip(self):
        """Test save and load round trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.json"

            original = GoldenSet(entries=[
                GoldenEntry(query="q1", relevant_ids=["a", "b"]),
                GoldenEntry(query="q2", relevant_ids=["c"]),
            ])
            save_golden(original, path)
            loaded = load_golden(path)

            assert len(loaded) == len(original)
            assert loaded.entries[0].query == original.entries[0].query
            assert loaded.entries[0].relevant_ids == original.entries[0].relevant_ids

    def test_save_golden_creates_parent_dirs(self):
        """Test that save_golden creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "output.json"

            gs = GoldenSet(entries=[
                GoldenEntry(query="test", relevant_ids=["a"]),
            ])
            save_golden(gs, path)

            assert path.exists()
            assert path.parent.exists()
