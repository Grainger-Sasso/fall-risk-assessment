"""Tests for the core identifier types (replaces the stale test/identifier_test.py)."""

import pytest

from src.identifiers.identifier import IdentifierGenerator
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.user.user_identifier import UserIdentifier


class TestUserIdentifier:
    def test_holds_value(self):
        identifier = UserIdentifier("user_01")
        assert identifier.value == "user_01"

    def test_rejects_empty_value(self):
        with pytest.raises(ValueError):
            UserIdentifier("")


class TestInstrumentIdentifier:
    def test_holds_name_and_version(self):
        identifier = InstrumentIdentifier("actigraph", "0001")
        assert identifier.name == "actigraph"


class TestIdentifierGenerator:
    def test_generates_prefixed_unique_ids(self):
        generator = IdentifierGenerator("imu", IMUDataIdentifier)
        first = generator.generate_identifier()
        second = generator.generate_identifier()
        assert first.value.startswith("imu")
        assert first.value != second.value
        assert isinstance(first, IMUDataIdentifier)
