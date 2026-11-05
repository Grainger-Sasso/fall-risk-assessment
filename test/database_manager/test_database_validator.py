import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional
from unittest.mock import MagicMock

from src.database_manager.data_access.domain_io_router import DomainIORouter
from src.database_manager.database_errors import DatabaseValidationError
from src.database_manager.database_manager import DatabaseManager
from src.database_manager.database_validator import DatabaseValidator
from src.database_manager.metadata.metadata_repository import MetadataRepository
from src.database_manager.metadata.relation_types import (
    FEATURE_TO_IMU,
    IMU_TO_INSTRUMENT_SPEC,
    IMU_TO_USER,
)
from src.database_manager.metadata.sqlite_store import SQLiteStore
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.user.user_identifier import UserIdentifier

SPEC = "instrument_specification"


class _PayloadStore:
    """In-memory stand-in for payloads on disk, keyed by record directory name."""

    def __init__(self) -> None:
        self.imu_headers: Dict[str, SimpleNamespace] = {}
        self.users: Dict[str, SimpleNamespace] = {}
        self.features: Dict[str, SimpleNamespace] = {}
        self.specs: Dict[str, SimpleNamespace] = {}
        self.unreadable: Dict[str, Exception] = {}

    def reader(self, payloads: Dict[str, SimpleNamespace]) -> Callable:
        def read(*args: object) -> SimpleNamespace:
            directory = Path(args[-1])
            if directory.name in self.unreadable:
                raise self.unreadable[directory.name]
            if directory.name not in payloads:
                raise FileNotFoundError(f"No payload file in {directory}")
            return payloads[directory.name]

        return read


class _ValidatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repository = MetadataRepository(SQLiteStore(self.root / "index.db"))
        self.payloads = _PayloadStore()
        self.io_router = MagicMock(spec=DomainIORouter)
        self.io_router.import_imu_header.side_effect = self.payloads.reader(
            self.payloads.imu_headers
        )
        self.io_router.import_user.side_effect = self.payloads.reader(self.payloads.users)
        self.io_router.import_features.side_effect = self.payloads.reader(
            self.payloads.features
        )
        self.io_router.import_instrument_spec.side_effect = self.payloads.reader(
            self.payloads.specs
        )
        self.db_manager = DatabaseManager(
            metadata_repository=self.repository, io_router=self.io_router
        )
        self.validator = DatabaseValidator()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _index(self, id_type: str, identifier: str) -> None:
        self.repository.upsert_record(id_type, identifier, self.root / identifier)

    def _relate(self, source: tuple, target: tuple, relation_type: str) -> None:
        self.repository.add_relation(source[0], source[1], target[0], target[1], relation_type)

    def _add_user(self, user_id: str, *, payload: bool = True) -> None:
        self._index("user_data", user_id)
        if payload:
            self.payloads.users[user_id] = SimpleNamespace(
                user_identifier=UserIdentifier(user_id)
            )

    def _add_imu(
        self,
        imu_id: str,
        user_id: str,
        *,
        payload_user_id: Optional[str] = None,
        payload_imu_id: Optional[str] = None,
        instrument_name: str = "placeholder",
        link: bool = True,
        spec_id: Optional[str] = None,
    ) -> None:
        self._index("imu_data", imu_id)
        self.payloads.imu_headers[imu_id] = SimpleNamespace(
            imu_data_identifier=IMUDataIdentifier(payload_imu_id or imu_id),
            user_identifier=UserIdentifier(payload_user_id or user_id),
            instrument_identifier=SimpleNamespace(name=instrument_name),
        )
        if link:
            self._relate(("imu_data", imu_id), ("user_data", user_id), IMU_TO_USER)
        if spec_id is not None:
            self._relate(("imu_data", imu_id), (SPEC, spec_id), IMU_TO_INSTRUMENT_SPEC)

    def _add_feature(
        self,
        feature_id: str,
        imu_id: str,
        *,
        payload_imu_id: Optional[str] = None,
        link: bool = True,
    ) -> None:
        self._index("feature", feature_id)
        payload_imu = IMUDataIdentifier(payload_imu_id or imu_id)
        self.payloads.features[feature_id] = SimpleNamespace(
            get_associated_data_id=lambda: payload_imu
        )
        if link:
            self._relate(("feature", feature_id), ("imu_data", imu_id), FEATURE_TO_IMU)

    def _add_spec(self, spec_id: str, imu_name: str = "LTMM_IMU", *, payload: bool = True) -> None:
        self._index(SPEC, spec_id)
        if payload:
            self.payloads.specs[spec_id] = SimpleNamespace(imu_name=imu_name)

    def _assert_issues(self, validate: Callable, expected_fragments: list) -> DatabaseValidationError:
        with self.assertRaises(DatabaseValidationError) as ctx:
            validate(self.db_manager)
        error = ctx.exception
        self.assertEqual(len(error.issues), len(expected_fragments), error.issues)
        for fragment in expected_fragments:
            self.assertTrue(
                any(fragment in issue for issue in error.issues),
                f"{fragment!r} not in {error.issues}",
            )
        return error


class TestValidateIndexStructure(_ValidatorTestCase):
    def test_allowed_records_and_relations_pass(self) -> None:
        self._add_spec("s1")
        self._add_user("u1")
        self._add_imu("i1", "u1", spec_id="s1")
        self._add_feature("f1", "i1")

        self.assertTrue(self.validator.validate_index_structure(self.db_manager))

    def test_unknown_record_type_is_reported(self) -> None:
        self._index("mystery", "m1")
        self._index("mystery", "m2")

        self._assert_issues(
            self.validator.validate_index_structure, ["2 record(s) have unknown type 'mystery'"]
        )

    def test_disallowed_relations_are_reported(self) -> None:
        self._add_user("u1")
        self._add_imu("i1", "u1", link=False)
        self._relate(("imu_data", "i1"), ("user_data", "u1"), "owned_by")
        self._relate(("user_data", "u1"), ("imu_data", "i1"), IMU_TO_USER)

        self._assert_issues(
            self.validator.validate_index_structure,
            [
                "'owned_by' relation(s) from imu_data to user_data",
                "'imu_to_user' relation(s) from user_data to imu_data",
            ],
        )

    def test_duplicate_links_are_reported(self) -> None:
        self._add_spec("s1")
        self._add_spec("s2")
        self._add_user("u1")
        self._add_user("u2")
        self._add_imu("i1", "u1", spec_id="s1")
        self._relate(("imu_data", "i1"), ("user_data", "u2"), IMU_TO_USER)
        self._relate(("imu_data", "i1"), (SPEC, "s2"), IMU_TO_INSTRUMENT_SPEC)
        self._add_feature("f1", "i1")
        self._index("imu_data", "i2")
        self._relate(("feature", "f1"), ("imu_data", "i2"), FEATURE_TO_IMU)

        self._assert_issues(
            self.validator.validate_index_structure,
            [
                "feature f1 has 2 'feature_to_imu' relations",
                "imu_data i1 has 2 'imu_to_instrument_spec' relations",
                "imu_data i1 has 2 'imu_to_user' relations",
            ],
        )


class TestValidateIMUData(_ValidatorTestCase):
    def test_consistent_index_passes_using_headers_only(self) -> None:
        self._add_user("u1")
        self._add_user("u2")
        self._add_imu("i1", "u1")
        self._add_imu("i2", "u2")

        self.assertTrue(self.validator.validate_imu_data(self.db_manager))
        self.io_router.import_imu.assert_not_called()
        self.assertEqual(self.io_router.import_imu_header.call_count, 2)

    def test_empty_index_passes(self) -> None:
        self.assertTrue(self.validator.validate_imu_data(self.db_manager))

    def test_missing_imu_file_is_reported(self) -> None:
        self._add_user("u1")
        self._add_imu("i1", "u1")
        del self.payloads.imu_headers["i1"]

        self._assert_issues(
            self.validator.validate_imu_data,
            ["IMU i1: Failed to load IMU metadata for i1: No payload file"],
        )

    def test_payload_imu_identifier_mismatch_is_reported(self) -> None:
        self._add_user("u1")
        self._add_imu("i1", "u1", payload_imu_id="i_other")

        self._assert_issues(self.validator.validate_imu_data, ["i_other"])

    def test_payload_user_not_indexed_is_reported(self) -> None:
        self._add_user("u1")
        self._add_imu("i1", "u1", payload_user_id="u_ghost")

        self._assert_issues(
            self.validator.validate_imu_data, ["user u_ghost from payload is not indexed"]
        )

    def test_index_link_disagreeing_with_payload_is_reported(self) -> None:
        self._add_user("u1")
        self._add_user("u2")
        self._add_imu("i1", "u2", payload_user_id="u1")
        self._add_imu("i2", "u1", link=False)

        self._assert_issues(
            self.validator.validate_imu_data,
            [
                "IMU i1: index links user u2, payload names user u1",
                "IMU i2: index links no user, payload names user u1",
            ],
        )

    def test_user_link_to_wrong_record_type_is_reported(self) -> None:
        self._add_user("u1")
        self._add_imu("i1", "u1", link=False)
        self._index("feature", "f1")
        self._relate(("imu_data", "i1"), ("feature", "f1"), IMU_TO_USER)

        self._assert_issues(
            self.validator.validate_imu_data, ["Expected user identifier for IMU relation"]
        )

    def test_unreadable_user_is_reported_once_per_user(self) -> None:
        self._add_user("u1", payload=False)
        self._add_imu("i1", "u1")
        self._add_imu("i2", "u1")

        error = self._assert_issues(self.validator.validate_imu_data, ["User u1"])

        self.assertEqual(self.io_router.import_user.call_count, 1)
        self.assertIn("1 issue(s)", str(error))

    def test_all_issues_are_reported_together(self) -> None:
        self._add_user("u1")
        self._add_imu("i1", "u1")
        self._add_imu("i2", "u1", payload_user_id="u_ghost")
        self._add_imu("i3", "u1", payload_imu_id="i_other")
        self.payloads.unreadable["i1"] = OSError("corrupt header")

        self._assert_issues(
            self.validator.validate_imu_data, ["corrupt header", "u_ghost", "i_other"]
        )


class TestValidateIMUSpecLinks(_ValidatorTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._add_user("u1")

    def test_single_spec_accepts_any_instrument_name_without_loading_it(self) -> None:
        self._add_spec("s1", imu_name="LTMM_LabWalks_IMU")
        self._add_imu("i1", "u1", instrument_name="placeholder", spec_id="s1")

        self.assertTrue(self.validator.validate_imu_data(self.db_manager))
        self.io_router.import_instrument_spec.assert_not_called()

    def test_single_spec_reports_unlinked_imu(self) -> None:
        self._add_spec("s1")
        self._add_imu("i1", "u1")

        self._assert_issues(
            self.validator.validate_imu_data,
            ["IMU i1: not linked to the only indexed instrument spec s1"],
        )

    def test_no_specs_accepts_unlinked_imu(self) -> None:
        self._add_imu("i1", "u1")

        self.assertTrue(self.validator.validate_imu_data(self.db_manager))

    def test_multiple_specs_accept_name_match_ignoring_case_and_whitespace(self) -> None:
        self._add_spec("s_axivity", imu_name="Axivity AX6")
        self._add_spec("s_opal", imu_name="Opal")
        self._add_imu("i1", "u1", instrument_name="  axivity ax6 ", spec_id="s_axivity")
        self._add_imu("i2", "u1", instrument_name="OPAL", spec_id="s_opal")

        self.assertTrue(self.validator.validate_imu_data(self.db_manager))
        self.assertEqual(self.io_router.import_instrument_spec.call_count, 2)

    def test_multiple_specs_report_link_to_non_matching_spec(self) -> None:
        self._add_spec("s_axivity", imu_name="Axivity AX6")
        self._add_spec("s_opal", imu_name="Opal")
        self._add_imu("i1", "u1", instrument_name="Opal", spec_id="s_axivity")

        self._assert_issues(
            self.validator.validate_imu_data,
            ["IMU i1: linked to instrument spec s_axivity, which does not match instrument 'Opal'"],
        )

    def test_multiple_specs_report_unlinked_imu_with_matching_spec(self) -> None:
        self._add_spec("s_axivity", imu_name="Axivity AX6")
        self._add_spec("s_opal", imu_name="Opal")
        self._add_imu("i1", "u1", instrument_name="Opal")

        self._assert_issues(
            self.validator.validate_imu_data,
            ["IMU i1: not linked to instrument spec s_opal"],
        )

    def test_multiple_specs_accept_unlinked_imu_without_matching_spec(self) -> None:
        self._add_spec("s_axivity", imu_name="Axivity AX6")
        self._add_spec("s_opal", imu_name="Opal")
        self._add_imu("i1", "u1", instrument_name="placeholder")

        self.assertTrue(self.validator.validate_imu_data(self.db_manager))

    def test_unloadable_linked_spec_is_left_to_spec_validation(self) -> None:
        self._add_spec("s_axivity", imu_name="Axivity AX6", payload=False)
        self._add_spec("s_opal", imu_name="Opal")
        self._add_imu("i1", "u1", instrument_name="Opal", spec_id="s_axivity")
        self._add_imu("i2", "u1", instrument_name="Opal", spec_id="s_opal")

        self.assertTrue(self.validator.validate_imu_data(self.db_manager))
        self.assertEqual(self.io_router.import_instrument_spec.call_count, 2)


class TestValidateFeatureData(_ValidatorTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._add_user("u1")
        self._add_imu("i1", "u1")
        self._add_imu("i2", "u1")

    def test_consistent_index_passes_reading_each_imu_header_once(self) -> None:
        self._add_feature("f1", "i1")
        self._add_feature("f2", "i1")
        self._add_feature("f3", "i2")

        self.assertTrue(self.validator.validate_feature_data(self.db_manager))
        self.io_router.import_imu.assert_not_called()
        self.assertEqual(self.io_router.import_imu_header.call_count, 2)

    def test_unreadable_feature_payload_is_reported(self) -> None:
        self._add_feature("f1", "i1")
        self.payloads.unreadable["f1"] = OSError("truncated features.h5")

        self._assert_issues(self.validator.validate_feature_data, ["truncated features.h5"])

    def test_payload_imu_not_indexed_is_reported(self) -> None:
        self._add_feature("f1", "i1", payload_imu_id="i_ghost")

        self._assert_issues(
            self.validator.validate_feature_data, ["IMU i_ghost from payload is not indexed"]
        )

    def test_index_link_disagreeing_with_payload_is_reported(self) -> None:
        self._add_feature("f1", "i2", payload_imu_id="i1")
        self._add_feature("f2", "i1", link=False)

        self._assert_issues(
            self.validator.validate_feature_data,
            [
                "Feature f1: index links IMU i2, payload names IMU i1",
                "Feature f2: index links no IMU, payload names IMU i1",
            ],
        )

    def test_imu_link_to_wrong_record_type_is_reported(self) -> None:
        self._add_feature("f1", "i1", link=False)
        self._relate(("feature", "f1"), ("user_data", "u1"), FEATURE_TO_IMU)

        self._assert_issues(
            self.validator.validate_feature_data, ["Expected IMU identifier for feature relation"]
        )

    def test_missing_linked_imu_file_is_reported(self) -> None:
        self._add_feature("f1", "i1")
        del self.payloads.imu_headers["i1"]

        self._assert_issues(self.validator.validate_feature_data, ["IMU i1"])


class TestValidateInstrumentSpecData(_ValidatorTestCase):
    def test_loadable_specs_pass(self) -> None:
        self._add_spec("s1")

        self.assertTrue(self.validator.validate_instrument_spec_data(self.db_manager))

    def test_every_unloadable_spec_is_reported(self) -> None:
        self._add_spec("s1", payload=False)
        self._add_spec("s2", payload=False)

        self._assert_issues(
            self.validator.validate_instrument_spec_data,
            ["Instrument spec s1", "Instrument spec s2"],
        )


class TestValidateDatabase(_ValidatorTestCase):
    def test_consistent_database_passes(self) -> None:
        self._add_spec("s1")
        self._add_user("u1")
        self._add_imu("i1", "u1", spec_id="s1")
        self._add_feature("f1", "i1")

        self.assertTrue(self.validator.validate_database(self.db_manager))

    def test_issues_from_every_scope_are_reported_together(self) -> None:
        self._add_spec("s1", payload=False)
        self._add_user("u1")
        self._add_imu("i1", "u1", spec_id="s1")
        self._add_imu("i2", "u1", payload_user_id="u_ghost", spec_id="s1")
        self._add_feature("f1", "i1")
        self.payloads.unreadable["f1"] = OSError("truncated features.h5")
        self._index("mystery", "m1")

        error = self._assert_issues(
            self.validator.validate_database,
            [
                "unknown type 'mystery'",
                "u_ghost",
                "truncated features.h5",
                "Instrument spec s1",
            ],
        )
        self.assertEqual(error.scope, "Database")


if __name__ == "__main__":
    unittest.main()
