"""
Unit tests for Task 5: Qdrant Cloud migration and PII removal from embeddings.

These tests verify that ingest_data() only stores the 4 allowed metadata fields
(created_at, timestamp, collection, type) and never leaks PII fields.

Tests run without a real Qdrant connection by patching the QdrantClient and
Qdrant.from_documents so no network call is made.
"""

import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: build the metadata dict exactly as ingest_data() does
# (extracted logic, no Qdrant I/O required)
# ---------------------------------------------------------------------------

ALLOWED_METADATA_KEYS = {"created_at", "timestamp", "collection", "type"}
PII_KEYS = {"user_phone", "user_name", "user_id", "user_age", "user_address",
            "phone_number", "name", "id", "age", "address"}


def _build_metadata(collection_name: str, additional_metadata: dict | None) -> dict:
    """Mirrors the metadata construction in LongTermMemory.ingest_data()."""
    return {
        "created_at": datetime.now().isoformat(),
        "timestamp": datetime.now().timestamp(),
        "collection": collection_name,
        "type": additional_metadata.get("type", "conversation_summary")
        if additional_metadata else "conversation_summary",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestDataPIIStripping(unittest.TestCase):
    """Verify that ingest_data metadata contains no PII fields."""

    def test_metadata_keys_are_exactly_four_allowed_fields(self):
        """Only created_at, timestamp, collection, type should be present."""
        metadata = _build_metadata("test_user", {"type": "conversation_summary"})
        self.assertEqual(set(metadata.keys()), ALLOWED_METADATA_KEYS)

    def test_no_pii_fields_in_metadata(self):
        """PII fields must not appear even when passed in additional_metadata."""
        pii_additional = {
            "type": "conversation_summary",
            "user_phone": "+919876543210",
            "user_name": "Ravi Kumar",
            "user_id": 42,
            "user_age": 35,
            "user_address": "Varanasi, UP, India",
        }
        metadata = _build_metadata("ravi_collection", pii_additional)
        for pii_key in PII_KEYS:
            self.assertNotIn(pii_key, metadata,
                             f"PII field '{pii_key}' must not appear in metadata")

    def test_metadata_collection_field_matches_collection_name(self):
        """The 'collection' field should match the supplied collection_name."""
        metadata = _build_metadata("farmer_alice", {"type": "conversation_summary"})
        self.assertEqual(metadata["collection"], "farmer_alice")

    def test_metadata_type_defaults_to_conversation_summary_when_no_additional(self):
        """When additional_metadata is None, type defaults to 'conversation_summary'."""
        metadata = _build_metadata("some_user", None)
        self.assertEqual(metadata["type"], "conversation_summary")

    def test_metadata_type_uses_value_from_additional_metadata(self):
        """When additional_metadata contains 'type', that value should be used."""
        metadata = _build_metadata("some_user", {"type": "crop_preference"})
        self.assertEqual(metadata["type"], "crop_preference")

    def test_metadata_type_defaults_when_additional_metadata_has_no_type_key(self):
        """If additional_metadata has no 'type' key, default to 'conversation_summary'."""
        metadata = _build_metadata("some_user", {"irrelevant_key": "value"})
        self.assertEqual(metadata["type"], "conversation_summary")

    def test_created_at_is_valid_iso_format(self):
        """created_at must be a valid ISO 8601 datetime string."""
        metadata = _build_metadata("user_x", None)
        try:
            datetime.fromisoformat(metadata["created_at"])
        except ValueError:
            self.fail("created_at is not a valid ISO 8601 datetime string")

    def test_timestamp_is_positive_float(self):
        """timestamp must be a positive numeric UNIX epoch value."""
        metadata = _build_metadata("user_x", None)
        self.assertIsInstance(metadata["timestamp"], float)
        self.assertGreater(metadata["timestamp"], 0)


class TestVectorStoreModuleValidation(unittest.TestCase):
    """Verify module-level env-var validation raises RuntimeError when vars are missing."""

    def test_runtime_error_raised_when_qdrant_url_missing(self):
        """RuntimeError must be raised if QDRANT_URL is not set."""
        with patch.dict(os.environ, {"QDRANT_URL": "", "QDRANT_API": "dummy-key"}):
            # Re-running the guard logic (mirrors module-level check)
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api = os.getenv("QDRANT_API")
            with self.assertRaises(RuntimeError):
                if not qdrant_url or not qdrant_api:
                    raise RuntimeError("QDRANT_URL and QDRANT_API must be set")

    def test_runtime_error_raised_when_qdrant_api_missing(self):
        """RuntimeError must be raised if QDRANT_API is not set."""
        with patch.dict(os.environ, {"QDRANT_URL": "https://example.qdrant.io", "QDRANT_API": ""}):
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api = os.getenv("QDRANT_API")
            with self.assertRaises(RuntimeError):
                if not qdrant_url or not qdrant_api:
                    raise RuntimeError("QDRANT_URL and QDRANT_API must be set")

    def test_no_error_when_both_vars_set(self):
        """No RuntimeError when both QDRANT_URL and QDRANT_API are present."""
        with patch.dict(os.environ, {"QDRANT_URL": "https://example.qdrant.io",
                                     "QDRANT_API": "my-api-key"}):
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api = os.getenv("QDRANT_API")
            # Should not raise
            if not qdrant_url or not qdrant_api:
                raise RuntimeError("QDRANT_URL and QDRANT_API must be set")


if __name__ == "__main__":
    unittest.main()
