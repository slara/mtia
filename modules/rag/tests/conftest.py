"""Shared test fixtures."""
from __future__ import annotations

import uuid

import pytest


@pytest.fixture
def ephemeral_client() -> str:
    """A unique Qdrant client namespace to keep tests isolated."""
    return "test_" + uuid.uuid4().hex[:12]


@pytest.fixture(autouse=True)
def _documents_tmp_dir(tmp_path, monkeypatch):
    """Redirect DOCUMENTS_DIR to a tmp dir so storage tests don't stomp on disk."""
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path / "documents"))
    yield
