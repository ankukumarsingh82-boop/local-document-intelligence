"""Fail any test that opens a socket beyond this machine."""

from __future__ import annotations

import socket

import pytest

from eval.netguard import install


@pytest.fixture(autouse=True)
def forbid_non_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    original = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", install(original))
