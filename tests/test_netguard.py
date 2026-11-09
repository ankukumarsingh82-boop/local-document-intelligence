"""The socket guard rejects a public address before any connection."""

from __future__ import annotations

import socket

import pytest

from eval.netguard import check_address, install


def test_a_public_address_is_refused() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError, match="non-loopback"):
            sock.connect(("203.0.113.5", 9))
    finally:
        sock.close()


def test_loopback_names_pass_the_check() -> None:
    check_address(("127.0.0.1", 9))
    check_address(("::1", 9, 0, 0))
    check_address("localhost")


def test_the_real_connect_is_not_called(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[object] = []

    def original(self: socket.socket, address: object) -> None:
        called.append(address)

    monkeypatch.setattr(socket.socket, "connect", install(original))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError, match="non-loopback"):
            sock.connect(("198.51.100.8", 9))
    finally:
        sock.close()
    assert called == []
