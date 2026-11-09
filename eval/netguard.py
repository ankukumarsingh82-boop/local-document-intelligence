"""Refuse a test that opens a socket to anything but this machine."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable

LOOPBACK_NAMES = frozenset({"localhost"})


def host_of(address: object) -> str | None:
    """The host of an IP address. A filesystem socket has no host."""
    if isinstance(address, tuple) and address:
        return str(address[0])
    if isinstance(address, str) and address and not address.startswith(("/", "\x00")):
        return address
    return None


def is_loopback(host: str) -> bool:
    name = host.strip("[]").casefold()
    if name in LOOPBACK_NAMES:
        return True
    try:
        parsed = ipaddress.ip_address(name)
    except ValueError:
        return False
    return bool(parsed.is_loopback)


def check_address(address: object) -> None:
    host = host_of(address)
    if host is None or is_loopback(host):
        return
    raise AssertionError(f"non-loopback socket: {host}")


def install(connect: Callable[..., object]) -> Callable[..., object]:
    """Wrap socket.connect. The real connect runs only after the check."""

    def guarded(self: socket.socket, address: object) -> object:
        check_address(address)
        return connect(self, address)

    return guarded
