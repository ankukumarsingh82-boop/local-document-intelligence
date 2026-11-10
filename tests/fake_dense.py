"""A dense list for tests. The product path never constructs this."""

from __future__ import annotations


class FakeDenseRetriever:
    def __init__(self, ordered: list[str] | None = None) -> None:
        self.ordered = list(ordered or [])
        self.calls = 0

    def search(self, frame: object, depth: int) -> list[str]:
        self.calls += 1
        return self.ordered[:depth]
