"""Logits processors for local generation.

The Checks filter is used only by the grammar condition.
The JSON filter is used by ldi ask. It keeps the reply on one object
whose evidence ids come from the list the program built.
"""

from __future__ import annotations

import re

_KINDS = ("calc", "date", "within", "cite")
_COMPLETE = re.compile(rf"^- (?:{'|'.join(_KINDS)}):.*$")


def suffix_ok(suffix: str) -> bool:
    """True when the text after Checks: can still become valid check lines."""
    body = suffix.lstrip("\n")
    if body == "":
        return True
    lines = body.split("\n")
    finished = lines[:-1]
    last = lines[-1]
    for line in finished:
        if not _COMPLETE.match(line):
            return False
    if last == "":
        return True
    return _line_prefix(last)


def checks_processor(tokenizer, prompt_length: int, eos_id: int | None):
    """Force greedy picks, after Checks:, to stay on a readable check line."""
    import mlx.core as mx

    def processor(tokens, logits):
        ids = _ids(tokens)
        if len(ids) <= prompt_length:
            return logits
        text = tokenizer.decode(ids[prompt_length:])
        if "Checks:" not in text:
            return logits
        suffix = text.split("Checks:")[-1]
        flat = logits.reshape(-1)
        order = mx.argsort(flat).tolist()
        chosen = None
        for index in reversed(order[-128:]):
            candidate = tokenizer.decode(ids[prompt_length:] + [int(index)])
            piece = candidate.split("Checks:")[-1]
            if suffix_ok(piece):
                chosen = int(index)
                break
        if chosen is None and suffix_ok(suffix) and eos_id is not None:
            chosen = eos_id
        if chosen is None:
            return logits
        masked = mx.full(flat.shape, -1e9)
        masked[chosen] = flat[chosen]
        return masked.reshape(logits.shape)

    return processor


def _line_prefix(line: str) -> bool:
    for kind in _KINDS:
        head = f"- {kind}:"
        if head.startswith(line) or line.startswith(head):
            return True
    return False


def _ids(tokens) -> list[int]:
    raw = tokens.tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(item) for item in raw]


_STATUSES = ("answered", "not_found", "conflict")
_MASK = -1e9


class _Done(Exception):
    pass


class _Bad(Exception):
    pass


class AnswerJSON:
    """One finished object. The calc string is empty when the model left it out."""

    def __init__(self, status: str, evidence: tuple[str, ...], answer: str, calc: str | None):
        self.status = status
        self.evidence = evidence
        self.answer = answer
        self.calc = calc


class _View:
    def __init__(self, closed: bool, status: str, evidence: tuple[str, ...], answer: str, calc: str | None):
        self.closed = closed
        self.status = status
        self.evidence = evidence
        self.answer = answer
        self.calc = calc


def json_prefix_ok(text: str, allowed: set[str]) -> bool:
    """True when text can still be finished as one evidence object."""
    return inspect_json(text, allowed) is not None


def parse_answer_json(text: str, allowed: set[str]) -> AnswerJSON | None:
    view = inspect_json(text.strip(), allowed)
    if view is None or not view.closed or view.status not in _STATUSES:
        return None
    return AnswerJSON(view.status, view.evidence, view.answer, view.calc)


def inspect_json(text: str, allowed: set[str]) -> _View | None:
    if text == "":
        return _View(False, "", (), "", None)
    if text.lstrip() == "":
        return None
    parser = _Parser(text.lstrip(), set(allowed))
    try:
        parser.parse()
    except _Done:
        return parser.view(False)
    except _Bad:
        return None
    return parser.view(True)


def expression_prefix_ok(expr: str, allowed: set[str]) -> bool:
    """True when expr can still become ids combined with + - * / and parentheses."""
    depth = 0
    expect = True
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            if not expect:
                return False
            depth += 1
            i += 1
            continue
        if ch == ")":
            if expect or depth <= 0:
                return False
            depth -= 1
            expect = False
            i += 1
            continue
        if ch in "+-*/":
            if expect:
                return False
            expect = True
            i += 1
            continue
        if ch == "E":
            if not expect:
                return False
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            ident = expr[i:j]
            if j == len(expr):
                return any(item.startswith(ident) for item in allowed)
            if ident not in allowed:
                return False
            expect = False
            i = j
            continue
        return False
    return True


def expression_complete(expr: str, allowed: set[str]) -> bool:
    if not expr.strip() or not expression_prefix_ok(expr, allowed):
        return False
    tokens = _lex(expr, allowed)
    if not tokens:
        return False
    try:
        index = _consume_expr(tokens, 0)
    except _Bad:
        return False
    return index == len(tokens)


def choose_free(text: str, ranked: list[tuple[float, str, int]], allowed: set[str]) -> int | None:
    """Pick the best already-sorted piece that can continue an answer string."""
    for _score, piece, index in ranked[:96]:
        if piece and json_prefix_ok(text + piece, allowed):
            return index
    return None


def choose_structural(text: str, ranked: list[tuple[float, str, int]], allowed: set[str]) -> int | None:
    """Pick the highest-scoring token that continues the JSON structure."""
    starts = _start_chars(text, allowed)
    ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
    for _score, piece, index in ordered:
        if not _relevant(piece, starts):
            continue
        if json_prefix_ok(text + piece, allowed):
            return index
    return None


def json_processor(tokenizer, prompt_length: int, allowed: set[str], eos_id: int | None):
    """Force each new token to keep a valid prefix of the evidence object."""
    strings = _token_strings(tokenizer)
    buckets = _buckets(strings)
    allowed = set(allowed)
    prompt_text: dict[str, str | None] = {"value": None}

    def processor(tokens, logits):
        import mlx.core as mx

        ids = _ids(tokens)
        if len(ids) < prompt_length:
            return logits
        if prompt_text["value"] is None:
            prompt_text["value"] = tokenizer.decode(ids[:prompt_length])
        full = tokenizer.decode(ids)
        prefix = prompt_text["value"] or ""
        text = full[len(prefix) :] if full.startswith(prefix) else tokenizer.decode(ids[prompt_length:])
        flat = logits.reshape(-1)
        view = inspect_json(text, allowed)
        if view is not None and view.closed:
            masked = mx.full(flat.shape, _MASK)
            if eos_id is not None and 0 <= eos_id < flat.shape[0]:
                masked[int(eos_id)] = flat[int(eos_id)]
            return masked.reshape(logits.shape)
        values = flat.tolist()
        if _free(text, allowed):
            order = mx.argsort(flat).tolist()
            top = list(reversed(order[-96:]))
            ranked = [(values[index], strings[index] if index < len(strings) else "", index) for index in top]
            chosen = choose_free(text, ranked, allowed)
        else:
            chosen = None
        if chosen is None:
            starts = _start_chars(text, allowed)
            seen: set[int] = set()
            ranked = []
            for ch in starts:
                for index in buckets.get(ch, ()):
                    if index in seen or index >= len(strings):
                        continue
                    seen.add(index)
                    piece = strings[index]
                    if piece and len(piece) <= 80:
                        ranked.append((values[index], piece, index))
            chosen = choose_structural(text, ranked, allowed)
        if chosen is None:
            return logits
        masked = mx.full(flat.shape, _MASK)
        masked[chosen] = flat[chosen]
        return masked.reshape(logits.shape)

    return processor


class _Parser:
    def __init__(self, text: str, allowed: set[str]):
        self.text = text
        self.n = len(text)
        self.i = 0
        self.allowed = allowed
        self.status = ""
        self.ids: list[str] = []
        self.answer = ""
        self.calc: str | None = None

    def view(self, closed: bool) -> _View:
        return _View(closed, self.status, tuple(self.ids), self.answer, self.calc)

    def parse(self) -> None:
        self._literal('{"status":"')
        self._status()
        self._literal('","evidence":[')
        self._evidence()
        self._literal(',"answer":"')
        self.answer = self._raw(False)
        if self.i >= self.n:
            raise _Done()
        if self.text[self.i] == "}":
            self.i += 1
            if self.i != self.n:
                raise _Bad()
            return
        self._literal(',"calc":')
        self._calc()
        self._literal("}")
        if self.i != self.n:
            raise _Bad()

    def _literal(self, literal: str) -> None:
        for ch in literal:
            if self.i >= self.n:
                raise _Done()
            if self.text[self.i] != ch:
                raise _Bad()
            self.i += 1

    def _status(self) -> None:
        start = self.i
        while self.i < self.n:
            piece = self.text[start : self.i + 1]
            if any(option.startswith(piece) for option in _STATUSES):
                self.i += 1
                continue
            break
        piece = self.text[start : self.i]
        if piece in _STATUSES:
            self.status = piece
            return
        if self.i >= self.n and any(option.startswith(piece) for option in _STATUSES):
            raise _Done()
        raise _Bad()

    def _evidence(self) -> None:
        if self.i >= self.n:
            raise _Done()
        if self.text[self.i] == "]":
            self.i += 1
            return
        while True:
            self._one_id()
            if self.i >= self.n:
                raise _Done()
            if self.text[self.i] == "]":
                self.i += 1
                return
            self._literal(",")

    def _one_id(self) -> None:
        self._literal('"')
        start = self.i
        while self.i < self.n:
            ch = self.text[self.i]
            if ch == '"':
                break
            if ch == "E" and self.i == start:
                self.i += 1
                continue
            if ch.isdigit() and self.i > start:
                self.i += 1
                continue
            raise _Bad()
        ident = self.text[start : self.i]
        if self.i >= self.n:
            if any(item.startswith(ident) for item in self.allowed):
                raise _Done()
            raise _Bad()
        if ident not in self.allowed:
            raise _Bad()
        self.i += 1
        self.ids.append(ident)

    def _raw(self, expr: bool) -> str:
        chars: list[str] = []
        while self.i < self.n:
            ch = self.text[self.i]
            if ch == '"':
                text = "".join(chars)
                if expr and not expression_complete(text, self.allowed):
                    raise _Bad()
                self.i += 1
                return text
            if ch == "\\":
                if expr or self.i + 1 >= self.n:
                    if self.i + 1 >= self.n:
                        raise _Done()
                    raise _Bad()
                chars.append({"n": "\n", "t": "\t", "r": "\r"}.get(self.text[self.i + 1], self.text[self.i + 1]))
                self.i += 2
                continue
            if ord(ch) < 32:
                raise _Bad()
            chars.append(ch)
            self.i += 1
        text = "".join(chars)
        if expr and not expression_prefix_ok(text, self.allowed):
            raise _Bad()
        raise _Done()

    def _calc(self) -> None:
        if self.i >= self.n:
            raise _Done()
        if self.text[self.i] == '"':
            self.i += 1
            self.calc = self._raw(True)
            return
        if self.text[self.i] == "{":
            self._literal('{"tool":"calculator","args":{"expr":"')
            self.calc = self._raw(True)
            self._literal("}}")
            return
        rest = self.text[self.i :]
        if '"' .startswith(rest) or "{".startswith(rest):
            raise _Done()
        raise _Bad()


def _lex(expr: str, allowed: set[str]) -> list[str] | None:
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
            continue
        if expr[i] in "+-*/()":
            tokens.append(expr[i])
            i += 1
            continue
        match = re.match(r"E\d+", expr[i:])
        if match is None or match.group(0) not in allowed:
            return None
        tokens.append(match.group(0))
        i += match.end()
    return tokens


def _consume_expr(tokens: list[str], index: int) -> int:
    index = _consume_term(tokens, index)
    while index < len(tokens) and tokens[index] in "+-":
        index = _consume_term(tokens, index + 1)
    return index


def _consume_term(tokens: list[str], index: int) -> int:
    index = _consume_factor(tokens, index)
    while index < len(tokens) and tokens[index] in "*/":
        index = _consume_factor(tokens, index + 1)
    return index


def _consume_factor(tokens: list[str], index: int) -> int:
    if index >= len(tokens):
        raise _Bad()
    if tokens[index] == "(":
        index = _consume_expr(tokens, index + 1)
        if index >= len(tokens) or tokens[index] != ")":
            raise _Bad()
        return index + 1
    if not tokens[index].startswith("E"):
        raise _Bad()
    return index + 1


def _free(text: str, allowed: set[str]) -> bool:
    return json_prefix_ok(text + "q", allowed)


def _start_chars(text: str, allowed: set[str]) -> set[str]:
    found: set[str] = set()
    for ch in "{}[]\",:-+*/()_ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        if json_prefix_ok(text + ch, allowed):
            found.add(ch)
    return found


def _relevant(piece: str, starts: set[str]) -> bool:
    if not piece:
        return False
    if piece[0] in starts:
        return True
    stripped = piece.lstrip()
    return bool(stripped) and piece[0].isspace() and stripped[0] in starts


def _vocab_size(tokenizer) -> int:
    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return value
    try:
        return len(tokenizer)
    except TypeError:
        return 0


def _token_strings(tokenizer) -> list[str]:
    cached = getattr(tokenizer, "_ldi_token_strings", None)
    if cached is not None:
        return cached
    strings: list[str] = []
    for index in range(_vocab_size(tokenizer)):
        try:
            piece = tokenizer.decode([index])
        except Exception:
            piece = ""
        strings.append(piece if isinstance(piece, str) else "")
    try:
        setattr(tokenizer, "_ldi_token_strings", strings)
    except Exception:
        pass
    return strings


def _buckets(strings: list[str]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for index, piece in enumerate(strings):
        if not piece:
            continue
        keys = {piece[0]}
        stripped = piece.lstrip()
        if stripped:
            keys.add(stripped[0])
        for key in keys:
            buckets.setdefault(key, []).append(index)
    return buckets
