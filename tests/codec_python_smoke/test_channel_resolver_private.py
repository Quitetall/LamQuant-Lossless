"""Owner-local contracts for channel resolver private normalization."""

from lamquant_codec.channel_resolver import _try_resolve_atom


def test_resolves_canonical() -> None:
    assert _try_resolve_atom("Fp1") == "Fp1"


def test_resolves_case_insensitive() -> None:
    assert _try_resolve_atom("fp1") == "Fp1"


def test_strips_trailing_dots() -> None:
    assert _try_resolve_atom("Fp1.") == "Fp1"


def test_returns_none_unknown() -> None:
    assert _try_resolve_atom("zzz") is None
