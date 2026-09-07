"""Tests for HERMES_READ_DENY_ROOT, the read-side sibling of
HERMES_WRITE_SAFE_ROOT.

Some Hermes deployments run profiles that represent someone other than the
operator (a third-party DM/group channel, a shared trial sandbox). Those
profiles' core ``terminal``/``read_file`` tools have no restriction from
reading the operator's own personal files (a notes vault, financial records,
anything outside Hermes entirely) — the built-in denylist in
``get_read_block_error`` only covers Hermes's own credential stores, not the
operator's files. ``HERMES_READ_DENY_ROOT`` lets a profile's own ``.env`` add
one or more directories to the deny list on top of those built-ins.

Same defense-in-depth caveat as every other entry in this module: this stops
the Read/Glob/Grep tools and the terminal tool's recognized read-only
builtins, not an arbitrary shell command a model chooses to run instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def vault(tmp_path):
    """A scratch 'vault' directory with a file inside it, and a sibling
    directory that must NOT be affected by a deny root pointed at the vault."""
    v = tmp_path / "Trinity-Wiki"
    (v / "atlas").mkdir(parents=True)
    (v / "atlas" / "Someone.md").write_text("confidential", encoding="utf-8")
    (v / "projects").mkdir()
    (v / "projects" / "Plan.md").write_text("not confidential", encoding="utf-8")
    sibling = tmp_path / "Trinity-Wiki-Extra"
    sibling.mkdir()
    (sibling / "x.md").write_text("unrelated", encoding="utf-8")
    return v, sibling


def test_unset_env_var_denies_nothing(vault, monkeypatch):
    """No HERMES_READ_DENY_ROOT set at all — the mechanism is a no-op,
    same as HERMES_WRITE_SAFE_ROOT's own unset behavior."""
    from agent.file_safety import get_read_block_error

    v, _ = vault
    monkeypatch.delenv("HERMES_READ_DENY_ROOT", raising=False)
    assert get_read_block_error(str(v / "atlas" / "Someone.md")) is None


def test_file_under_deny_root_is_blocked(vault, monkeypatch):
    from agent.file_safety import get_read_block_error

    v, _ = vault
    monkeypatch.setenv("HERMES_READ_DENY_ROOT", str(v))
    err = get_read_block_error(str(v / "atlas" / "Someone.md"))
    assert err is not None
    assert "HERMES_READ_DENY_ROOT" in err
    assert "not a security boundary" in err


def test_the_deny_root_itself_is_blocked(vault, monkeypatch):
    """The root path exactly, not only something under it."""
    from agent.file_safety import get_read_block_error

    v, _ = vault
    monkeypatch.setenv("HERMES_READ_DENY_ROOT", str(v))
    assert get_read_block_error(str(v)) is not None


def test_file_outside_deny_root_is_not_blocked(vault, monkeypatch):
    from agent.file_safety import get_read_block_error

    v, _ = vault
    monkeypatch.setenv("HERMES_READ_DENY_ROOT", str(v))
    assert get_read_block_error("/tmp/some-unrelated-file.md") is None


def test_similarly_prefixed_sibling_directory_is_not_blocked(vault, monkeypatch):
    """A directory whose NAME starts with the deny root's name, but is not
    actually nested under it, must not false-match on a bare string prefix
    (e.g. 'Trinity-Wiki' must not swallow 'Trinity-Wiki-Extra')."""
    from agent.file_safety import get_read_block_error

    v, sibling = vault
    monkeypatch.setenv("HERMES_READ_DENY_ROOT", str(v))
    assert get_read_block_error(str(sibling / "x.md")) is None


def test_multiple_roots_separated_by_pathsep(vault, tmp_path, monkeypatch):
    from agent.file_safety import get_read_block_error

    v, _ = vault
    other = tmp_path / "other-sensitive"
    other.mkdir()
    (other / "y.md").write_text("also confidential", encoding="utf-8")
    monkeypatch.setenv(
        "HERMES_READ_DENY_ROOT", os.pathsep.join([str(v), str(other)])
    )
    assert get_read_block_error(str(v / "projects" / "Plan.md")) is not None
    assert get_read_block_error(str(other / "y.md")) is not None


def test_get_read_deny_roots_resolves_and_dedupes(tmp_path, monkeypatch):
    from agent.file_safety import get_read_deny_roots

    a = tmp_path / "a"
    a.mkdir()
    monkeypatch.setenv(
        "HERMES_READ_DENY_ROOT", os.pathsep.join([str(a), str(a) + os.sep])
    )
    roots = get_read_deny_roots()
    assert roots == {os.path.realpath(str(a))}


def test_get_read_deny_roots_empty_when_unset(monkeypatch):
    from agent.file_safety import get_read_deny_roots

    monkeypatch.delenv("HERMES_READ_DENY_ROOT", raising=False)
    assert get_read_deny_roots() == set()
