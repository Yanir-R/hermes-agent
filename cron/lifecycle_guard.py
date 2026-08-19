"""Gateway lifecycle guard for cron job creation (#30719).

An agent running inside a gateway can schedule a cron job that calls
``hermes gateway restart`` (or ``launchctl kickstart ai.hermes.gateway``
or ``systemctl restart hermes-gateway``).  When the cron fires, the
gateway dies, the supervisor (launchd KeepAlive / systemd Restart=)
revives it, auto-resume picks up the offending session, and the resumed
turn re-runs the same logic — a SIGTERM-respawn loop every ~10 seconds
until manually broken.

This module rejects cron job specs whose prompt or script contains a
direct shell-level gateway-lifecycle command.  It is enforced at
``cron.jobs.create_job`` so it fires on every job-creation path: the
``hermes cron create`` CLI subcommand AND the agent's ``cronjob`` model
tool (which calls ``create_job`` directly, bypassing the CLI layer).

The pattern is intentionally command-shaped: it anchors on a concrete
command identifier (``hermes gateway``, ``launchctl ... hermes-gateway``,
``systemctl ... hermes-gateway``, ``pkill`` against the gateway) so it
cannot fire on prose.  A cron ``prompt`` is fed to a future LLM, not a
shell, so an over-broad substring match on English ("Kong API gateway
autoscaling and restart behavior") would produce a high false-positive
rate without preventing the actual foot-gun, which requires a real
command shape.

This is a defence-in-depth layer.  ``tools/terminal_tool.py`` blocks direct
commands and shell scripts they reference when ``_HERMES_GATEWAY=1``. It also
rejects ``launchctl submit`` in gateway sessions because launchd treats that
primitive as a persistent KeepAlive job, not a one-shot task. ``hermes gateway
stop|restart`` separately refuse to self-target from inside the gateway.
Blocking cron specs at creation time as well means the agent gets an immediate,
informative rejection instead of scheduling a job that will only fail
(silently) when it fires.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Callable, Iterator, Optional


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle. Each branch
# is anchored on a concrete command identifier so a match can only fire on
# actual shell-command-shaped strings, not on prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: `hermes gateway restart|stop` — the canonical foot-gun.
    # `start` is intentionally excluded: starting a gateway from inside a
    # gateway is benign (a no-op or "already running" error), and a
    # legitimate cron job might start a sibling profile's gateway.
    r"(?:hermes\s+gateway\s+(?:restart|stop))"
    # Branch B: launchctl ops on a hermes-gateway label. macOS launchd
    # labels look like `ai.hermes.gateway` / `hermes-gateway`. Requiring the
    # gateway identifier prevents blocking unrelated hermes services (e.g.
    # `launchctl unload ai.hermes.update-checker.plist`).
    # `submit` and `bootstrap` are included alongside the direct verbs
    # (kickstart/etc.): `launchctl submit -l ai.hermes.gateway-<suffix> --
    # <helper-script>` (or `launchctl bootstrap gui/<uid> <plist>`) creates
    # a NEW keepalive job wrapping an arbitrary helper, which is how a
    # blocked direct restart/kill gets laundered into a persistent restart
    # loop instead (#62891) — same foot-gun, indirect shape. Neutral-label
    # submissions that dodge this text anchor are caught separately by
    # `contains_launchctl_submit_command` (execution-aware, label-independent).
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart|submit|bootstrap)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill / kill targeting the hermes gateway process. Both
    # token orders because real reproductions show both.
    r"|(?:p?kill\b[^\n]*\bhermes\b[^\n]*\bgateway)"
    r"|(?:p?kill\b[^\n]*\bgateway\b[^\n]*\bhermes)"
)


# A backslash immediately followed by a newline is a POSIX shell line
# continuation — the shell joins the two lines before parsing. Every branch
# above uses `[^\n]*` between its verb and the gateway identifier so the
# match can't span unrelated lines of a longer cron prompt/script, but that
# also means a real multi-line shell invocation split across continuation
# lines (e.g. `launchctl submit \` / `  -l ai.hermes.gateway-... \` / `  -- ...`,
# the exact reported shape in #62891) would otherwise slip past. Collapse
# continuations to a single space before matching, mirroring what the shell
# itself does, rather than loosening `[^\n]*` and risking false positives
# across genuinely separate lines.
_SHELL_LINE_CONTINUATION = re.compile(r"\\\r?\n[ \t]*")


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern."""
    if not text:
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    return bool(_GATEWAY_LIFECYCLE_PATTERN.search(normalized))


_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ksh", "zsh"})

# THE ONE PLACE THIS RULE IS WRITTEN. `cron/scheduler.py` picks a job's
# interpreter by extension -- .sh/.bash under bash, everything else under
# python -- and deliberately does NOT honour the shebang. This guard must make
# the SAME choice, for the reason `_resolve_script_path` mirrors the
# scheduler's path resolution: a guard that scans a file as shell while the
# scheduler runs it as Python is auditing a program that will not exist.
SHELL_SCRIPT_SUFFIXES = frozenset({".sh", ".bash"})

# Callables whose string arguments reach a shell or an argv. A closed list of
# stdlib exec surfaces rather than a heuristic: a name absent from it
# contributes nothing, which under-reports rather than over-reports. That is
# the right direction for a SCAN, because `terminal_tool`'s in-gateway block
# and the gateway's own self-target refusal are the layers that actually stop
# the loop -- see this module's docstring on defence in depth.
_EXEC_CALL_NAMES = frozenset({
    "system", "popen", "run", "call", "check_call", "check_output",
    "Popen", "getoutput", "getstatusoutput", "spawn", "execv", "execvp",
})
_SHELL_OPTIONS_WITH_VALUES = frozenset({"-O", "+O", "-o", "+o"})
_MAX_REFERENCED_SCRIPT_BYTES = 1024 * 1024
_MAX_REFERENCED_SCRIPT_DEPTH = 8
_CONTROL_CHARS = frozenset(";&|()")




_ReadRemoteScriptFn = Callable[[str], Optional[str]]


def _literal_strings(node) -> Iterator[str]:
    """Every string constant reachable from *node* without leaving the literal.

    Sequence elements are joined by the caller so an argv form reads as the
    command line it becomes: ["hermes", "gateway", "restart"] has to look like
    `hermes gateway restart` for the existing pattern to see it. f-string
    interpolations are unknowable here and contribute nothing, the same limit
    as a concatenation.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            yield node.value
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for element in node.elts:
            yield from _literal_strings(element)
    elif isinstance(node, ast.JoinedStr):
        for value in node.values:
            yield from _literal_strings(value)


def _python_exec_arguments(source: str) -> Iterator[str]:
    """Command text a Python source could hand to a shell or an argv.

    Only literals passed to a name in `_EXEC_CALL_NAMES`, matched on the
    attribute rather than the module so `os.system`, `subprocess.run` and a
    bare `run` from `from subprocess import run` all resolve alike.

    WHAT THIS DELIBERATELY DOES NOT SEE, stated because scanning less is the
    point: a command assembled through a variable, through concatenation, or
    passed to a wrapper whose own name is not in the list. All three are missed
    by the raw-text scan this replaces for Python too -- and that scan
    additionally misses the IDIOMATIC argv form, which this catches. Measured
    on the reporting machine: raw text found one match across 42 Python files
    and it was a false positive. Zero true positives.
    """
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in _EXEC_CALL_NAMES:
            continue
        for argument in [*node.args, *(kw.value for kw in node.keywords)]:
            parts = list(_literal_strings(argument))
            if parts:
                yield " ".join(parts)


def scannable_text(raw: str, path: Optional[Path]) -> str:
    """The part of *raw* that can actually reach a shell.

    Shell scripts are scanned whole, byte for byte, exactly as before: in a
    shell every line IS a command, so there is nothing to narrow and narrowing
    would cost real coverage -- `pkill -f "hermes.*gateway"` lives inside
    quotes by necessity, and branch D exists for it.

    Everything else is Python by the scheduler's dispatch, where a raw scan is
    a category error: it reads prose, docstrings and fault messages as
    commands. That produced the exact reverse of a guard -- refusing a job for
    a remediation string telling a human what to run, while missing
    `subprocess.run(["hermes", "gateway", "restart"])`.

    A file that will not parse falls back to the raw scan: unparseable means
    unknowable, and unknowable fails closed.
    """
    if path is not None and path.suffix.lower() in SHELL_SCRIPT_SUFFIXES:
        return raw
    try:
        return "\n".join(_python_exec_arguments(raw))
    except (SyntaxError, ValueError, RecursionError):
        return raw


def _iter_command_segments(command: str) -> Iterator[list[str]]:
    """Yield shell-tokenized command segments, honoring quotes and comments."""
    normalized = command.replace("\\\n", "")
    for line in normalized.splitlines() or [normalized]:
        try:
            lexer = shlex.shlex(
                line,
                posix=True,
                punctuation_chars=";&|()",
            )
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            continue

        segment: list[str] = []
        for token in tokens:
            if token and set(token) <= _CONTROL_CHARS:
                if segment:
                    yield segment
                    segment = []
                continue
            segment.append(token)
        if segment:
            yield segment


def _command_token_index(segment: list[str]) -> Optional[int]:
    """Return the executable token index after simple env assignments."""
    for index, token in enumerate(segment):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue
        return index
    return None


def contains_launchctl_submit_command(command: str) -> bool:
    """Detect an executed ``launchctl submit``/``bootstrap``, not quoted text.

    Label-independent by design: the label of a submitted/bootstrapped job is
    chosen by whoever writes it, so a neutral name (``ai.hermes.svc-reload-tmp``)
    defeats any label-anchored regex (#62891, second reproduction). Both verbs
    register a NEW persistent launchd job (``submit`` jobs get KeepAlive
    semantics; ``bootstrap`` loads an arbitrary plist), which is never safe to
    do from inside the gateway process.
    """
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        if Path(segment[index]).name == "launchctl":
            arguments = segment[index + 1 :]
            if arguments and arguments[0].lower() in {"submit", "bootstrap"}:
                return True
    return False


def _resolve_terminal_script_path(candidate: str, cwd: Optional[str]) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path(cwd or Path.cwd()) / path
    return path


def _iter_referenced_shell_scripts(
    command: str,
    *,
    cwd: Optional[str] = None,
) -> Iterator[Path]:
    """Yield scripts executed directly or through a POSIX shell."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        executable = segment[index]
        executable_name = Path(executable).name

        if executable_name in {".", "source"}:
            if len(segment) > index + 1:
                yield _resolve_terminal_script_path(segment[index + 1], cwd)
            continue

        if executable_name in _SHELL_EXECUTABLES:
            arguments = segment[index + 1 :]
            arg_index = 0
            while arg_index < len(arguments):
                argument = arguments[arg_index]
                if argument == "--":
                    arg_index += 1
                    break
                if argument in {"-c", "--command"}:
                    break
                if argument in _SHELL_OPTIONS_WITH_VALUES:
                    arg_index += 2
                    continue
                if argument.startswith("-"):
                    arg_index += 1
                    continue
                break
            if arg_index < len(arguments) and arguments[arg_index] not in {
                "-c",
                "--command",
            }:
                yield _resolve_terminal_script_path(arguments[arg_index], cwd)
            continue

        if "/" in executable or executable.endswith((".sh", ".bash", ".zsh")):
            yield _resolve_terminal_script_path(executable, cwd)


def _iter_shell_command_payloads(command: str) -> Iterator[str]:
    """Yield code passed through ``sh|bash|... -c`` for recursive scanning."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None or Path(segment[index]).name not in _SHELL_EXECUTABLES:
            continue
        arguments = segment[index + 1 :]
        for arg_index, argument in enumerate(arguments[:-1]):
            if argument in {"-c", "--command"}:
                yield arguments[arg_index + 1]
                break


def _resolve_script_directory(script_path: str) -> Optional[str]:
    """Return the directory *script_path* resolves to, handling relative names."""
    try:
        path = _resolve_script_path(script_path)
        if path.is_absolute():
            return str(path.parent)
    except Exception:
        pass
    return None


def _read_referenced_script(path: Path) -> tuple[Optional[str], bool]:
    """Return ``(text, unsafe)`` using bounded, regular-file-only reads."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        # ValueError, not only OSError: a path carrying a NUL byte raises
        # `ValueError: embedded null character` rather than an OSError, and an
        # uncaught one reaches the operator AS THE BLOCK REASON, because
        # GatewayLifecycleBlocked subclasses ValueError. Reachable from
        # ordinary input -- a referenced binary under the 1MiB cap is read,
        # decoded with errors="replace", and recursed into as shell text.
        return None, False
    try:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            # A DIRECTORY IS NOT A SCRIPT THAT COULD NOT BE READ. It is a token
            # that was never a script reference, and it cannot be the execution
            # vector this guard exists to close: `bash somedir` fails, so no
            # command shape hides behind one.
            #
            # It used to share the conservative answer below, which made any
            # job unschedulable if its script mentioned a directory path at the
            # start of a line -- `_iter_referenced_shell_scripts` reads a
            # leading token containing "/" as a command. Reported against a
            # script whose module docstring documented a privacy boundary
            # ("Not the vault -- <path> plaintext-mirrors to Drive nightly").
            # The operator was told the job "contains a gateway lifecycle
            # command or persistent launchctl submit operation"; it contained
            # neither, and `contains_gateway_lifecycle_command` on the same
            # text returns False.
            #
            # That is the failure mode this module warns about in its own
            # docstring, from the other side: the check was answering "could I
            # read every path this file mentions?" and reporting the answer to
            # "does this file contain a lifecycle command?".
            return None, False
        if not stat.S_ISREG(metadata.st_mode):
            # FIFOs, sockets and devices keep failing closed, and the asymmetry
            # with the directory case above is the point: these CAN be read,
            # and what they yield to this scan need not be what they yield when
            # the job runs.
            return None, True
        if metadata.st_size > _MAX_REFERENCED_SCRIPT_BYTES:
            return None, True
        data = os.read(descriptor, _MAX_REFERENCED_SCRIPT_BYTES + 1)
    except OSError:
        return None, False
    finally:
        os.close(descriptor)
    if len(data) > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    return data.decode("utf-8", errors="replace"), False


def _contains_unsafe_gateway_action(
    command: str,
    *,
    cwd: Optional[str],
    depth: int,
    visited: set[Path],
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    if contains_gateway_lifecycle_command(command) or contains_launchctl_submit_command(
        command
    ):
        return True
    if depth >= _MAX_REFERENCED_SCRIPT_DEPTH:
        return True

    for payload in _iter_shell_command_payloads(command):
        if _contains_unsafe_gateway_action(
            payload,
            cwd=cwd,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True

    for script_path in _iter_referenced_shell_scripts(command, cwd=cwd):
        try:
            resolved = script_path.resolve(strict=False)
        except (OSError, ValueError):
            # Same NUL-byte case as the open above; `lstat` raises ValueError.
            # An unresolvable path is not a lifecycle command, it is a path
            # this scan cannot follow.
            resolved = script_path
        if resolved in visited:
            continue
        visited.add(resolved)
        script_text, unsafe = _read_referenced_script(script_path)
        if unsafe:
            return True
        if script_text is None and read_remote_script is not None:
            # Local path missing; try the remote backend if one is available.
            script_text = read_remote_script(str(script_path))
        if not script_text:
            continue
        # Relative references inside a script resolve against that script's
        # directory, not the original command's cwd.
        script_dir = _resolve_script_directory(str(resolved)) or cwd
        if script_text and _contains_unsafe_gateway_action(
            scannable_text(script_text, resolved),
            cwd=script_dir,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True
    return False


def contains_gateway_lifecycle_command_or_referenced_script(
    command: str,
    *,
    cwd: Optional[str] = None,
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    """Detect lifecycle/submit commands, including bounded nested scripts."""
    return _contains_unsafe_gateway_action(
        command,
        cwd=cwd,
        depth=0,
        visited=set(),
        read_remote_script=read_remote_script,
    )




def _resolve_script_path(script_path: str) -> Path:
    """Resolve a cron ``script`` value the same way the scheduler does.

    The scheduler (``cron.scheduler``) resolves a bare/relative script path
    under ``<HERMES_HOME>/scripts/`` and only accepts absolute paths as-is.
    We MUST mirror that here so the guard scans the file that will actually
    run — otherwise a job whose script lives at the scheduler's real location
    (``~/.hermes/scripts/restart.sh``) but is passed as the bare name
    ``restart.sh`` would read as a nonexistent relative path and silently
    scan prompt-only content, letting the command through.
    """
    from hermes_constants import get_hermes_home

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        return raw
    return get_hermes_home() / "scripts" / raw


def _read_script_for_scanning(script_path: str) -> str:
    """Read a cron script with the bounded terminal-script scanner.

    Non-regular or oversized inputs fail closed by returning a lifecycle-shaped
    sentinel, while missing/unreadable paths remain empty so ordinary scheduler
    path validation can report them.
    """
    script_text, unsafe = _read_referenced_script(_resolve_script_path(script_path))
    if unsafe:
        return "hermes gateway restart"
    return script_text or ""


def check_gateway_lifecycle(
    prompt: Optional[str],
    script: Optional[str] = None,
) -> None:
    """Raise ``GatewayLifecycleBlocked`` if *prompt* or *script* contains a
    gateway-lifecycle command pattern.

    ``prompt`` is scanned directly.  ``script``, when supplied, is read from
    disk and concatenated for the scan.  Both are considered together so a
    job cannot slip through by splitting the command across the prompt and
    the script.

    Callers should let the exception propagate when they want the create to
    fail with a ``ValueError``-shaped error (the agent's ``cronjob`` tool
    surfaces this as a tool error; the CLI prints it in red and exits 1).
    """
    combined = prompt or ""
    if script:
        script_text = _read_script_for_scanning(script)
        if script_text:
            # Narrowed by interpreter BEFORE joining: a prompt is prose fed to
            # an LLM and a script is a program, so concatenating first would
            # scan the program as prose.
            combined = (
                f"{combined}\n"
                f"{scannable_text(script_text, _resolve_script_path(script))}"
            )

    script_dir = _resolve_script_directory(script) if script else None
    if contains_gateway_lifecycle_command_or_referenced_script(
        combined,
        cwd=script_dir,
    ):
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command or persistent "
            "launchctl submit operation. This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision "
            "(#30719). Run `hermes gateway restart` from a shell outside "
            "the running gateway instead."
        )
