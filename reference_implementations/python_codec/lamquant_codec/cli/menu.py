"""
OpenHuman LamQuant — Interactive Menu

This module contains the menu rendering, input handling, history,
and terminal helpers. Zero heavy imports. Starts in <50ms.

The actual command implementations live in their own modules and
are imported lazily only when the user selects them.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from lamquant_codec._paths import REPO_ROOT as ROOT


# ────────────────────────────────────────────────────────────────────
# Terminal (no dependencies)
# ────────────────────────────────────────────────────────────────────

def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

def _supports_unicode():
    try:
        "┌─┐│└─┘✓✗⠋".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False

_C = _supports_color()
_U = _supports_unicode()

DIM   = "\033[90m" if _C else ""
CYN   = "\033[36m" if _C else ""
GRN   = "\033[32m" if _C else ""
RED   = "\033[31m" if _C else ""
YEL   = "\033[33m" if _C else ""
BLD   = "\033[1m"  if _C else ""
RST   = "\033[0m"  if _C else ""

H  = "─" if _U else "-"
DOT = "·" if _U else "-"
OK = "✓" if _U else "OK"
NO = "✗" if _U else "X"


def clear(full=False):
    if sys.stdout.isatty():
        if full:
            # Clear screen + scrollback buffer, then home cursor
            sys.stdout.write("\033[3J\033[H\033[2J\033[H")
        else:
            sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


# ────────────────────────────────────────────────────────────────────
# Prompt (optional prompt_toolkit)
# ────────────────────────────────────────────────────────────────────

try:
    from prompt_toolkit import prompt as _pt_prompt
    from prompt_toolkit.completion import PathCompleter, WordCompleter
    _HAS_PT = True
    _path_completer = PathCompleter(expanduser=True)
except ImportError:
    _HAS_PT = False

_autocomplete = True   # on by default (prompt_toolkit tab-completion)
_instant_nav = False   # single keypress, no Enter needed (default off)

def set_autocomplete(enabled: bool):
    global _autocomplete
    _autocomplete = enabled

def set_instant_nav(enabled: bool):
    global _instant_nav
    _instant_nav = enabled

def instant_input(msg="  > "):
    """Read a single keypress if instant_nav is on, else normal input()."""
    if not _instant_nav or not sys.stdout.isatty() or sys.platform == 'win32':
        return input(msg).strip().lower()
    import tty, termios
    sys.stdout.write(msg)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        sys.stdout.write(ch + "\n")
        sys.stdout.flush()
        return ch.lower()
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def prompt(msg="  > ", completer=None):
    try:
        if completer and _HAS_PT and _autocomplete and sys.stdout.isatty():
            return _pt_prompt(msg, completer=completer).strip()
        return input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt

def prompt_path(msg="  > "):
    if _HAS_PT and _autocomplete:
        return prompt(msg, completer=_path_completer)
    return input(msg).strip()

def prompt_menu(msg="  > ", options=None):
    if _HAS_PT and _autocomplete and options:
        return prompt(msg, completer=WordCompleter(options, ignore_case=True))
    return prompt(msg)


# ────────────────────────────────────────────────────────────────────
# Version (lazy, no heavy imports)
# ────────────────────────────────────────────────────────────────────

def version():
    try:
        from lamquant_codec import __version__
        return __version__
    except Exception:
        return "0.0.0"

def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=2, cwd=ROOT)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

def gen_tag():
    v = version()
    parts = v.split(".")
    return f"Gen {parts[0]}.{parts[1]}" if len(parts) >= 2 else f"v{v}"

def cli_version():
    try:
        from lamquant_codec import __cli_version__
        return __cli_version__
    except Exception:
        return "1.0.0"


# ────────────────────────────────────────────────────────────────────
# History (persistent, locked, atomic) — shared with Rust TUI + Tauri GUI
#
# On-disk schema is canonical at ``specs/history-schema.json``. The
# resolver mirrors the Rust ``crates/lamquant-history/src/lib.rs``
# precedence so all three front-ends read/write the same file.
# ────────────────────────────────────────────────────────────────────

def _history_dir():
    """Resolve the per-OS history directory.

    Precedence (mirrors `lamquant_history::history_path` in Rust):
      1. ``LAMQUANT_HISTORY`` env (test override / advanced users).
      2. ``$XDG_CONFIG_HOME/lamquant/`` (Linux + explicit XDG).
      3. ``~/Library/Application Support/lamquant/`` (macOS).
      4. ``%APPDATA%\\lamquant\\`` (Windows).
      5. ``~/.config/lamquant/`` (Linux fallback).
    """
    override = os.environ.get("LAMQUANT_HISTORY")
    if override:
        return Path(override).parent
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "lamquant"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "lamquant"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "lamquant"
    return Path.home() / ".config" / "lamquant"


def _history_path():
    """Full path to ``history.json`` (honours LAMQUANT_HISTORY override)."""
    override = os.environ.get("LAMQUANT_HISTORY")
    if override:
        return Path(override)
    return _history_dir() / "history.json"


def _empty_history():
    return {
        "schema_version": "1.0",
        "recent_operations": [],
        "recent_paths": {"inputs": [], "outputs": []},
        "interrupted": False,
        "last_op": None,
        "last_input": None,
        "last_output": None,
    }


def _load_or_migrate(path: Path):
    """Read history JSON; migrate the legacy flat shape if encountered."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return _empty_history()
    # Spec format already.
    if "recent_paths" in data:
        return data
    # Legacy flat shape: pre-Phase-2 had `recent_inputs`/`recent_outputs`
    # at top level. Migrate quietly so users don't lose history.
    return {
        "schema_version": "1.0",
        "recent_operations": data.get("recent_operations", []),
        "recent_paths": {
            "inputs":  data.get("recent_inputs",  []),
            "outputs": data.get("recent_outputs", []),
        },
        "interrupted":  data.get("interrupted",  False),
        "last_op":      data.get("last_op",      None),
        "last_input":   data.get("last_input",   None),
        "last_output":  data.get("last_output",  None),
    }


def _atomic_write_locked(path: Path, body: str):
    """Atomic write with an OS-level advisory lock on a sibling lock file.

    Mirrors the Rust ``lamquant_history`` writer so the TUI, GUI, and this
    Python entrypoint cooperate when they happen to write simultaneously.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".json.lock")
    tmp = path.with_suffix(".json.tmp")
    lock_fh = None
    try:
        # Acquire the lock. Best-effort on non-POSIX platforms.
        lock_fh = open(lock_path, "a+")
        try:
            import fcntl  # POSIX
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # Windows: opening with default share mode is best we can do.
        tmp.write_text(body)
        tmp.replace(path)
    finally:
        if lock_fh is not None:
            try:
                import fcntl
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fh.close()


def load_history():
    """Read the spec-format history JSON. Returns the canonical shape."""
    path = _history_path()
    return _load_or_migrate(path)


def update_history(action: str, target: str, result: str):
    """Append an op to the rolling 50-entry log, atomically + locked."""
    path = _history_path()
    try:
        history = _load_or_migrate(path)
        history["recent_operations"].insert(0, {
            "action": action, "target": target,
            "when": datetime.now(timezone.utc).isoformat(),
            "result": result,
        })
        history["recent_operations"] = history["recent_operations"][:50]
        _atomic_write_locked(path, json.dumps(history, indent=2))
    except Exception as e:
        print(f"warning: could not update history: {e}", file=sys.stderr)


def add_recent_path(kind: str, path: str):
    """Push ``path`` to the front of the appropriate recent list."""
    if kind not in ("inputs", "outputs", "input", "output"):
        return
    if kind in ("input", "output"):
        kind = kind + "s"
    target_path = _history_path()
    try:
        history = _load_or_migrate(target_path)
        paths = history.setdefault("recent_paths", {}).setdefault(kind, [])
        if path in paths:
            paths.remove(path)
        paths.insert(0, path)
        history["recent_paths"][kind] = paths[:20]
        _atomic_write_locked(target_path, json.dumps(history, indent=2))
    except Exception as e:
        print(f"warning: could not save recent path: {e}", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────
# State detection
# ────────────────────────────────────────────────────────────────────

def find_interrupted_runs():
    candidates = []
    history = load_history()
    for path in history.get("recent_paths", {}).get("outputs", [])[:10]:
        sf = Path(path) / ".lamquant_state.json"
        if sf.exists():
            try:
                state = json.loads(sf.read_text())
                stats = state.get("statistics_so_far", {})
                if stats.get("files_remaining", 0) > 0:
                    candidates.append((path, state))
            except Exception:
                pass
    if Path(".lamquant_state.json").exists():
        try:
            state = json.loads(Path(".lamquant_state.json").read_text())
            candidates.append((str(Path.cwd()), state))
        except Exception:
            pass
    return candidates

def config_status():
    try:
        from lamquant_codec.cli.config import _find_config_file
        cf = _find_config_file()
        return str(cf) if cf else None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# Error recovery
# ────────────────────────────────────────────────────────────────────

def save_crash_report(exc):
    crash_dir = Path(tempfile.gettempdir())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = crash_dir / f"lamquant_crash_{ts}.txt"
    try:
        path.write_text(traceback.format_exc())
    except Exception:
        pass
    return str(path)

def run_safely(fn, args=None):
    try:
        return fn(args or [])
    except KeyboardInterrupt:
        print(f"\n  {YEL}Cancelled.{RST}")
        return None
    except SystemExit:
        raise  # let x/exit propagate — don't swallow it
    except Exception as e:
        crash = save_crash_report(e)
        print(f"\n  {RED}Error: {e}{RST}")
        print(f"  Crash report: {DIM}{crash}{RST}")
        return None


# ────────────────────────────────────────────────────────────────────
# Input matching
# ────────────────────────────────────────────────────────────────────

def match_input(text, options_map):
    text = text.strip().lower()
    if not text:
        return None
    if text == "x":
        return "__exit__"
    if text in ("q", "quit"):
        return "__quit__"
    if text in ("b", "back"):
        return "__back__"
    if text in ("?", "h", "help"):
        return "__help__"
    if text.startswith("!"):
        # Shell escape disabled for safety (FDA clinical tool).
        # os.system(text[1:]) was command injection from user input.
        print(f"  {DIM}Shell escape disabled.{RST}")
        return "__shell__"
    if text in options_map:
        return text
    for key, label in options_map.items():
        if label.lower().startswith(text):
            return key
    return None


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd or str(ROOT)).returncode
