"""Bounded filesystem tools scoped to the configured workspace directory."""

from __future__ import annotations

import fnmatch
import os
import selectors
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build",
}
MAX_FILE_BYTES = 256_000
MAX_LIST_RESULTS = 200
MAX_SEARCH_RESULTS = 50
MAX_FILES_SCANNED = 5_000
MAX_COMMAND_OUTPUT_CHARS = 12_000
ALLOWED_COMMANDS = {"rg", "grep", "find", "ls", "cat", "head", "tail", "wc", "pwd"}


class WorkspaceFiles:
    """Read, search, create, and edit files without leaving the workspace root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace root must be a directory")

    @staticmethod
    def _is_private(path: Path) -> bool:
        name = path.name.lower()
        return name == ".env" or (name.startswith(".env.") and name != ".env.example")

    def _resolve(self, relative_path: str, *, allow_root: bool = False) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Use a relative path inside the configured workspace directory")
        if not relative_path.strip() and not allow_root:
            raise ValueError("A workspace-relative path is required")
        cursor = self.root
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Symbolic links are not available to workspace tools")
        resolved = (self.root / candidate).resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Path resolves outside the configured workspace directory")
        if any(part in IGNORED_DIRS for part in candidate.parts):
            raise ValueError("This generated or dependency directory is excluded from filesystem tools")
        if self._is_private(candidate):
            raise ValueError("Secret environment files are not available to workspace tools")
        return resolved

    @staticmethod
    def _text_file(path: Path) -> str:
        if path.is_symlink():
            raise ValueError("Symbolic links are not available to workspace tools")
        if not path.is_file():
            raise ValueError("Path is not a regular file")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"File is too large ({size} bytes; limit is {MAX_FILE_BYTES})")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("File is not UTF-8 text") from exc

    def list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("pattern", "**/*")).strip() or "**/*"
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("pattern must stay inside the workspace directory")
        limit = int(arguments.get("limit", 100))
        if not 1 <= limit <= MAX_LIST_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_LIST_RESULTS}")

        files, directories, scanned = [], [], 0
        for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
            base = Path(directory)
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in IGNORED_DIRS and not (base / name).is_symlink()
            )
            for dirname in dirnames:
                relative_dir = (base / dirname).relative_to(self.root).as_posix()
                if (fnmatch.fnmatch(relative_dir, pattern) or pattern == "**/*") and len(directories) < limit:
                    directories.append(relative_dir)
            for filename in sorted(filenames):
                path = base / filename
                if path.is_symlink() or self._is_private(path):
                    continue
                scanned += 1
                if scanned > MAX_FILES_SCANNED:
                    break
                relative = path.relative_to(self.root).as_posix()
                if fnmatch.fnmatch(relative, pattern) or (pattern == "**/*" and fnmatch.fnmatch(filename, "*")):
                    files.append(relative)
                    if len(files) >= limit:
                        break
            if len(files) >= limit or scanned > MAX_FILES_SCANNED:
                break
        return {
            "root": str(self.root),
            "pattern": pattern,
            "directories": directories,
            "files": files,
            "truncated": len(files) >= limit or len(directories) >= limit or scanned > MAX_FILES_SCANNED,
        }

    def read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path", ""))
        path = self._resolve(relative)
        content = self._text_file(path)
        return {"path": Path(relative).as_posix(), "content": content}

    def search_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", ""))
        if not query:
            raise ValueError("query cannot be empty")
        pattern = str(arguments.get("pattern", "**/*")).strip() or "**/*"
        limit = int(arguments.get("limit", 20))
        if not 1 <= limit <= MAX_SEARCH_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_SEARCH_RESULTS}")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("pattern must stay inside the workspace directory")

        matches, scanned = [], 0
        for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
            base = Path(directory)
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in IGNORED_DIRS and not (base / name).is_symlink()
            )
            for filename in sorted(filenames):
                path = base / filename
                if path.is_symlink() or self._is_private(path):
                    continue
                scanned += 1
                if scanned > MAX_FILES_SCANNED:
                    break
                relative = path.relative_to(self.root).as_posix()
                if not (fnmatch.fnmatch(relative, pattern) or (pattern == "**/*" and fnmatch.fnmatch(filename, "*"))):
                    continue
                try:
                    content = self._text_file(path)
                except (ValueError, OSError):
                    continue
                for line_number, line in enumerate(content.splitlines(), 1):
                    if query.casefold() in line.casefold():
                        matches.append({"path": relative, "line": line_number, "text": line[:500]})
                        if len(matches) >= limit:
                            break
                if len(matches) >= limit:
                    break
            if len(matches) >= limit or scanned > MAX_FILES_SCANNED:
                break
        return {"query": query, "matches": matches, "truncated": len(matches) >= limit or scanned > MAX_FILES_SCANNED}

    def write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path", ""))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be text")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(f"content exceeds the {MAX_FILE_BYTES}-byte limit")
        path = self._resolve(relative)
        existed = path.exists()
        if path.exists() and not path.is_file():
            raise ValueError("Target must be a regular file")
        if path.exists() and path.is_symlink():
            raise ValueError("Symbolic links are not available to workspace tools")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": Path(relative).as_posix(),
            "bytes_written": len(content.encode("utf-8")),
            "action": "updated" if existed else "created",
        }

    def edit_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path", ""))
        old_text, new_text = arguments.get("old_text"), arguments.get("new_text")
        if not isinstance(old_text, str) or not old_text:
            raise ValueError("old_text must be a non-empty exact text match")
        if not isinstance(new_text, str):
            raise ValueError("new_text must be text")
        path = self._resolve(relative)
        content = self._text_file(path)
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise ValueError(f"Expected exactly one matching block, found {occurrences}; no changes made")
        updated = content.replace(old_text, new_text, 1)
        if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(f"edited file exceeds the {MAX_FILE_BYTES}-byte limit")
        path.write_text(updated, encoding="utf-8")
        return {"path": Path(relative).as_posix(), "replacements": 1, "bytes_written": len(updated.encode("utf-8"))}

    def run_bash(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one allowlisted, read-only discovery command with cwd at the workspace root."""
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip() or len(command) > 2_000:
            raise ValueError("command must be a non-empty string up to 2,000 characters")
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"Could not parse command: {exc}") from exc
        if not parts:
            raise ValueError("command cannot be empty")
        executable = Path(parts[0]).name
        if executable not in ALLOWED_COMMANDS:
            raise ValueError(f"'{executable}' is not allowed; use one of: {', '.join(sorted(ALLOWED_COMMANDS))}")
        args = parts[1:]
        if executable == "find" and any(
            arg in {"-exec", "-execdir", "-delete", "-ok", "-okdir"}
            or arg.startswith(("-fprint", "-fprintf", "-fls"))
            for arg in args
        ):
            raise ValueError("find execution, deletion, and output-file actions are not allowed")
        if executable == "grep" and any(
            arg in {"--recursive", "-r", "-R"}
            or (arg.startswith("-") and not arg.startswith("--") and any(flag in arg[1:] for flag in "rR"))
            for arg in args
        ):
            raise ValueError("Recursive grep is disabled; use rg, which respects workspace ignore rules")
        if executable == "rg" and any(
            arg in {"--pre", "--hidden", "-u", "-uu", "-uuu"}
            or arg.startswith(("--pre=", "--no-ignore")) for arg in args
        ):
            raise ValueError("rg options that expose ignored files or execute preprocessors are not allowed")
        for arg in args:
            # Commands are launched without a shell, but also reject path escapes and absolute paths.
            value = arg.split("=", 1)[-1] if arg.startswith("-") and "=" in arg else arg
            path_value = Path(value)
            if path_value.is_absolute() or ".." in path_value.parts or "../" in value or value.startswith("~/"):
                raise ValueError("command arguments must not access paths outside the workspace")
            if self._is_private(path_value):
                raise ValueError("Commands cannot read secret environment files")
            possible_path = self.root / path_value
            cursor = self.root
            for part in path_value.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError("Commands cannot follow symbolic links")
            resolved = possible_path.resolve(strict=False)
            if resolved != self.root and self.root not in resolved.parents:
                raise ValueError("command arguments must not access paths outside the workspace")

        binary = shutil.which(executable)
        if not binary:
            raise RuntimeError(f"Required command is not installed: {executable}")
        process = subprocess.Popen(
            [binary, *args],
            cwd=self.root,
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        output = bytearray()
        truncated = timed_out = False
        deadline = time.monotonic() + 15
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                events = selector.select(remaining)
                if not events:
                    timed_out = True
                    break
                chunk = os.read(process.stdout.fileno(), min(4096, MAX_COMMAND_OUTPUT_CHARS + 1 - len(output)))
                if not chunk:
                    selector.unregister(process.stdout)
                    break
                output.extend(chunk)
                if len(output) > MAX_COMMAND_OUTPUT_CHARS:
                    truncated = True
                    break
        if timed_out or truncated:
            process.terminate()
        try:
            return_code = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        process.stdout.close()
        decoded_output = bytes(output[:MAX_COMMAND_OUTPUT_CHARS]).decode("utf-8", errors="replace")
        return {
            "command": command,
            "return_code": None if timed_out else return_code,
            "timed_out": timed_out,
            "output": decoded_output,
            "output_truncated": truncated,
        }
