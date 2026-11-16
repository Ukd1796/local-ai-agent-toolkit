"""Bounded filesystem tools scoped to the configured workspace directory."""

from __future__ import annotations

import ast
import fnmatch
import os
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
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")
        if start_line is None and end_line is None:
            return {"path": Path(relative).as_posix(), "content": content}

        for label, value in (("start_line", start_line), ("end_line", end_line)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{label} must be a positive 1-based line number")
        start = start_line if start_line is not None else 1
        lines = content.splitlines(keepends=True)
        if start > len(lines):
            raise ValueError(f"start_line {start} is past the end of the file ({len(lines)} lines)")
        end = end_line if end_line is not None else len(lines)
        if end < start:
            raise ValueError("end_line must be greater than or equal to start_line")
        end = min(end, len(lines))
        return {
            "path": Path(relative).as_posix(),
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "content": "".join(lines[start - 1:end]),
        }

    def find_function_or_method(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Locate Python function/method definitions by exact name in one specified file."""
        relative = str(arguments.get("path", ""))
        name = str(arguments.get("name", "")).strip()
        if not name or not name.isidentifier():
            raise ValueError("name must be a valid Python function or method name")
        path = self._resolve(relative)
        if path.suffix != ".py":
            raise ValueError("find_function_or_method currently supports Python (.py) files")
        source = self._text_file(path)
        try:
            tree = ast.parse(source, filename=Path(relative).as_posix())
        except SyntaxError as exc:
            raise ValueError(f"Could not parse Python file: {exc}") from exc

        matches: list[dict[str, Any]] = []

        def visit(node: ast.AST, scope: tuple[str, ...] = (), in_class: bool = False) -> None:
            for child in ast.iter_child_nodes(node):
                child_scope = scope
                child_in_class = in_class
                if isinstance(child, ast.ClassDef):
                    child_scope = (*scope, child.name)
                    child_in_class = True
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == name:
                        start_line = min(
                            [child.lineno, *(decorator.lineno for decorator in child.decorator_list)]
                        )
                        matches.append({
                            "kind": "method" if in_class else "function",
                            "name": child.name,
                            "scope": ".".join(scope) or None,
                            "line": start_line,
                            "end_line": child.end_lineno,
                            "signature": source.splitlines()[child.lineno - 1].strip(),
                        })
                    child_scope = (*scope, child.name)
                    child_in_class = False
                visit(child, child_scope, child_in_class)

        visit(tree)
        return {"path": Path(relative).as_posix(), "name": name, "matches": matches}

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
