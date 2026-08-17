"""Helpers for matching and downloading printer-side video files."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from backend.app.core.config import settings
from backend.app.services.bambu_ftp import download_file_async

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv")
MAX_PRINTER_ZIP_BYTES = 10 * 1024**3
PRINTER_ZIP_FREE_SPACE_RESERVE = 256 * 1024**2
_STALE_BUNDLE_SECONDS = 24 * 60 * 60
_BUNDLE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")


class PrinterFilesZipTooLargeError(ValueError):
    """The selected printer files exceed the bounded ZIP staging limit."""


class PrinterFilesZipInsufficientSpaceError(OSError):
    """The app data volume cannot safely stage the selected files."""


@dataclass(frozen=True)
class PrinterFilesZipResult:
    """Result of staging one printer ZIP."""

    path: Path
    requested: int
    successful: int
    failed_paths: tuple[str, ...]
    total_bytes: int


def _naive_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def match_ipcam_chunks(
    files: list[dict],
    started_at: datetime | None,
    completed_at: datetime | None,
    *,
    now: datetime | None = None,
) -> list[dict]:
    """Return `/ipcam` chunks whose completion time overlaps a print.

    Bambu's `ipcam-record.*.mp4` files are fixed-size chunks. On the tested X1C
    and H2D firmware, their FTP mtime is the chunk completion time in the same
    UTC-naive basis used by archive timestamps. Some firmware reports FTP LIST
    mtimes in printer-local time instead; LIST carries no timezone with which
    to correct those values reliably. A ten-minute tail includes the final
    chunk, whose mtime lands after the print-complete event.
    """

    start = _naive_utc(started_at)
    if start is None:
        return []
    end = _naive_utc(completed_at) or _naive_utc(now) or datetime.now(timezone.utc).replace(tzinfo=None)
    lower = start - timedelta(minutes=1)
    upper = max(start, end) + timedelta(minutes=10)

    matches: list[dict] = []
    for file in files:
        name = str(file.get("name") or "")
        mtime = file.get("mtime")
        if file.get("is_directory") or not name.lower().startswith("ipcam-record."):
            continue
        if not name.lower().endswith(VIDEO_SUFFIXES) or not isinstance(mtime, datetime):
            continue
        timestamp = _naive_utc(mtime)
        if timestamp is not None and lower <= timestamp <= upper:
            matches.append(file)

    matches.sort(key=lambda item: _naive_utc(item.get("mtime")) or datetime.min)
    return matches


def _zip_arcname(remote_path: str, used: set[str]) -> str:
    """Return a safe, unique relative archive name for a printer path."""

    parts = [part for part in PurePosixPath(remote_path).parts if part not in ("/", "", ".", "..")]
    candidate = "/".join(parts) or "printer-file"
    stem = candidate
    suffix = ""
    if "." in PurePosixPath(candidate).name:
        suffix = "".join(PurePosixPath(candidate).suffixes)
        stem = candidate[: -len(suffix)] if suffix else candidate
    counter = 2
    while candidate in used:
        candidate = f"{stem}-{counter}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _printer_zip_root() -> Path:
    """Return the dedicated staging root on the persistent app data volume."""

    root = settings.archive_dir / "temp" / "printer-file-downloads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prune_stale_bundles(root: Path) -> None:
    """Remove abandoned bundles after token expiry, without touching archives."""

    cutoff = time.time() - _STALE_BUNDLE_SECONDS
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def printer_files_zip_path(printer_id: int, token: str) -> Path | None:
    """Resolve the staged ZIP for a resource-bound browser token."""

    bundle_key = f"{printer_id}-{token}"
    if not _BUNDLE_KEY_RE.fullmatch(bundle_key):
        return None
    return _printer_zip_root() / bundle_key / "printer-files.zip"


def bind_printer_files_zip_to_token(
    result: PrinterFilesZipResult,
    printer_id: int,
    token: str,
) -> PrinterFilesZipResult:
    """Move a prepared bundle to the path derived from its persisted token."""

    target = printer_files_zip_path(printer_id, token)
    if target is None:
        raise ValueError("Invalid printer ZIP token")
    result.path.parent.rename(target.parent)
    return replace(result, path=target)


def _check_initial_space(root: Path, sizes: dict[str, int]) -> None:
    expected_total = sum(sizes.values())
    if expected_total > MAX_PRINTER_ZIP_BYTES:
        raise PrinterFilesZipTooLargeError(
            f"Selected files total {expected_total} bytes; the limit is {MAX_PRINTER_ZIP_BYTES} bytes"
        )

    largest_file = max(sizes.values(), default=0)
    # In the worst case the ZIP is as large as the inputs while the largest
    # source is still staged beside it. Keep a reserve for the database/logs.
    required = expected_total + largest_file + PRINTER_ZIP_FREE_SPACE_RESERVE
    free = shutil.disk_usage(root).free
    if free < required:
        raise PrinterFilesZipInsufficientSpaceError(
            f"The app data volume needs {required} bytes free to stage this selection; {free} bytes are available"
        )


async def build_printer_files_zip(
    printer,
    paths: list[str],
    sizes: dict[str, int],
    *,
    bundle_key: str | None = None,
) -> PrinterFilesZipResult:
    """Download printer files one at a time into a disk-backed ZIP.

    The previous implementation held every source file and the final ZIP in
    memory. Continuous `/ipcam` chunks are commonly ~250 MB each, so selecting
    only a few could exhaust both server and browser memory.
    """

    root = _printer_zip_root()
    _prune_stale_bundles(root)
    _check_initial_space(root, sizes)

    if bundle_key is None:
        bundle_dir = Path(tempfile.mkdtemp(prefix="bundle-", dir=root))
    else:
        if not _BUNDLE_KEY_RE.fullmatch(bundle_key):
            raise ValueError("Invalid printer ZIP bundle key")
        bundle_dir = root / bundle_key
        bundle_dir.mkdir(mode=0o700)
    zip_path = bundle_dir / "printer-files.zip"
    successful = 0
    total_bytes = 0
    failed_paths: list[str] = []
    used_names: set[str] = set()

    try:
        with zipfile.ZipFile(zip_path, "w", allowZip64=True) as archive:
            for index, remote_path in enumerate(paths):
                if not isinstance(remote_path, str) or not remote_path.startswith("/") or "\x00" in remote_path:
                    logger.warning("Skipping invalid printer file path: %r", remote_path)
                    failed_paths.append(remote_path)
                    continue
                staged_path = bundle_dir / f"download-{index}"
                try:
                    downloaded = await download_file_async(
                        printer.ip_address,
                        printer.access_code,
                        remote_path,
                        staged_path,
                        timeout=600,
                        socket_timeout=60,
                        printer_model=printer.model,
                    )
                    if not downloaded:
                        failed_paths.append(remote_path)
                        continue
                    file_size = staged_path.stat().st_size
                    if total_bytes + file_size > MAX_PRINTER_ZIP_BYTES:
                        raise PrinterFilesZipTooLargeError(
                            f"Downloaded files exceed the {MAX_PRINTER_ZIP_BYTES}-byte limit"
                        )
                    if shutil.disk_usage(root).free < file_size + PRINTER_ZIP_FREE_SPACE_RESERVE:
                        raise PrinterFilesZipInsufficientSpaceError(
                            "The app data volume ran out of safe staging space while building the ZIP"
                        )
                    compression = (
                        zipfile.ZIP_STORED if remote_path.lower().endswith(VIDEO_SUFFIXES) else zipfile.ZIP_DEFLATED
                    )
                    archive.write(
                        staged_path,
                        _zip_arcname(remote_path, used_names),
                        compress_type=compression,
                    )
                    successful += 1
                    total_bytes += file_size
                except (PrinterFilesZipTooLargeError, PrinterFilesZipInsufficientSpaceError):
                    raise
                except Exception as exc:
                    logger.warning("Failed to add %s to printer ZIP: %s", remote_path, exc)
                    failed_paths.append(remote_path)
                finally:
                    staged_path.unlink(missing_ok=True)
    except Exception:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise

    if successful == 0:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise FileNotFoundError("No files could be downloaded")
    return PrinterFilesZipResult(
        path=zip_path,
        requested=len(paths),
        successful=successful,
        failed_paths=tuple(failed_paths),
        total_bytes=total_bytes,
    )


def remove_printer_files_zip(zip_path: Path) -> None:
    """Remove a completed download bundle after FileResponse finishes."""

    shutil.rmtree(zip_path.parent, ignore_errors=True)
