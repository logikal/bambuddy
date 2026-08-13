"""Helpers for matching and downloading printer-side video files."""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from backend.app.services.bambu_ftp import download_file_async

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv")


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

    Bambu's `ipcam-record.*.mp4` files are fixed-size chunks. Their FTP mtime
    is the chunk completion time in the same UTC-naive basis used by archive
    timestamps. A ten-minute tail includes the final chunk, whose mtime lands
    after the print-complete event.
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


async def build_printer_files_zip(printer, paths: list[str]) -> tuple[Path, int]:
    """Download printer files one at a time into a disk-backed ZIP.

    The previous implementation held every source file and the final ZIP in
    memory. Continuous `/ipcam` chunks are commonly ~250 MB each, so selecting
    only a few could exhaust both server and browser memory.
    """

    bundle_dir = Path(tempfile.mkdtemp(prefix="bambuddy-printer-files-"))
    zip_path = bundle_dir / "printer-files.zip"
    successful = 0
    used_names: set[str] = set()

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for index, remote_path in enumerate(paths):
                if not isinstance(remote_path, str) or not remote_path.startswith("/") or "\x00" in remote_path:
                    logger.warning("Skipping invalid printer file path: %r", remote_path)
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
                        continue
                    archive.write(staged_path, _zip_arcname(remote_path, used_names))
                    successful += 1
                except Exception as exc:
                    logger.warning("Failed to add %s to printer ZIP: %s", remote_path, exc)
                finally:
                    staged_path.unlink(missing_ok=True)
    except Exception:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise

    if successful == 0:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise FileNotFoundError("No files could be downloaded")
    return zip_path, successful


def remove_printer_files_zip(zip_path: Path) -> None:
    """Remove a completed download bundle after FileResponse finishes."""

    shutil.rmtree(zip_path.parent, ignore_errors=True)
