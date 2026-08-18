import asyncio
import os
import shutil
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.core.config import settings
from backend.app.services import printer_media
from backend.app.services.printer_media import (
    MAX_PRINTER_ZIP_BYTES,
    PrinterFilesZipInsufficientSpaceError,
    PrinterFilesZipTooLargeError,
    build_printer_files_zip,
    match_ipcam_chunks,
    prune_stale_printer_file_bundles,
    remove_printer_files_zip,
)


def test_match_ipcam_chunks_uses_archive_window_and_ignores_non_video_entries():
    files = [
        {"name": "index", "mtime": datetime(2026, 8, 12, 10, 5), "is_directory": False},
        {"name": "ipcam-record.before.mp4", "mtime": datetime(2026, 8, 12, 9, 50), "is_directory": False},
        {"name": "ipcam-record.first.mp4", "mtime": datetime(2026, 8, 12, 10, 4), "is_directory": False},
        {"name": "ipcam-record.last.mp4", "mtime": datetime(2026, 8, 12, 11, 8), "is_directory": False},
        {"name": "ipcam-record.after.mp4", "mtime": datetime(2026, 8, 12, 11, 11), "is_directory": False},
    ]

    matched = match_ipcam_chunks(
        files,
        datetime(2026, 8, 12, 10, 0),
        datetime(2026, 8, 12, 11, 0),
    )

    assert [file["name"] for file in matched] == ["ipcam-record.first.mp4", "ipcam-record.last.mp4"]


@pytest.mark.asyncio
async def test_build_printer_files_zip_stages_on_data_volume_and_compresses_by_type(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")

    payloads = {
        "/ipcam/chunk.mp4": (b"video-") * 512,
        "/cache/model.gcode": (b"G1 X1 Y1\n") * 512,
    }

    async def fake_download(_ip, _code, remote_path, local_path: Path, **_kwargs):
        local_path.write_bytes(payloads[remote_path])
        return True

    with patch(
        "backend.app.services.printer_media.download_file_async",
        new=AsyncMock(side_effect=fake_download),
    ):
        result = await build_printer_files_zip(
            printer,
            ["/ipcam/chunk.mp4", "/cache/model.gcode"],
            {path: len(payload) for path, payload in payloads.items()},
        )
    zip_path = result.path

    try:
        assert result.successful == 2
        assert zip_path.is_relative_to(settings.archive_dir / "temp" / "printer-file-downloads")
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.namelist() == ["ipcam/chunk.mp4", "cache/model.gcode"]
            assert archive.getinfo("ipcam/chunk.mp4").compress_type == zipfile.ZIP_STORED
            assert archive.getinfo("cache/model.gcode").compress_type == zipfile.ZIP_DEFLATED
        assert not list(zip_path.parent.glob("download-*"))
    finally:
        remove_printer_files_zip(zip_path)

    assert not zip_path.parent.exists()


@pytest.mark.asyncio
async def test_build_printer_files_zip_offloads_blocking_zip_and_filesystem_work(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")
    event_loop_thread = threading.get_ident()
    offloaded_threads: list[int] = []
    real_prune = printer_media._prune_stale_bundles
    real_space_check = printer_media._check_initial_space

    def tracking_prune(root):
        offloaded_threads.append(threading.get_ident())
        return real_prune(root)

    def tracking_space_check(root, sizes):
        offloaded_threads.append(threading.get_ident())
        return real_space_check(root, sizes)

    async def fake_download(_ip, _code, _remote_path, local_path: Path, **_kwargs):
        local_path.write_bytes(b"G1 X1 Y1\n" * 512)
        return True

    monkeypatch.setattr(printer_media, "_prune_stale_bundles", tracking_prune)
    monkeypatch.setattr(printer_media, "_check_initial_space", tracking_space_check)
    with patch("backend.app.services.printer_media.download_file_async", new=AsyncMock(side_effect=fake_download)):
        result = await build_printer_files_zip(printer, ["/model.gcode"], {"/model.gcode": 4608})

    try:
        assert offloaded_threads
        assert all(thread_id != event_loop_thread for thread_id in offloaded_threads)
    finally:
        remove_printer_files_zip(result.path)


@pytest.mark.asyncio
async def test_prune_stale_printer_file_bundles_removes_hour_old_abandoned_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")
    root = settings.archive_dir / "temp" / "printer-file-downloads"
    stale = root / "stale"
    fresh = root / "fresh"
    stale.mkdir(parents=True)
    fresh.mkdir()
    (stale / "printer-files.zip").write_bytes(b"stale")
    (fresh / "printer-files.zip").write_bytes(b"fresh")
    old = time.time() - 60 * 60 - 1
    os.utime(stale, (old, old))

    await prune_stale_printer_file_bundles()

    assert not stale.exists()
    assert fresh.exists()


@pytest.mark.asyncio
async def test_build_printer_files_zip_skips_relative_and_nul_paths(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")

    async def fake_download(_ip, _code, remote_path, local_path: Path, **_kwargs):
        local_path.write_bytes(b"valid")
        return True

    download = AsyncMock(side_effect=fake_download)
    with patch("backend.app.services.printer_media.download_file_async", new=download):
        result = await build_printer_files_zip(
            printer,
            ["relative.gcode", "/bad\x00.gcode", "/valid.gcode"],
            {"relative.gcode": 1, "/bad\x00.gcode": 1, "/valid.gcode": 5},
        )

    try:
        assert result.requested == 3
        assert result.successful == 1
        assert result.failed_paths == ("relative.gcode", "/bad\x00.gcode")
        download.assert_awaited_once()
        assert download.await_args.args[2] == "/valid.gcode"
    finally:
        remove_printer_files_zip(result.path)


@pytest.mark.asyncio
async def test_build_printer_files_zip_rejects_oversized_selection_before_download(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")
    download = AsyncMock()

    with (
        patch("backend.app.services.printer_media.download_file_async", new=download),
        pytest.raises(PrinterFilesZipTooLargeError),
    ):
        await build_printer_files_zip(
            printer,
            ["/huge.mp4"],
            {"/huge.mp4": MAX_PRINTER_ZIP_BYTES + 1},
        )

    download.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_printer_files_zip_rejects_insufficient_data_volume_space(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")
    monkeypatch.setattr("backend.app.services.printer_media.shutil.disk_usage", lambda _path: SimpleNamespace(free=1))

    with pytest.raises(PrinterFilesZipInsufficientSpaceError):
        await build_printer_files_zip(printer, ["/small.gcode"], {"/small.gcode": 5})


@pytest.mark.asyncio
async def test_build_printer_files_zip_rejects_a_short_transfer(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")

    async def fake_download(_ip, _code, _remote_path, local_path: Path, **_kwargs):
        local_path.write_bytes(b"short")
        return True

    with (
        patch("backend.app.services.printer_media.download_file_async", new=AsyncMock(side_effect=fake_download)),
        pytest.raises(FileNotFoundError),
    ):
        await build_printer_files_zip(printer, ["/video.mp4"], {"/video.mp4": 100})


def test_match_ipcam_chunks_caps_an_unfinished_archive_window():
    files = [
        {
            "name": "ipcam-record.next-week.mp4",
            "mtime": datetime(2026, 8, 20, 10, 0),
            "is_directory": False,
        }
    ]

    assert (
        match_ipcam_chunks(
            files,
            datetime(2026, 8, 12, 10, 0),
            None,
            now=datetime(2026, 8, 21, 10, 0),
        )
        == []
    )


@pytest.mark.asyncio
async def test_build_printer_files_zip_cleans_bundle_on_cancellation(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")

    async def cancel_download(*_args, **_kwargs):
        raise asyncio.CancelledError

    with (
        patch("backend.app.services.printer_media.download_file_async", new=AsyncMock(side_effect=cancel_download)),
        pytest.raises(asyncio.CancelledError),
    ):
        await build_printer_files_zip(printer, ["/video.mp4"], {"/video.mp4": 100})

    root = settings.archive_dir / "temp" / "printer-file-downloads"
    assert not list(root.glob("bundle-*"))


@pytest.mark.asyncio
async def test_build_printer_files_zip_reports_per_file_progress(tmp_path, monkeypatch):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")
    progress: list[tuple[int, int]] = []

    async def fake_download(_ip, _code, remote_path, local_path: Path, **_kwargs):
        if remote_path.endswith("missing.gcode"):
            return False
        local_path.write_bytes(b"ok")
        return True

    async def report(successful: int, failed: int) -> None:
        progress.append((successful, failed))

    with patch(
        "backend.app.services.printer_media.download_file_async",
        new=AsyncMock(side_effect=fake_download),
    ):
        result = await build_printer_files_zip(
            printer,
            ["/ok.gcode", "/missing.gcode"],
            {"/ok.gcode": 2, "/missing.gcode": 2},
            progress_callback=report,
        )

    try:
        assert progress == [(1, 0), (1, 1)]
    finally:
        remove_printer_files_zip(result.path)


@pytest.mark.asyncio
async def test_build_printer_files_zip_serializes_capacity_consumers_across_jobs(tmp_path, monkeypatch):
    """Two preparations cannot spend the same free-space observation concurrently."""
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")
    active = 0
    max_active = 0

    async def fake_download(_ip, _code, _remote_path, local_path: Path, **_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.05)
            local_path.write_bytes(b"ok")
            return True
        finally:
            active -= 1

    with patch(
        "backend.app.services.printer_media.download_file_async",
        new=AsyncMock(side_effect=fake_download),
    ):
        first, second = await asyncio.gather(
            build_printer_files_zip(printer, ["/first.gcode"], {"/first.gcode": 2}),
            build_printer_files_zip(printer, ["/second.gcode"], {"/second.gcode": 2}),
        )

    try:
        assert max_active == 1
    finally:
        remove_printer_files_zip(first.path)
        remove_printer_files_zip(second.path)


@pytest.mark.asyncio
async def test_shutdown_awaits_download_jobs_and_publishes_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "archive_dir", tmp_path / "archive")
    job_id = "shutdown-job-abcdefghijklmnop"
    printer_media._ensure_printer_zip_root()
    started = asyncio.Event()

    async def wait_forever():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_forever())
    printer_media._LOCAL_JOB_TASKS[job_id] = task
    await started.wait()

    await printer_media.stop_printer_download_cleanup()

    assert task.done()
    assert task.cancelled()
    assert printer_media._LOCAL_JOB_TASKS == {}
    assert printer_media._job_cancel_path(job_id).exists()
