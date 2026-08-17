import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.core.config import settings
from backend.app.services.printer_media import (
    MAX_PRINTER_ZIP_BYTES,
    PrinterFilesZipInsufficientSpaceError,
    PrinterFilesZipTooLargeError,
    build_printer_files_zip,
    match_ipcam_chunks,
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

    async def fake_download(_ip, _code, remote_path, local_path: Path, **_kwargs):
        local_path.write_bytes((remote_path.encode() + b"-") * 512)
        return True

    with patch(
        "backend.app.services.printer_media.download_file_async",
        new=AsyncMock(side_effect=fake_download),
    ):
        result = await build_printer_files_zip(
            printer,
            ["/ipcam/chunk.mp4", "/cache/model.gcode"],
            {"/ipcam/chunk.mp4": 10_000, "/cache/model.gcode": 10_000},
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
