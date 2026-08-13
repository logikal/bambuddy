import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.services.printer_media import (
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
async def test_build_printer_files_zip_stages_one_file_at_a_time(tmp_path):
    printer = SimpleNamespace(ip_address="printer", access_code="code", model="X1C")

    async def fake_download(_ip, _code, remote_path, local_path: Path, **_kwargs):
        local_path.write_bytes(remote_path.encode())
        return True

    with patch(
        "backend.app.services.printer_media.download_file_async",
        new=AsyncMock(side_effect=fake_download),
    ):
        zip_path, successful = await build_printer_files_zip(
            printer,
            ["/ipcam/chunk.mp4", "/timelapse/chunk.mp4"],
        )

    try:
        assert successful == 2
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.namelist() == ["ipcam/chunk.mp4", "timelapse/chunk.mp4"]
            assert archive.read("ipcam/chunk.mp4") == b"/ipcam/chunk.mp4"
        assert not list(zip_path.parent.glob("download-*"))
    finally:
        remove_printer_files_zip(zip_path)

    assert not zip_path.parent.exists()
