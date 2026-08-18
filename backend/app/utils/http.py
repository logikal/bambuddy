"""HTTP response helpers."""

from pathlib import Path
from urllib.parse import quote


def safe_download_filename(filename: str, fallback: str = "download", max_chars: int = 200) -> str:
    """Return a basename safe for a bounded download response header."""

    basename = Path(filename.replace("\\", "/")).name
    cleaned = "".join("_" if ord(char) < 32 or ord(char) == 127 else char for char in basename).strip(" .")
    if not cleaned:
        return fallback
    if len(cleaned) <= max_chars:
        return cleaned
    suffixes = "".join(Path(cleaned).suffixes)
    suffix = suffixes if len(suffixes) <= 32 else ""
    stem_chars = max(1, max_chars - len(suffix))
    return f"{cleaned[:stem_chars]}{suffix}"


def build_content_disposition(filename: str, disposition: str = "attachment") -> str:
    """Build an RFC 6266-compliant Content-Disposition header value.

    Starlette/uvicorn encodes response headers as latin-1, so any non-ASCII
    character in a raw `filename="..."` parameter raises UnicodeEncodeError.
    The fix is RFC 5987's `filename*=UTF-8''<percent-encoded>` form alongside
    a stripped ASCII fallback in the legacy `filename="..."` parameter — every
    modern browser prefers the `*` form when present.
    """
    ascii_fallback = filename.encode("ascii", "ignore").decode("ascii").strip(" ._-") or "download"
    ascii_fallback = ascii_fallback.replace('"', "").replace("\\", "")
    return f"{disposition}; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quote(filename)}"
