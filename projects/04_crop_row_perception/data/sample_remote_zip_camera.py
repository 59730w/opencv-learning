"""Download selected camera frames from a large remote ZIP using HTTP ranges."""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
import zlib
from pathlib import Path


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(f"range request returned HTTP {response.status}")
        return response.read()


def remote_zip_entries(url: str, total_size: int) -> list[dict[str, int | str]]:
    tail_start = max(0, total_size - 65557)
    tail = fetch_range(url, tail_start, total_size - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    _, _, _, _, entry_count, central_size, central_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_at
    )
    central = fetch_range(url, central_offset, central_offset + central_size - 1)
    entries: list[dict[str, int | str]] = []
    cursor = 0
    for _ in range(entry_count):
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError(f"invalid central directory signature at {cursor}")
        compression, compressed_size, uncompressed_size = fields[4], fields[8], fields[9]
        name_len, extra_len, comment_len, local_offset = fields[10], fields[11], fields[12], fields[16]
        name_start = cursor + 46
        name = central[name_start : name_start + name_len].decode("utf-8")
        entries.append(
            {
                "name": name,
                "compression": compression,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_offset": local_offset,
            }
        )
        cursor = name_start + name_len + extra_len + comment_len
    return entries


def download_entry(url: str, entry: dict[str, int | str]) -> bytes:
    local_offset = int(entry["local_offset"])
    header = fetch_range(url, local_offset, local_offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header for {entry['name']}")
    data_start = local_offset + 30 + fields[9] + fields[10]
    compressed_size = int(entry["compressed_size"])
    compressed = fetch_range(url, data_start, data_start + compressed_size - 1)
    method = int(entry["compression"])
    if method == 0:
        return compressed
    if method == 8:
        return zlib.decompress(compressed, -15)
    raise RuntimeError(f"unsupported compression method {method}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--contains", default="/camera/kinect/color/")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidates = [
        entry
        for entry in remote_zip_entries(args.url, args.size)
        if args.contains in str(entry["name"]).replace("\\", "/")
        and str(entry["name"]).lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not candidates:
        raise RuntimeError("no matching camera images found")
    count = min(args.count, len(candidates))
    indexes = sorted({round(index * (len(candidates) - 1) / max(1, count - 1)) for index in range(count)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, candidate_index in enumerate(indexes):
        entry = candidates[candidate_index]
        content = download_entry(args.url, entry)
        suffix = Path(str(entry["name"])).suffix.lower()
        output = args.output_dir / f"sample_{ordinal:03d}{suffix}"
        output.write_bytes(content)
        records.append(
            {
                "sample_index": candidate_index,
                "archive_path": entry["name"],
                "output": output.name,
                "bytes": len(content),
            }
        )
    manifest = {
        "source_url": args.url,
        "archive_size_bytes": args.size,
        "camera_filter": args.contains,
        "candidate_frame_count": len(candidates),
        "sample_count": len(records),
        "records": records,
    }
    (args.output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest | {"records": records[:2]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
