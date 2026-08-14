import argparse
import csv
import hashlib
import math
import os
import time
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "external_images" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_test"
AUDIT_PATH = OUTPUT_DIR / "external_image_audit.csv"
CONTACT_SHEET_PATH = OUTPUT_DIR / "external_images_contact_sheet.jpg"
EXPECTED_TOTAL = 26


def read_metadata():
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {
        "filename",
        "test_type",
        "expected_class",
        "view_type",
        "direct_image_url",
        "creator",
        "license",
        "record_id",
    }
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"清单应有{EXPECTED_TOTAL}条记录，实际为{len(rows)}条")
    if not rows or not required.issubset(rows[0]):
        raise ValueError("metadata.csv缺少必需字段")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("metadata.csv存在重复文件名")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("metadata.csv存在重复来源记录")
    return rows


def destination_for(row):
    test_type = row["test_type"]
    if test_type not in {"positive", "negative"}:
        raise ValueError(f"未知test_type：{test_type}")
    return PROJECT_ROOT / "external_images" / test_type / row["filename"]


def thumbnail_url(row, width=1600):
    marker = "File:"
    if marker not in row["source_page_url"]:
        raise ValueError(f"无法从来源页解析Commons文件名：{row['source_page_url']}")
    commons_filename = unquote(row["source_page_url"].split(marker, 1)[1])
    request = requests.Request(
        "GET",
        "https://commons.wikimedia.org/w/thumb.php",
        params={"f": commons_filename, "w": width},
    ).prepare()
    return request.url


def download_one(session, row, destination, retries, delay):
    if destination.exists() and destination.stat().st_size > 0:
        print(f"已存在，跳过下载：{row['filename']}", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)

    original_url = row["direct_image_url"]
    fallback_url = thumbnail_url(row)
    current_url = original_url
    for attempt in range(1, retries + 1):
        try:
            with session.get(current_url, stream=True, timeout=(20, 120)) as response:
                if response.status_code == 429 and current_url == original_url:
                    current_url = fallback_url
                    print(
                        f"{row['filename']}：原图入口限流，切换Commons官方1600px入口",
                        flush=True,
                    )
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    requested_wait = float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** (attempt - 1)
                    wait_seconds = min(60, requested_wait)
                    print(
                        f"{row['filename']}：HTTP {response.status_code}，{wait_seconds:.0f}秒后重试",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if not content_type.startswith("image/"):
                    raise RuntimeError(f"响应不是图片：{content_type or 'unknown'}")

                with temporary.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
                if temporary.stat().st_size == 0:
                    raise RuntimeError("下载文件为空")
                os.replace(temporary, destination)
                print(
                    f"下载完成：{row['filename']} ({destination.stat().st_size / 1024 / 1024:.2f} MB)",
                    flush=True,
                )
                time.sleep(delay)
                return
        except (requests.RequestException, RuntimeError, OSError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"下载失败：{row['filename']}：{error}") from error
            wait_seconds = min(60, 5 * 2 ** (attempt - 1))
            print(f"{row['filename']}：{error}，{wait_seconds}秒后重试", flush=True)
            time.sleep(wait_seconds)

    raise RuntimeError(f"下载失败：{row['filename']}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(row, path):
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV无法读取：{path}")
    height, width, channels = image.shape
    if width < 224 or height < 224:
        raise ValueError(f"图片尺寸过小：{path.name} = {width}x{height}")
    return {
        "filename": row["filename"],
        "test_type": row["test_type"],
        "expected_class": row["expected_class"],
        "view_type": row["view_type"],
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_audit(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with AUDIT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(metadata_rows):
    columns = 4
    cell_width, image_height, label_height = 360, 270, 58
    rows = math.ceil(len(metadata_rows) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, row in enumerate(metadata_rows):
        path = destination_for(row)
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            preview = ImageOps.contain(source, (cell_width - 12, image_height - 12))
        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + label_height)
        image_x = x + (cell_width - preview.width) // 2
        image_y = y + (image_height - preview.height) // 2
        canvas.paste(preview, (image_x, image_y))
        draw.rectangle((x, y, x + cell_width - 1, y + image_height + label_height - 1), outline="gray")
        expected = row["expected_class"] or "negative"
        draw.text((x + 6, y + image_height + 5), row["filename"], fill="black", font=font)
        draw.text((x + 6, y + image_height + 25), f"{expected} | {row['view_type']}", fill="black", font=font)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(CONTACT_SHEET_PATH, quality=92)


def update_metadata_status(rows):
    fieldnames = list(rows[0])
    today = date.today().isoformat()
    for row in rows:
        row["download_date"] = today
        row["notes"] = (
            "downloaded from recorded original URL or official Commons 1600px fallback; "
            "machine-validated; pending manual visual confirmation"
        )
    with METADATA_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="下载并校验固定的外部测试图片")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    metadata_rows = read_metadata()
    session = requests.Session()
    session.headers.update({
        "User-Agent": "forest-species-classification-learning/1.0 (educational external validation)",
        "Referer": "https://commons.wikimedia.org/",
    })

    for index, row in enumerate(metadata_rows, start=1):
        print(f"[{index:02d}/{len(metadata_rows)}] {row['filename']}", flush=True)
        download_one(session, row, destination_for(row), args.retries, args.delay)

    audit_rows = [inspect_image(row, destination_for(row)) for row in metadata_rows]
    if len({row["sha256"] for row in audit_rows}) != len(audit_rows):
        raise ValueError("外部测试图片存在完全重复文件")
    write_audit(audit_rows)
    make_contact_sheet(metadata_rows)
    update_metadata_status(metadata_rows)

    positive_count = sum(row["test_type"] == "positive" for row in metadata_rows)
    negative_count = sum(row["test_type"] == "negative" for row in metadata_rows)
    total_size = sum(row["file_size_bytes"] for row in audit_rows)
    print(f"正样本：{positive_count}", flush=True)
    print(f"负样本：{negative_count}", flush=True)
    print(f"唯一SHA-256：{len({row['sha256'] for row in audit_rows})}", flush=True)
    print(f"总占用：{total_size / 1024 / 1024:.2f} MB", flush=True)
    print(f"校验清单：{AUDIT_PATH}", flush=True)
    print(f"联系表：{CONTACT_SHEET_PATH}", flush=True)
    print("第六天第三步机器校验成功；尚未运行模型", flush=True)


if __name__ == "__main__":
    main()
