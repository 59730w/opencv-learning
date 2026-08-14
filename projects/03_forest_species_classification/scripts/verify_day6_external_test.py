import csv
import hashlib
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external_images"
STRICT_OUTPUT = PROJECT_ROOT / "outputs" / "external_test"
POSTHOC_OUTPUT = PROJECT_ROOT / "outputs" / "external_test_posthoc_multicrop"


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_readable_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise AssertionError(f"图片不可读取：{path}")
    return image.shape


def main():
    metadata = load_csv(EXTERNAL_ROOT / "metadata.csv")
    audit = load_csv(STRICT_OUTPUT / "external_image_audit.csv")
    strict_predictions = load_csv(STRICT_OUTPUT / "external_predictions.csv")
    strict_metrics = load_json(STRICT_OUTPUT / "external_metrics.json")
    posthoc_predictions = load_csv(
        POSTHOC_OUTPUT / "multicrop_external_predictions.csv"
    )
    posthoc_metrics = load_json(POSTHOC_OUTPUT / "multicrop_metrics.json")
    sequence_risk = load_json(STRICT_OUTPUT / "sequence_split_risk.json")
    report_path = STRICT_OUTPUT / "external_evaluation_report.md"
    report = report_path.read_text(encoding="utf-8")

    image_paths = sorted(
        list((EXTERNAL_ROOT / "positive").glob("*.jpg"))
        + list((EXTERNAL_ROOT / "negative").glob("*.jpg"))
    )
    actual_hashes = {path.name: sha256(path) for path in image_paths}
    audited_hashes = {
        row["filename"]: row["sha256"] for row in audit
    }

    positive_metadata = [
        row for row in metadata if row["test_type"] == "positive"
    ]
    negative_metadata = [
        row for row in metadata if row["test_type"] == "negative"
    ]
    strict_positive = [
        row for row in strict_predictions if row["test_type"] == "positive"
    ]
    posthoc_positive = [
        row for row in posthoc_predictions if row["test_type"] == "positive"
    ]

    strict_top1 = sum(
        row["top1_correct"] == "True" for row in strict_positive
    )
    strict_top3 = sum(
        row["expected_in_top3"] == "True" for row in strict_positive
    )
    posthoc_top1 = sum(
        row["top1_correct"] == "True" for row in posthoc_positive
    )
    posthoc_top3 = sum(
        row["expected_in_top3"] == "True" for row in posthoc_positive
    )

    assert len(metadata) == 26
    assert len(positive_metadata) == 20
    assert len(negative_metadata) == 6
    assert len({row["filename"] for row in metadata}) == 26
    assert len({row["record_id"] for row in metadata}) == 26
    assert len(image_paths) == 26
    assert len(actual_hashes) == 26
    assert len(set(actual_hashes.values())) == 26
    assert actual_hashes == audited_hashes
    assert not list(EXTERNAL_ROOT.rglob("*.part"))

    assert len(strict_predictions) == 26
    assert strict_top1 == 1
    assert strict_top3 == 4
    assert strict_metrics["checkpoint_epoch"] == 13
    assert strict_metrics["positive"]["top1_correct"] == strict_top1
    assert strict_metrics["positive"]["top3_correct"] == strict_top3
    assert strict_metrics["negative"]["images"] == 6

    assert len(posthoc_predictions) == 26
    assert posthoc_top1 == 2
    assert posthoc_top3 == 3
    assert posthoc_metrics["strict_external_test"] is False
    assert (
        posthoc_metrics["external_multicrop"]["positive"]["top1_correct"]
        == posthoc_top1
    )
    assert (
        posthoc_metrics["external_multicrop"]["positive"]["top3_correct"]
        == posthoc_top3
    )

    assert sequence_risk["manifest_images"] == 5558
    assert sequence_risk["adjacent_cross_split_pairs"] == 2442
    assert sequence_risk["classes_involved"] == 50
    assert len(sequence_risk["pairs"]) == 2442

    image_artifacts = [
        STRICT_OUTPUT / "external_images_contact_sheet.jpg",
        STRICT_OUTPUT / "external_prediction_contact_sheet.jpg",
        STRICT_OUTPUT / "generalization_gap.png",
        STRICT_OUTPUT / "external_prediction_bias.png",
    ]
    artifact_shapes = {
        path.name: require_readable_image(path) for path in image_artifacts
    }

    required_report_text = [
        "严格外部测试",
        "| Top-1 | 1/20 | 5.00% |",
        "| Top-3 | 4/20 | 20.00% |",
        "2,442 对",
        "不是新的严格外部测试",
        "尚不是可部署的野外树种识别系统",
        "当前26张图片为冻结证据",
    ]
    for text in required_report_text:
        assert text in report, f"报告缺少关键内容：{text}"

    print("外部图片/正样本/负样本:", len(metadata), len(positive_metadata), len(negative_metadata))
    print("唯一SHA-256:", len(set(actual_hashes.values())))
    print("严格Top-1/Top-3:", strict_top1, strict_top3)
    print("post-hoc Top-1/Top-3:", posthoc_top1, posthoc_top3)
    print("相邻编号跨子集风险对:", sequence_risk["adjacent_cross_split_pairs"])
    print("可读取图像成果:", artifact_shapes)
    print("正式报告:", report_path)
    print("第六天全部成果验证成功")


if __name__ == "__main__":
    main()
