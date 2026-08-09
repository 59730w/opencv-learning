# Experiment Log

## Baseline

- Model: YOLOv8n
- Epochs: 50
- Image size: 640
- Batch size: 4
- Device: NVIDIA GeForce RTX 3050 Laptop GPU
- Training images: 516
- Validation images: 147
- Test images: 74
- Class: smoke

## Validation Metrics

- Precision: 0.875
- Recall: 0.898
- mAP50: 0.924
- mAP50-95: 0.537

## Test Metrics

- Precision: 0.9271
- Recall: 0.8378
- mAP50: 0.9485
- mAP50-95: 0.5614

## External Inference

Confidence threshold: 0.25

- External smoke images: 5
- Smoke images detected: 0
- Missed smoke images: 5
- Hard negative images: 8
- False-positive images: 0

## Error Analysis

The model performs well on the original dataset but fails to detect smoke
in external images. The original dataset contains many visually similar
scenes and no background-only images in the test split.

A low-confidence diagnostic showed that external smoke predictions had a
maximum confidence of only 0.058. A foggy negative image reached 0.145,
so simply lowering the confidence threshold would introduce false positives
without solving the missed detections.

## Conclusion

The baseline successfully verifies the complete YOLOv8 engineering workflow,
but its real-world generalization is limited. Future improvement should use
more diverse smoke images, hard negative images, and scene-independent data
splits. The current external test images should remain outside the training
set for honest comparison.
