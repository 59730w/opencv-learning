import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ForestSpeciesPredictor

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "resnet18_baseline" / "best.pt"
)
DEFAULT_CLASS_MAP = (
    PROJECT_ROOT / "datasets" / "processed" / "class_to_idx.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Top-k inference on one close-up bark image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    return parser.parse_args()


def main():
    args = parse_args()
    device = None if args.device == "auto" else args.device
    predictor = ForestSpeciesPredictor(
        checkpoint_path=args.checkpoint,
        class_map_path=args.class_map,
        device=device,
    )
    predictions = predictor.predict_path(args.image, top_k=args.top_k)

    print("设备:", predictor.device)
    print("检查点轮次:", predictor.checkpoint_epoch)
    print("输入图片:", args.image.resolve())
    print(f"Top-{args.top_k} 预测:")
    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"{rank}. {prediction.class_name} | "
            f"索引 {prediction.class_index} | "
            f"概率 {prediction.probability:.2%}"
        )
    print("限制: 仅供近距离树皮闭集分类学习演示，不具备未知类别拒绝能力。")


if __name__ == "__main__":
    main()
