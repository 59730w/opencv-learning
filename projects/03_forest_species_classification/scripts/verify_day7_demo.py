import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import MODEL_LIMITATION, build_demo, predict_for_gradio
from src.inference import ForestSpeciesPredictor

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "resnet18_baseline" / "best.pt"
)
CLASS_MAP_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "class_to_idx.json"
)
SAMPLE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "BarkVN-50"
    / "v1"
    / "images"
    / "BarkVN-50_mendeley"
    / "Acacia"
    / "IMG_6353.JPG"
)


def main():
    assert CHECKPOINT_PATH.is_file()
    assert CLASS_MAP_PATH.is_file()
    assert SAMPLE_PATH.is_file()

    predictor = ForestSpeciesPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        class_map_path=CLASS_MAP_PATH,
    )
    predictions = predictor.predict_path(SAMPLE_PATH, top_k=3)

    assert predictor.checkpoint_epoch == 13
    assert len(predictor.class_names) == 50
    assert len(predictions) == 3
    assert predictions[0].class_name == "Acacia"
    assert all(
        predictions[index].probability
        >= predictions[index + 1].probability
        for index in range(len(predictions) - 1)
    )

    with Image.open(SAMPLE_PATH) as image:
        labels, message = predict_for_gradio(image, predictor=predictor)

    assert list(labels) == [item.class_name for item in predictions]
    assert "Top-1: Acacia" in message
    assert "5.00%" in MODEL_LIMITATION

    demo = build_demo()
    config = demo.get_config_file()
    assert config["title"] == "BarkVN-50 树皮分类 Demo"
    assert len(config["components"]) >= 6

    print("设备:", predictor.device)
    print("检查点轮次:", predictor.checkpoint_epoch)
    print("类别数量:", len(predictor.class_names))
    print("验证图片:", SAMPLE_PATH.relative_to(PROJECT_ROOT))
    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"Top-{rank}: {prediction.class_name} "
            f"({prediction.probability:.2%})"
        )
    print("Gradio 组件数量:", len(config["components"]))
    print("第七天 Demo 核心验证成功")


if __name__ == "__main__":
    main()
