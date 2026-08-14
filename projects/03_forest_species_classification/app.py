from functools import lru_cache
from pathlib import Path

import gradio as gr

from src.inference import ForestSpeciesPredictor

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "resnet18_baseline" / "best.pt"
)
CLASS_MAP_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "class_to_idx.json"
)

MODEL_LIMITATION = (
    "This is a 50-class close-up bark classifier. Its strict external Top-1 "
    "accuracy was only 5.00%. The confidence is a closed-set score, not proof "
    "that the image belongs to one of these species. Do not use it for field "
    "identification or safety-critical decisions."
)

LIMITATION_ZH = (
    "⚠️ **适用边界：**本 Demo 是 50 类近距离树皮闭集分类器。严格外部测试 "
    "Top-1 仅为 **5.00%**。概率只表示模型在 50 类中的相对倾向，不能证明上传图片"
    "一定属于这些树种，也不能用于野外可靠鉴定。"
)


@lru_cache(maxsize=1)
def get_predictor():
    return ForestSpeciesPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        class_map_path=CLASS_MAP_PATH,
    )


def predict_for_gradio(image, predictor=None):
    if image is None:
        return {}, "Please upload a close-up bark image first."

    active_predictor = predictor if predictor is not None else get_predictor()
    predictions = active_predictor.predict(image, top_k=3)
    labels = {
        prediction.class_name: prediction.probability
        for prediction in predictions
    }
    lines = [
        f"Top-{rank}: {prediction.class_name} "
        f"({prediction.probability:.2%})"
        for rank, prediction in enumerate(predictions, start=1)
    ]
    lines.append(
        "提示：该结果仅供学习演示；模型不具备未知类别拒绝能力。"
    )
    return labels, "\n".join(lines)


def build_demo():
    with gr.Blocks(title="BarkVN-50 树皮分类 Demo") as demo:
        gr.Markdown("# BarkVN-50 树皮分类 Demo")
        gr.Markdown(
            "上传一张近距离、树皮占画面主体的图片，模型将给出 Top-3 预测。"
        )
        gr.Markdown(LIMITATION_ZH)

        with gr.Row():
            image_input = gr.Image(
                type="pil",
                label="树皮图片",
                sources=["upload", "webcam", "clipboard"],
            )
            prediction_output = gr.Label(
                num_top_classes=3,
                label="Top-3 预测",
            )

        predict_button = gr.Button("开始识别", variant="primary")
        status_output = gr.Textbox(
            label="详细结果与限制",
            lines=5,
            interactive=False,
        )
        predict_button.click(
            fn=predict_for_gradio,
            inputs=image_input,
            outputs=[prediction_output, status_output],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(inbrowser=True, show_error=True)
