import importlib

import gradio as gr


def test_demo_builds_without_loading_the_model():
    app = importlib.import_module("app")

    demo = app.build_demo()

    assert isinstance(demo, gr.Blocks)


def test_demo_api_schema_can_be_generated():
    app = importlib.import_module("app")

    api_info = app.build_demo().get_api_info()

    assert "named_endpoints" in api_info


def test_demo_rejects_missing_image_before_model_loading():
    app = importlib.import_module("app")

    labels, message = app.predict_for_gradio(None)

    assert labels == {}
    assert "upload" in message.lower()


def test_demo_exposes_honest_model_limitations():
    app = importlib.import_module("app")

    assert "5.00%" in app.MODEL_LIMITATION
    assert "close-up bark" in app.MODEL_LIMITATION.lower()
