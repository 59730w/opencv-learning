from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import onnx
import torch
from onnx import checker, shape_inference

from day53_common import IMAGE_SIZE, build_model_from_checkpoint, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the forest-species ResNet18 checkpoint to ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    dimensions = value_info.type.tensor_type.shape.dim
    shape: list[int | str | None] = []
    for dimension in dimensions:
        if dimension.HasField("dim_value"):
            shape.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    model, class_names, checkpoint = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        class_map_path=args.class_map,
    )
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        (example_input,),
        args.output,
        input_names=["images"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=False,
        dynamic_axes={
            "images": {0: "batch"},
            "logits": {0: "batch"},
        },
        external_data=False,
    )

    onnx_model = onnx.load(args.output)
    checker.check_model(onnx_model)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    onnx.helper.set_model_props(
        inferred_model,
        {
            "lesson": "Day53 PyTorch to ONNX export",
            "task": "50-class close-up bark classification learning baseline",
            "input_preprocess": (
                "RGB; Resize short side to 256; CenterCrop 224; ToTensor; "
                "ImageNet mean/std normalization"
            ),
            "class_count": str(len(class_names)),
            "checkpoint_epoch": str(checkpoint.get("epoch")),
            "limitation": "closed-set same-source baseline; no unknown rejection",
        },
    )
    onnx.save(inferred_model, args.output)
    checker.check_model(onnx.load(args.output))

    graph = inferred_model.graph
    input_info = graph.input[0]
    output_info = graph.output[0]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "class_count": len(class_names),
        "onnx_path": str(args.output.resolve()),
        "onnx_size_bytes": args.output.stat().st_size,
        "onnx_sha256": sha256_file(args.output),
        "opset_imports": {
            item.domain or "ai.onnx": item.version
            for item in inferred_model.opset_import
        },
        "input": {"name": input_info.name, "shape": tensor_shape(input_info)},
        "output": {"name": output_info.name, "shape": tensor_shape(output_info)},
        "node_count": len(graph.node),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["torch", "torchvision", "onnx", "onnxruntime", "onnxscript"]
        },
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"ONNX model: {args.output.resolve()}")
    print(f"Input: {summary['input']['name']} {summary['input']['shape']}")
    print(f"Output: {summary['output']['name']} {summary['output']['shape']}")
    print(f"Classes: {summary['class_count']}")
    print(f"Nodes: {summary['node_count']}")
    print(f"SHA256: {summary['onnx_sha256']}")
    print("DAY53_ONNX_EXPORT_OK")


if __name__ == "__main__":
    main()
