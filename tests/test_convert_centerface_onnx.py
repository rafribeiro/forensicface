from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from forensicface.tools.convert_centerface_onnx import (
    CONVERSION_ID,
    ConversionError,
    OUTPUT_NAMES,
    convert_centerface_model,
    main,
    sha256_file,
    verify_conversion,
)


def _fixed_centerface_model() -> onnx.ModelProto:
    image_input = helper.make_tensor_value_info(
        "input.1",
        TensorProto.FLOAT,
        [10, 3, 32, 32],
    )
    graph_inputs = [image_input]
    initializers = []
    nodes = []
    outputs = []

    for index, (name, channels) in enumerate(
        zip(("537", "538", "539", "540"), (1, 2, 2, 10))
    ):
        weight_name = f"head.{index}.weight"
        weights = np.arange(channels * 3, dtype=np.float32).reshape(
            channels,
            3,
            1,
            1,
        )
        weights = (weights + 1.0) / 10.0
        initializers.append(numpy_helper.from_array(weights, name=weight_name))
        # Match the old-export pattern: initializers are also graph inputs.
        graph_inputs.append(
            helper.make_tensor_value_info(
                weight_name,
                TensorProto.FLOAT,
                list(weights.shape),
            )
        )
        nodes.append(
            helper.make_node(
                "Conv",
                ["input.1", weight_name],
                [name],
                strides=[4, 4],
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                name,
                TensorProto.FLOAT,
                [10, channels, 8, 8],
            )
        )

    unused_name = "unused.num_batches_tracked"
    unused_value = np.array(0, dtype=np.int64)
    initializers.append(numpy_helper.from_array(unused_value, name=unused_name))
    graph_inputs.append(
        helper.make_tensor_value_info(
            unused_name,
            TensorProto.INT64,
            [],
        )
    )

    graph = helper.make_graph(
        nodes,
        "synthetic-centerface",
        graph_inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="synthetic-pytorch",
        opset_imports=[helper.make_opsetid("", 9)],
    )
    model.ir_version = 4
    onnx.checker.check_model(model)
    return model


def _save_source(tmp_path: Path) -> Path:
    source = tmp_path / "centerface.onnx"
    onnx.save_model(_fixed_centerface_model(), source)
    return source


def _shape(value_info) -> list[int | str]:
    dimensions = value_info.type.tensor_type.shape.dim
    return [
        int(item.dim_value) if item.HasField("dim_value") else item.dim_param
        for item in dimensions
    ]


def test_convert_centerface_model_corrects_graph_interface():
    source = _fixed_centerface_model()
    corrected = convert_centerface_model(
        source,
        source_sha256="abc123",
    )

    assert len(source.graph.input) == 6
    assert len(corrected.graph.input) == 1
    assert _shape(corrected.graph.input[0]) == ["batch", 3, "height", "width"]
    assert [item.name for item in corrected.graph.output] == list(OUTPUT_NAMES)
    assert [_shape(item) for item in corrected.graph.output] == [
        ["batch", 1, "output_height", "output_width"],
        ["batch", 2, "output_height", "output_width"],
        ["batch", 2, "output_height", "output_width"],
        ["batch", 10, "output_height", "output_width"],
    ]
    assert len(corrected.graph.initializer) == 4
    metadata = {item.key: item.value for item in corrected.metadata_props}
    assert metadata["forensicface.conversion"] == CONVERSION_ID
    assert metadata["forensicface.source_sha256"] == "abc123"
    assert metadata["forensicface.output_order"] == ",".join(OUTPUT_NAMES)
    assert metadata["forensicface.removed_unused_initializers"] == "1"
    onnx.checker.check_model(corrected)


def test_conversion_preserves_outputs_and_supports_dynamic_shapes():
    source = _fixed_centerface_model()
    corrected = convert_centerface_model(source, source_sha256="abc123")

    verify_conversion(
        source,
        corrected,
        dynamic_test_shape=(2, 3, 64, 96),
    )


def test_conversion_can_preserve_numeric_output_names():
    corrected = convert_centerface_model(
        _fixed_centerface_model(),
        source_sha256="abc123",
        rename_outputs=False,
    )

    assert [item.name for item in corrected.graph.output] == [
        "537",
        "538",
        "539",
        "540",
    ]


def test_conversion_rejects_non_centerface_output_contract():
    source = _fixed_centerface_model()
    source.graph.output[-1].type.tensor_type.shape.dim[1].dim_value = 8

    with pytest.raises(ConversionError, match="expected 10"):
        convert_centerface_model(source, source_sha256="abc123")


def test_main_writes_verified_model_and_refuses_overwrite(tmp_path, capsys):
    source = _save_source(tmp_path)
    destination = tmp_path / "detection" / "centerface" / "centerface.onnx"
    source_hash = sha256_file(source)

    rc = main(
        [
            str(source),
            str(destination),
            "--expected-source-sha256",
            source_hash,
        ]
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert destination.is_file()
    assert "source parity and dynamic-shape checks passed" in output
    corrected = onnx.load(destination)
    assert [item.name for item in corrected.graph.output] == list(OUTPUT_NAMES)

    rc = main([str(source), str(destination)])
    error = capsys.readouterr().err
    assert rc == 2
    assert "--force" in error


def test_main_rejects_source_hash_mismatch_without_writing(tmp_path, capsys):
    source = _save_source(tmp_path)
    destination = tmp_path / "corrected.onnx"

    rc = main(
        [
            str(source),
            str(destination),
            "--expected-source-sha256",
            "0" * 64,
        ]
    )

    assert rc == 2
    assert not destination.exists()
    assert "SHA-256 mismatch" in capsys.readouterr().err
