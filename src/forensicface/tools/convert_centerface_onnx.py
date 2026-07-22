"""Create an ONNX Runtime-friendly CenterFace model artifact.

The CenterFace ONNX file distributed with the reference implementation was
exported by an old PyTorch version.  Although the network is fully
convolutional, its graph metadata declares a fixed ``[10, 3, 32, 32]`` input,
fixed output shapes, and exposes initializers as graph inputs.  OpenCV DNN
accepts that export at arbitrary image sizes, but ONNX Runtime correctly
enforces the declared fixed dimensions.

This offline tool makes only graph-interface corrections; it does not modify
the learned tensor values:

* batch, height, and width become dynamic;
* initializer tensors are removed from ``graph.input``;
* initializers unused by every graph node are removed;
* the four outputs receive stable semantic names by default;
* the source checksum and conversion identity are recorded in model metadata;
* ONNX validation and ONNX Runtime numerical parity checks run before saving.

Usage::

    python -m forensicface.tools.convert_centerface_onnx \
        centerface.onnx \
        ~/.forensicface/models/detection/centerface/centerface.onnx

The destination is never overwritten unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np
import onnx
import onnxruntime as ort


CONVERSION_ID = "forensicface.centerface.dynamic-v1"
OUTPUT_NAMES = ("heatmap", "scale", "offset", "landmarks")
OUTPUT_CHANNELS = (1, 2, 2, 10)
DEFAULT_DYNAMIC_TEST_SHAPE = (1, 3, 64, 96)


class ConversionError(ValueError):
    """Raised when a source model does not satisfy the CenterFace contract."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dimension_value(dimension) -> int | None:
    if dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    return None


def _set_symbolic_dimension(dimension, name: str) -> None:
    dimension.ClearField("dim_value")
    dimension.dim_param = name


def _set_fixed_dimension(dimension, value: int) -> None:
    dimension.ClearField("dim_param")
    dimension.dim_value = value


def _runtime_graph_inputs(model: onnx.ModelProto):
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return [item for item in model.graph.input if item.name not in initializer_names]


def _validate_source_contract(model: onnx.ModelProto) -> None:
    runtime_inputs = _runtime_graph_inputs(model)
    if len(runtime_inputs) != 1:
        raise ConversionError(
            "Expected exactly one CenterFace image input after excluding "
            f"initializers; found {[item.name for item in runtime_inputs]}."
        )

    input_shape = runtime_inputs[0].type.tensor_type.shape.dim
    if len(input_shape) != 4:
        raise ConversionError(
            f"Expected a rank-4 NCHW image input; found rank {len(input_shape)}."
        )
    channels = _dimension_value(input_shape[1])
    if channels not in (None, 3):
        raise ConversionError(
            f"Expected a three-channel image input; found channel dimension {channels}."
        )

    if len(model.graph.output) != 4:
        raise ConversionError(
            "Expected four CenterFace outputs in heatmap, scale, offset, "
            f"landmark order; found {[item.name for item in model.graph.output]}."
        )

    for output, expected_channels in zip(model.graph.output, OUTPUT_CHANNELS):
        dimensions = output.type.tensor_type.shape.dim
        if len(dimensions) != 4:
            raise ConversionError(
                f"Expected rank-4 output '{output.name}'; found rank {len(dimensions)}."
            )
        channels = _dimension_value(dimensions[1])
        if channels not in (None, expected_channels):
            raise ConversionError(
                f"Output '{output.name}' has {channels} channels; expected "
                f"{expected_channels} for the CenterFace output contract."
            )


def _rename_graph_value(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return

    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    if new_name in existing_names:
        raise ConversionError(
            f"Cannot rename output '{old_name}' to '{new_name}': "
            "the target name already exists in the graph."
        )

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old_name:
                node.input[index] = new_name
        for index, name in enumerate(node.output):
            if name == old_name:
                node.output[index] = new_name

    for collection in (
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
    ):
        for value_info in collection:
            if value_info.name == old_name:
                value_info.name = new_name


def convert_centerface_model(
    source_model: onnx.ModelProto,
    *,
    source_sha256: str,
    rename_outputs: bool = True,
) -> onnx.ModelProto:
    """Return a corrected copy of a CenterFace ONNX model.

    Learned initializers are retained byte-for-byte.  Only graph interface
    metadata, initializer exposure, output names, and model metadata change.
    """
    model = onnx.ModelProto()
    model.CopyFrom(source_model)
    _validate_source_contract(model)

    runtime_input = _runtime_graph_inputs(model)[0]
    input_dimensions = runtime_input.type.tensor_type.shape.dim
    _set_symbolic_dimension(input_dimensions[0], "batch")
    _set_fixed_dimension(input_dimensions[1], 3)
    _set_symbolic_dimension(input_dimensions[2], "height")
    _set_symbolic_dimension(input_dimensions[3], "width")

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    retained_inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)

    referenced_names = {
        name
        for node in model.graph.node
        for name in node.input
        if name
    }
    retained_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in referenced_names
    ]
    removed_initializer_count = (
        len(model.graph.initializer) - len(retained_initializers)
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)

    if rename_outputs:
        original_output_names = [output.name for output in model.graph.output]
        for old_name, new_name in zip(original_output_names, OUTPUT_NAMES):
            _rename_graph_value(model, old_name, new_name)

    for output, expected_channels in zip(model.graph.output, OUTPUT_CHANNELS):
        dimensions = output.type.tensor_type.shape.dim
        _set_symbolic_dimension(dimensions[0], "batch")
        _set_fixed_dimension(dimensions[1], expected_channels)
        _set_symbolic_dimension(dimensions[2], "output_height")
        _set_symbolic_dimension(dimensions[3], "output_width")

    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update(
        {
            "forensicface.conversion": CONVERSION_ID,
            "forensicface.source_sha256": source_sha256,
            "forensicface.output_order": ",".join(OUTPUT_NAMES),
            "forensicface.removed_unused_initializers": str(
                removed_initializer_count
            ),
        }
    )
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    return model


def _ort_session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        return ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ConversionError(f"ONNX Runtime could not load the model: {exc}") from exc


def _run_session(session: ort.InferenceSession, input_value: np.ndarray):
    try:
        return session.run(
            None,
            {session.get_inputs()[0].name: input_value},
        )
    except Exception as exc:
        raise ConversionError(f"ONNX Runtime inference failed: {exc}") from exc


def _concrete_source_shape(session: ort.InferenceSession) -> tuple[int, ...]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ConversionError(
            f"Expected one ONNX Runtime image input; found {[item.name for item in inputs]}."
        )
    shape = inputs[0].shape
    if len(shape) != 4 or not all(isinstance(value, int) for value in shape):
        raise ConversionError(
            "The source model must declare one concrete NCHW shape for the "
            f"numerical parity check; found {shape}."
        )
    return tuple(int(value) for value in shape)


def verify_conversion(
    source_model: onnx.ModelProto,
    corrected_model: onnx.ModelProto,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    dynamic_test_shape: tuple[int, int, int, int] = DEFAULT_DYNAMIC_TEST_SHAPE,
) -> None:
    """Verify source parity and dynamic-shape execution with ONNX Runtime."""
    source_session = _ort_session(source_model)
    corrected_session = _ort_session(corrected_model)
    source_shape = _concrete_source_shape(source_session)

    generator = np.random.default_rng(0)
    source_input = generator.uniform(0.0, 255.0, size=source_shape).astype(np.float32)
    source_outputs = _run_session(source_session, source_input)
    corrected_outputs = _run_session(corrected_session, source_input)

    if len(source_outputs) != len(corrected_outputs):
        raise ConversionError(
            "Corrected model returned a different number of outputs during "
            "the numerical parity check."
        )
    for index, (source_output, corrected_output) in enumerate(
        zip(source_outputs, corrected_outputs)
    ):
        if not np.allclose(source_output, corrected_output, rtol=rtol, atol=atol):
            maximum_difference = float(
                np.max(np.abs(source_output - corrected_output))
            )
            raise ConversionError(
                f"Output {index} failed numerical parity: maximum absolute "
                f"difference {maximum_difference} exceeds rtol={rtol}, atol={atol}."
            )

    if len(dynamic_test_shape) != 4 or dynamic_test_shape[1] != 3:
        raise ConversionError(
            "Dynamic test shape must be NCHW with three channels; received "
            f"{dynamic_test_shape}."
        )
    batch, _, height, width = dynamic_test_shape
    if height % 32 or width % 32:
        raise ConversionError(
            "CenterFace dynamic test height and width must be multiples of 32; "
            f"received {height}x{width}."
        )

    dynamic_input = np.zeros(dynamic_test_shape, dtype=np.float32)
    dynamic_outputs = _run_session(corrected_session, dynamic_input)
    expected_spatial_shape = (height // 4, width // 4)
    for name, channels, output in zip(
        OUTPUT_NAMES,
        OUTPUT_CHANNELS,
        dynamic_outputs,
    ):
        expected_shape = (batch, channels, *expected_spatial_shape)
        if output.shape != expected_shape:
            raise ConversionError(
                f"Dynamic output '{name}' has shape {output.shape}; expected "
                f"{expected_shape}."
            )


def _save_atomically(model: onnx.ModelProto, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        onnx.save_model(model, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forensicface.tools.convert_centerface_onnx",
        description=(
            "Correct the old CenterFace ONNX export for dynamic ONNX Runtime "
            "inference and verify numerical parity before saving."
        ),
    )
    parser.add_argument("source", type=Path, help="source CenterFace ONNX file")
    parser.add_argument("destination", type=Path, help="corrected ONNX output path")
    parser.add_argument(
        "--expected-source-sha256",
        help="refuse conversion unless the source file has this SHA-256",
    )
    parser.add_argument(
        "--keep-output-names",
        action="store_true",
        help="preserve numeric source output names instead of semantic names",
    )
    parser.add_argument(
        "--skip-runtime-verification",
        action="store_true",
        help="skip ONNX Runtime numerical and dynamic-shape checks",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing destination after successful validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source = args.source.expanduser()
    destination = args.destination.expanduser()

    if not source.is_file():
        print(f"Source model does not exist: {source}", file=sys.stderr)
        return 2
    if source.resolve() == destination.resolve():
        print("Source and destination must be different files.", file=sys.stderr)
        return 2
    if destination.exists() and not args.force:
        print(
            f"Destination already exists: {destination}. Pass --force to overwrite it.",
            file=sys.stderr,
        )
        return 2

    source_hash = sha256_file(source)
    if (
        args.expected_source_sha256 is not None
        and source_hash.lower() != args.expected_source_sha256.lower()
    ):
        print(
            "Source SHA-256 mismatch: "
            f"expected {args.expected_source_sha256.lower()}, found {source_hash}.",
            file=sys.stderr,
        )
        return 2

    try:
        source_model = onnx.load(source)
        onnx.checker.check_model(source_model)
        corrected_model = convert_centerface_model(
            source_model,
            source_sha256=source_hash,
            rename_outputs=not args.keep_output_names,
        )
        if not args.skip_runtime_verification:
            verify_conversion(
                source_model,
                corrected_model,
                rtol=args.rtol,
                atol=args.atol,
            )
        onnx.checker.check_model(corrected_model)
        _save_atomically(corrected_model, destination)
    except (ConversionError, OSError, onnx.checker.ValidationError) as exc:
        print(f"CenterFace conversion failed: {exc}", file=sys.stderr)
        return 3

    saved_hash = sha256_file(destination)
    print(f"Source:      {source}")
    print(f"Source SHA:  {source_hash}")
    print(f"Destination: {destination}")
    print(f"Output SHA:  {saved_hash}")
    print(f"Outputs:     {[item.name for item in corrected_model.graph.output]}")
    print("Validation:  ONNX checker passed")
    if args.skip_runtime_verification:
        print("Runtime:     skipped by request")
    else:
        print("Runtime:     source parity and dynamic-shape checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
