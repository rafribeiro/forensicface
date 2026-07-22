import hashlib
from pathlib import Path

import pytest

from forensicface.tools.migrate_shared import (
    ActionKind,
    apply_plan,
    build_plan,
    format_plan,
    main,
)


def _make_legacy_tree(
    root: Path,
    models: list[str],
    *,
    shared_content: bytes = b"shared-bytes",
    cr_fiqa_content: bytes = b"crfiqa-bytes",
    recognition_per_model: dict[str, tuple[str, str, bytes]] | None = None,
) -> None:
    """Builds a fake legacy ``~/.forensicface/models`` tree.

    ``recognition_per_model`` maps model name -> (subdir_name, filename, content).
    """
    if recognition_per_model is None:
        recognition_per_model = {}
    for model in models:
        model_dir = root / model
        model_dir.mkdir(parents=True)
        (model_dir / "det_10g.onnx").write_bytes(shared_content + b"-det")
        (model_dir / "1k3d68.onnx").write_bytes(shared_content + b"-1k3d68")
        (model_dir / "genderage.onnx").write_bytes(shared_content + b"-genderage")
        (model_dir / "cr_fiqa").mkdir()
        (model_dir / "cr_fiqa" / "cr_fiqa_l.onnx").write_bytes(cr_fiqa_content)

        if model in recognition_per_model:
            subdir_name, filename, content = recognition_per_model[model]
            sub = model_dir / subdir_name
            sub.mkdir()
            (sub / filename).write_bytes(content)


def test_dry_run_plan_for_typical_legacy_layout(tmp_path):
    _make_legacy_tree(
        tmp_path,
        models=["sepaelv2", "sepaelv3"],
        recognition_per_model={
            "sepaelv2": ("adaface", "adaface_ir101web12m.onnx", b"sepaelv2-rec"),
            "sepaelv3": ("adaface", "adaface_ir101ms1mv3.onnx", b"sepaelv3-rec"),
        },
    )

    plan = build_plan(tmp_path)

    moves = [a for a in plan.actions if a.kind == ActionKind.MOVE_SHARED]
    rec_moves = [a for a in plan.actions if a.kind == ActionKind.MOVE_RECOGNITION]
    deletes = [a for a in plan.actions if a.kind == ActionKind.DELETE_DUPLICATE]

    # 4 shared kinds moved once; 4 shared kinds deleted once from the
    # second model (its copies are duplicates of the first).
    assert len(moves) == 4
    assert len(deletes) == 4
    assert len(rec_moves) == 2
    assert not plan.has_conflicts
    assert plan.bytes_to_free > 0


def test_apply_moves_files_and_removes_duplicates(tmp_path):
    _make_legacy_tree(
        tmp_path,
        models=["sepaelv2", "sepaelv3"],
        recognition_per_model={
            "sepaelv2": ("adaface", "adaface_ir101web12m.onnx", b"sepaelv2-rec"),
            "sepaelv3": ("adaface", "adaface_ir101ms1mv3.onnx", b"sepaelv3-rec"),
        },
    )

    plan = build_plan(tmp_path)
    apply_plan(plan)

    assert (tmp_path / "detection" / "scrfd" / "det_10g.onnx").is_file()
    assert (
        tmp_path / "pose" / "insightface-3d68" / "1k3d68.onnx"
    ).is_file()
    assert (
        tmp_path
        / "attributes"
        / "insightface-genderage"
        / "genderage.onnx"
    ).is_file()
    assert (
        tmp_path / "quality" / "cr-fiqa" / "cr_fiqa_l.onnx"
    ).is_file()
    assert (
        tmp_path / "recognition" / "sepaelv2" / "adaface_ir101web12m.onnx"
    ).is_file()
    assert (
        tmp_path / "recognition" / "sepaelv3" / "adaface_ir101ms1mv3.onnx"
    ).is_file()

    # Originals gone.
    assert not (tmp_path / "sepaelv2" / "det_10g.onnx").exists()
    assert not (tmp_path / "sepaelv3" / "det_10g.onnx").exists()
    assert not (tmp_path / "sepaelv2" / "cr_fiqa" / "cr_fiqa_l.onnx").exists()


def test_apply_is_idempotent(tmp_path):
    _make_legacy_tree(
        tmp_path,
        models=["sepaelv2"],
        recognition_per_model={
            "sepaelv2": ("adaface", "adaface_ir101web12m.onnx", b"rec"),
        },
    )

    apply_plan(build_plan(tmp_path))
    second_plan = build_plan(tmp_path)
    assert not second_plan.has_work
    assert not second_plan.has_conflicts


def test_apply_migrates_flat_shared_layout_to_task_alias_directories(tmp_path):
    (tmp_path / "detection").mkdir()
    (tmp_path / "detection" / "det_10g.onnx").write_bytes(b"detector")
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes" / "1k3d68.onnx").write_bytes(b"pose")
    (tmp_path / "attributes" / "genderage.onnx").write_bytes(b"genderage")
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "cr_fiqa_l.onnx").write_bytes(b"quality")
    (tmp_path / "recognition" / "sepaelv2").mkdir(parents=True)
    recognition = tmp_path / "recognition" / "sepaelv2" / "ada_face.onnx"
    recognition.write_bytes(b"recognition")

    plan = build_plan(tmp_path)
    moves = [a for a in plan.actions if a.kind == ActionKind.MOVE_SHARED]
    assert len(moves) == 4
    assert not plan.has_conflicts

    apply_plan(plan)

    assert (tmp_path / "detection" / "scrfd" / "det_10g.onnx").is_file()
    assert (
        tmp_path / "pose" / "insightface-3d68" / "1k3d68.onnx"
    ).is_file()
    assert (
        tmp_path
        / "attributes"
        / "insightface-genderage"
        / "genderage.onnx"
    ).is_file()
    assert (
        tmp_path / "quality" / "cr-fiqa" / "cr_fiqa_l.onnx"
    ).is_file()
    assert recognition.is_file()
    assert not (tmp_path / "detection" / "det_10g.onnx").exists()
    assert not (tmp_path / "attributes" / "1k3d68.onnx").exists()
    assert not (tmp_path / "attributes" / "genderage.onnx").exists()
    assert not (tmp_path / "quality" / "cr_fiqa_l.onnx").exists()


def test_plan_records_conflict_on_hash_mismatch(tmp_path):
    (tmp_path / "detection").mkdir()
    (tmp_path / "detection" / "det_10g.onnx").write_bytes(b"new-version")

    (tmp_path / "sepaelv2").mkdir()
    (tmp_path / "sepaelv2" / "det_10g.onnx").write_bytes(b"old-version")
    (tmp_path / "sepaelv2" / "1k3d68.onnx").write_bytes(b"any")
    (tmp_path / "sepaelv2" / "genderage.onnx").write_bytes(b"any")
    (tmp_path / "sepaelv2" / "cr_fiqa").mkdir()
    (tmp_path / "sepaelv2" / "cr_fiqa" / "cr_fiqa_l.onnx").write_bytes(b"any")

    plan = build_plan(tmp_path)

    assert plan.has_conflicts
    assert any(
        c.legacy_path.name == "det_10g.onnx" for c in plan.conflicts
    )


def test_apply_refuses_to_run_with_conflicts(tmp_path):
    (tmp_path / "detection").mkdir()
    (tmp_path / "detection" / "det_10g.onnx").write_bytes(b"new")

    (tmp_path / "sepaelv2").mkdir()
    (tmp_path / "sepaelv2" / "det_10g.onnx").write_bytes(b"old")
    (tmp_path / "sepaelv2" / "1k3d68.onnx").write_bytes(b"x")
    (tmp_path / "sepaelv2" / "genderage.onnx").write_bytes(b"x")
    (tmp_path / "sepaelv2" / "cr_fiqa").mkdir()
    (tmp_path / "sepaelv2" / "cr_fiqa" / "cr_fiqa_l.onnx").write_bytes(b"x")

    plan = build_plan(tmp_path)
    with pytest.raises(RuntimeError, match="conflicts"):
        apply_plan(plan)


def test_plan_skips_when_shared_already_has_matching_hash(tmp_path):
    payload = b"identical-detector"
    (tmp_path / "detection").mkdir()
    (tmp_path / "detection" / "det_10g.onnx").write_bytes(payload)

    (tmp_path / "sepaelv2").mkdir()
    (tmp_path / "sepaelv2" / "det_10g.onnx").write_bytes(payload)
    (tmp_path / "sepaelv2" / "1k3d68.onnx").write_bytes(b"a")
    (tmp_path / "sepaelv2" / "genderage.onnx").write_bytes(b"b")
    (tmp_path / "sepaelv2" / "cr_fiqa").mkdir()
    (tmp_path / "sepaelv2" / "cr_fiqa" / "cr_fiqa_l.onnx").write_bytes(b"c")

    plan = build_plan(tmp_path)
    deletes = [a for a in plan.actions if a.kind == ActionKind.DELETE_DUPLICATE]

    assert any(
        d.src == tmp_path / "sepaelv2" / "det_10g.onnx" for d in deletes
    )
    moves = [a for a in plan.actions if a.kind == ActionKind.MOVE_SHARED]
    assert not any(
        m.dst == tmp_path / "detection" / "det_10g.onnx" for m in moves
    )


def test_plan_deletes_matching_flat_file_when_namespaced_target_exists(tmp_path):
    payload = b"identical-detector"
    target = tmp_path / "detection" / "scrfd" / "det_10g.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    flat_source = tmp_path / "detection" / "det_10g.onnx"
    flat_source.write_bytes(payload)

    plan = build_plan(tmp_path)

    assert not plan.has_conflicts
    assert any(
        action.kind == ActionKind.DELETE_DUPLICATE
        and action.src == flat_source
        for action in plan.actions
    )
    apply_plan(plan)
    assert target.read_bytes() == payload
    assert not flat_source.exists()


def test_plan_blocks_when_flat_file_differs_from_namespaced_target(tmp_path):
    target = tmp_path / "detection" / "scrfd" / "det_10g.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"target")
    flat_source = tmp_path / "detection" / "det_10g.onnx"
    flat_source.write_bytes(b"flat")

    plan = build_plan(tmp_path)

    assert plan.has_conflicts
    assert plan.conflicts[0].legacy_path == flat_source
    assert plan.conflicts[0].shared_path == target


def test_plan_removes_empty_subdirectories_after_migration(tmp_path):
    _make_legacy_tree(
        tmp_path,
        models=["sepaelv2"],
        recognition_per_model={
            "sepaelv2": ("adaface", "adaface_ir101web12m.onnx", b"rec"),
        },
    )

    plan = build_plan(tmp_path)
    rmdir_targets = {
        a.src for a in plan.actions if a.kind == ActionKind.REMOVE_EMPTY_DIR
    }
    assert (tmp_path / "sepaelv2" / "cr_fiqa") in rmdir_targets
    assert (tmp_path / "sepaelv2" / "adaface") in rmdir_targets

    apply_plan(plan)
    assert not (tmp_path / "sepaelv2" / "cr_fiqa").exists()
    assert not (tmp_path / "sepaelv2" / "adaface").exists()


def test_plan_raises_on_multiple_recognition_files(tmp_path):
    model_dir = tmp_path / "sepaelv2"
    model_dir.mkdir()
    (model_dir / "det_10g.onnx").write_bytes(b"d")
    (model_dir / "1k3d68.onnx").write_bytes(b"k")
    (model_dir / "genderage.onnx").write_bytes(b"g")
    (model_dir / "cr_fiqa").mkdir()
    (model_dir / "cr_fiqa" / "cr_fiqa_l.onnx").write_bytes(b"f")
    sub_a = model_dir / "adaface"
    sub_a.mkdir()
    (sub_a / "first_face.onnx").write_bytes(b"r1")
    sub_b = model_dir / "other"
    sub_b.mkdir()
    (sub_b / "second_face.onnx").write_bytes(b"r2")

    plan = build_plan(tmp_path)
    assert plan.has_conflicts
    assert any("Multiple recognition" in c.shared_hash for c in plan.conflicts)


def test_main_dry_run_returns_zero(tmp_path, capsys):
    _make_legacy_tree(
        tmp_path,
        models=["sepaelv2"],
        recognition_per_model={
            "sepaelv2": ("adaface", "f_face.onnx", b"rec"),
        },
    )
    rc = main(["--models-root", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Move shared files" in out
    assert "Dry-run only" in out


def test_main_apply_with_yes(tmp_path, capsys):
    _make_legacy_tree(
        tmp_path,
        models=["sepaelv2"],
        recognition_per_model={
            "sepaelv2": ("adaface", "f_face.onnx", b"rec"),
        },
    )
    rc = main(["--models-root", str(tmp_path), "--apply", "--yes"])
    assert rc == 0
    assert (tmp_path / "detection" / "scrfd" / "det_10g.onnx").is_file()


def test_main_apply_returns_non_zero_on_conflicts(tmp_path):
    (tmp_path / "detection").mkdir()
    (tmp_path / "detection" / "det_10g.onnx").write_bytes(b"new")
    (tmp_path / "sepaelv2").mkdir()
    (tmp_path / "sepaelv2" / "det_10g.onnx").write_bytes(b"old")
    (tmp_path / "sepaelv2" / "1k3d68.onnx").write_bytes(b"x")
    (tmp_path / "sepaelv2" / "genderage.onnx").write_bytes(b"x")
    (tmp_path / "sepaelv2" / "cr_fiqa").mkdir()
    (tmp_path / "sepaelv2" / "cr_fiqa" / "cr_fiqa_l.onnx").write_bytes(b"x")

    rc = main(["--models-root", str(tmp_path), "--apply", "--yes"])
    assert rc == 3
    # File should remain untouched on the legacy side.
    assert (tmp_path / "sepaelv2" / "det_10g.onnx").read_bytes() == b"old"


def test_main_missing_models_root_returns_non_zero(tmp_path):
    rc = main(["--models-root", str(tmp_path / "does-not-exist")])
    assert rc == 2


def test_format_plan_includes_freed_space(tmp_path):
    _make_legacy_tree(
        tmp_path,
        models=["sepaelv2", "sepaelv3"],
        recognition_per_model={
            "sepaelv2": ("adaface", "a_face.onnx", b"r1"),
            "sepaelv3": ("adaface", "b_face.onnx", b"r2"),
        },
    )
    plan = build_plan(tmp_path)
    report = format_plan(plan, models_root=tmp_path)
    assert "Estimated space to free" in report
    assert "Delete duplicates" in report
