"""Migrate a ``~/.forensicface/models`` tree from the legacy per-model
layout to the new shared layout.

Legacy layout (each model carries identical copies of shared files):

    models_root/
      <model_name>/
        det_10g.onnx
        1k3d68.onnx
        genderage.onnx
        cr_fiqa/cr_fiqa_l.onnx
        <recognition_subdir>/<*face*>.onnx

New layout (shared files live once, recognition stays per-model):

    models_root/
      detection/det_10g.onnx
      attributes/1k3d68.onnx
      attributes/genderage.onnx
      quality/cr_fiqa_l.onnx
      recognition/<model_name>/<*face*>.onnx

Usage::

    python -m forensicface.tools.migrate_shared              # dry-run
    python -m forensicface.tools.migrate_shared --apply      # do it
    python -m forensicface.tools.migrate_shared --models-root /custom/path

The default is *dry-run* — nothing is moved or deleted until ``--apply``
is passed. Deletions only happen when SHA-256 hashes match the file
already saved in the shared location; mismatches abort the migration
with a clear report so the user can investigate.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


SHARED_FILE_KINDS: dict[str, tuple[str, str]] = {
    # legacy_relative_path -> (shared_subdir, basename_in_shared)
    "det_10g.onnx": ("detection", "det_10g.onnx"),
    "1k3d68.onnx": ("attributes", "1k3d68.onnx"),
    "genderage.onnx": ("attributes", "genderage.onnx"),
    "cr_fiqa/cr_fiqa_l.onnx": ("quality", "cr_fiqa_l.onnx"),
}


class ActionKind(str, Enum):
    MOVE_SHARED = "move_shared"
    DELETE_DUPLICATE = "delete_duplicate"
    MOVE_RECOGNITION = "move_recognition"
    SKIP_ALREADY_MIGRATED = "skip_already_migrated"
    REMOVE_EMPTY_DIR = "remove_empty_dir"


@dataclass
class Action:
    kind: ActionKind
    src: Path
    dst: Optional[Path] = None
    size: int = 0
    note: str = ""


@dataclass
class Conflict:
    legacy_path: Path
    shared_path: Path
    legacy_hash: str
    shared_hash: str

    def message(self) -> str:
        return (
            f"hash mismatch for {self.legacy_path}: legacy={self.legacy_hash[:12]}… "
            f"vs shared={self.shared_hash[:12]}… ({self.shared_path})"
        )


@dataclass
class MigrationPlan:
    actions: list[Action] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def bytes_to_free(self) -> int:
        return sum(a.size for a in self.actions if a.kind == ActionKind.DELETE_DUPLICATE)

    @property
    def has_work(self) -> bool:
        return any(
            a.kind
            in (
                ActionKind.MOVE_SHARED,
                ActionKind.DELETE_DUPLICATE,
                ActionKind.MOVE_RECOGNITION,
                ActionKind.REMOVE_EMPTY_DIR,
            )
            for a in self.actions
        )

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)


def _sha256(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_chunk), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_legacy_model_dirs(models_root: Path) -> list[Path]:
    """Returns subdirectories of ``models_root`` that look like legacy model
    folders (i.e. not the reserved shared names)."""
    reserved = {"detection", "attributes", "quality", "recognition"}
    if not models_root.is_dir():
        return []
    return sorted(
        p
        for p in models_root.iterdir()
        if p.is_dir() and p.name not in reserved and not p.name.startswith(".")
    )


def _find_recognition_file(model_dir: Path) -> Optional[Path]:
    """Finds the ``*face*.onnx`` recognition file inside any subfolder of
    ``model_dir``. Returns None if not found, raises if multiple matches.
    """
    matches: list[Path] = []
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir() or child.name == "cr_fiqa":
            continue
        for f in sorted(child.glob("*face*.onnx")):
            matches.append(f)
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple recognition models found under {model_dir}: {matches}"
        )
    return matches[0] if matches else None


def build_plan(models_root: Path) -> MigrationPlan:
    """Plans the migration. Pure function: reads disk + hashes, writes
    nothing. Safe to call from dry-run mode."""
    plan = MigrationPlan()
    model_dirs = _discover_legacy_model_dirs(models_root)

    canonical_shared_hash: dict[str, str] = {}

    for model_dir in model_dirs:
        for legacy_rel, (shared_subdir, shared_name) in SHARED_FILE_KINDS.items():
            legacy_path = model_dir / Path(legacy_rel)
            if not legacy_path.is_file():
                continue
            shared_path = models_root / shared_subdir / shared_name
            shared_key = str(shared_path)
            legacy_hash = _sha256(legacy_path)

            if shared_key not in canonical_shared_hash and shared_path.is_file():
                canonical_shared_hash[shared_key] = _sha256(shared_path)

            if shared_key in canonical_shared_hash:
                shared_hash = canonical_shared_hash[shared_key]
                if legacy_hash == shared_hash:
                    plan.actions.append(
                        Action(
                            kind=ActionKind.DELETE_DUPLICATE,
                            src=legacy_path,
                            size=legacy_path.stat().st_size,
                            note=f"duplicate of {shared_path}",
                        )
                    )
                else:
                    plan.conflicts.append(
                        Conflict(
                            legacy_path=legacy_path,
                            shared_path=shared_path,
                            legacy_hash=legacy_hash,
                            shared_hash=shared_hash,
                        )
                    )
            else:
                plan.actions.append(
                    Action(
                        kind=ActionKind.MOVE_SHARED,
                        src=legacy_path,
                        dst=shared_path,
                        size=legacy_path.stat().st_size,
                    )
                )
                canonical_shared_hash[shared_key] = legacy_hash

        try:
            rec_file = _find_recognition_file(model_dir)
        except RuntimeError as exc:
            plan.conflicts.append(
                Conflict(
                    legacy_path=model_dir,
                    shared_path=models_root / "recognition" / model_dir.name,
                    legacy_hash="-",
                    shared_hash=str(exc),
                )
            )
            rec_file = None

        if rec_file is not None:
            rec_target = models_root / "recognition" / model_dir.name / rec_file.name
            if rec_target.is_file():
                target_hash = _sha256(rec_target)
                src_hash = _sha256(rec_file)
                if target_hash == src_hash:
                    plan.actions.append(
                        Action(
                            kind=ActionKind.DELETE_DUPLICATE,
                            src=rec_file,
                            size=rec_file.stat().st_size,
                            note=f"already present at {rec_target}",
                        )
                    )
                else:
                    plan.conflicts.append(
                        Conflict(
                            legacy_path=rec_file,
                            shared_path=rec_target,
                            legacy_hash=src_hash,
                            shared_hash=target_hash,
                        )
                    )
            else:
                plan.actions.append(
                    Action(
                        kind=ActionKind.MOVE_RECOGNITION,
                        src=rec_file,
                        dst=rec_target,
                        size=rec_file.stat().st_size,
                    )
                )

    plan.actions.extend(_plan_empty_dir_cleanups(models_root, plan))
    return plan


def _plan_empty_dir_cleanups(
    models_root: Path, plan: MigrationPlan
) -> list[Action]:
    """Predicts which subfolders inside legacy model dirs become empty
    after the planned actions, so they can be removed. Read-only — does
    not actually delete anything."""
    moved_or_deleted: set[Path] = {a.src for a in plan.actions}
    cleanups: list[Action] = []
    for model_dir in _discover_legacy_model_dirs(models_root):
        for child in sorted(model_dir.iterdir()):
            if not child.is_dir():
                continue
            remaining = [
                f for f in child.rglob("*") if f.is_file() and f not in moved_or_deleted
            ]
            if not remaining:
                cleanups.append(
                    Action(
                        kind=ActionKind.REMOVE_EMPTY_DIR,
                        src=child,
                        note="empty after migration",
                    )
                )
    return cleanups


def apply_plan(plan: MigrationPlan) -> None:
    """Executes the plan. Caller is expected to have inspected conflicts
    first. Move actions create destination directories as needed."""
    if plan.has_conflicts:
        raise RuntimeError(
            "Refusing to apply plan with unresolved conflicts. "
            "Inspect the dry-run output and resolve them manually."
        )
    for action in plan.actions:
        if action.kind in (ActionKind.MOVE_SHARED, ActionKind.MOVE_RECOGNITION):
            if action.dst is None:
                raise RuntimeError(
                    f"Invalid migration action: {action.kind.value} requires "
                    f"a destination path for source {action.src}"
                )
            action.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(action.src), str(action.dst))
        elif action.kind == ActionKind.DELETE_DUPLICATE:
            action.src.unlink()
        elif action.kind == ActionKind.REMOVE_EMPTY_DIR:
            try:
                action.src.rmdir()
            except OSError:
                pass


def format_plan(plan: MigrationPlan, *, models_root: Path) -> str:
    lines: list[str] = []
    lines.append(f"Models root: {models_root}")
    lines.append("")

    moves = [a for a in plan.actions if a.kind == ActionKind.MOVE_SHARED]
    rec_moves = [a for a in plan.actions if a.kind == ActionKind.MOVE_RECOGNITION]
    deletes = [a for a in plan.actions if a.kind == ActionKind.DELETE_DUPLICATE]
    rmdirs = [a for a in plan.actions if a.kind == ActionKind.REMOVE_EMPTY_DIR]

    if moves:
        lines.append(f"Move shared files ({len(moves)}):")
        for a in moves:
            lines.append(f"  {a.src}  ->  {a.dst}")
        lines.append("")

    if rec_moves:
        lines.append(f"Move recognition files ({len(rec_moves)}):")
        for a in rec_moves:
            lines.append(f"  {a.src}  ->  {a.dst}")
        lines.append("")

    if deletes:
        lines.append(f"Delete duplicates ({len(deletes)}):")
        for a in deletes:
            lines.append(f"  {a.src}  ({a.note})")
        lines.append("")

    if rmdirs:
        lines.append(f"Remove empty directories ({len(rmdirs)}):")
        for a in rmdirs:
            lines.append(f"  {a.src}")
        lines.append("")

    if plan.conflicts:
        lines.append(f"Conflicts ({len(plan.conflicts)}) — migration BLOCKED:")
        for c in plan.conflicts:
            lines.append(f"  {c.message()}")
        lines.append("")

    if not plan.has_work and not plan.has_conflicts:
        lines.append("Nothing to do — layout already migrated.")
    else:
        freed_mb = plan.bytes_to_free / (1024 * 1024)
        lines.append(f"Estimated space to free: {freed_mb:,.1f} MB")
    return "\n".join(lines)


def _confirm(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes", "s", "sim"}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forensicface.tools.migrate_shared",
        description=(
            "Migrate ~/.forensicface/models from the legacy per-model "
            "layout to the new shared layout."
        ),
    )
    parser.add_argument(
        "--models-root",
        default=os.path.join(os.path.expanduser("~"), ".forensicface", "models"),
        help="Models root directory (default: ~/.forensicface/models).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the migration. Without this flag the tool only prints a dry-run plan.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt when used with --apply.",
    )
    args = parser.parse_args(argv)

    models_root = Path(args.models_root).expanduser()
    if not models_root.is_dir():
        print(f"Models root does not exist: {models_root}", file=sys.stderr)
        return 2

    plan = build_plan(models_root)
    print(format_plan(plan, models_root=models_root))

    if plan.has_conflicts:
        return 3

    if not args.apply:
        if plan.has_work:
            print("\nDry-run only. Re-run with --apply to perform the migration.")
        return 0

    if not plan.has_work:
        return 0

    if not args.yes and not _confirm("\nProceed? [y/N]: "):
        print("Aborted.")
        return 1

    apply_plan(plan)
    print("\nMigration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
