#!/usr/bin/env python3
"""Reproduce ETH3D DSLR relative pose AUC for InstantSfM and GLOMAP.

This script:
1) Optionally runs `colmap global_mapper` per scene using existing
   `database.db` and `images/` (no feature extraction/matching).
2) Evaluates relative pose AUC with the same metric used by
   `instantsfm/eval/colmap_eval/evaluation/utils.py`.
3) Writes per-method CSVs and a side-by-side comparison CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pycolmap
except ImportError as exc:
    raise SystemExit(
        "pycolmap is required. Install with: pip install pycolmap==3.10.0"
    ) from exc

from instantsfm.eval.colmap_eval.evaluation.utils import compute_auc, compute_rel_errors
from instantsfm.utils.read_write_model import read_cameras_text, read_images_text


# Compatibility aliases for environments where NumPy misses these names.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]
if not hasattr(np, "acos"):
    np.acos = np.arccos  # type: ignore[attr-defined]


@dataclass
class SceneEval:
    scene: str
    aucs: np.ndarray
    num_reg_images: int
    num_images: int
    errors: np.ndarray


class _MergedReconstruction:
    """Minimal wrapper that satisfies compute_rel_errors() API needs."""

    def __init__(self, images: dict[int, pycolmap.Image]):
        self.images = images

    def num_images(self) -> int:
        return len(self.images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dslr_root",
        type=Path,
        default=Path("dataset/eth3d/dslr"),
        help="Path to ETH3D DSLR scenes directory.",
    )
    parser.add_argument(
        "--scene_source_csv",
        type=Path,
        default=Path("dataset/instantsfm_eth3d_dslr_partial_auc.csv"),
        help=(
            "If present and --scenes is empty, use scene names where scope=scene "
            "from this CSV (for fair subset matching)."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[],
        help="Explicit scene names. Overrides --scene_source_csv.",
    )
    parser.add_argument(
        "--run_glomap",
        action="store_true",
        help="Run colmap global_mapper before evaluation.",
    )
    parser.add_argument(
        "--run_instantsfm",
        action="store_true",
        help="Run InstantSfM in isolated workspaces before evaluation.",
    )
    parser.add_argument(
        "--glomap_workers",
        type=int,
        default=3,
        help="Parallel scene count for global_mapper runs.",
    )
    parser.add_argument(
        "--global_mapper_num_threads",
        type=int,
        default=8,
        help="--GlobalMapper.num_threads value per scene.",
    )
    parser.add_argument(
        "--instantsfm_workers",
        type=int,
        default=1,
        help="Parallel scene count for InstantSfM runs.",
    )
    parser.add_argument(
        "--colmap_bin",
        type=str,
        default="colmap",
        help="COLMAP executable path.",
    )
    parser.add_argument(
        "--force_rerun_glomap",
        action="store_true",
        help="Delete existing sparse_glomap/ and rerun mapping.",
    )
    parser.add_argument(
        "--force_rerun_instantsfm",
        action="store_true",
        help="Delete existing InstantSfM variant workspaces and rerun mapping.",
    )
    parser.add_argument(
        "--instantsfm_sparse_name",
        type=str,
        default="sparse",
        help="Folder name for InstantSfM reconstructions under each scene.",
    )
    parser.add_argument(
        "--glomap_sparse_name",
        type=str,
        default="sparse_glomap",
        help="Folder name for GLOMAP reconstructions under each scene.",
    )
    parser.add_argument(
        "--instantsfm_workspace_root",
        type=Path,
        default=None,
        help=(
            "Workspace root for generated InstantSfM variants. Defaults to "
            "<output_dir>/<prefix>_instantsfm_workspaces."
        ),
    )
    parser.add_argument(
        "--instantsfm_pair_correspondence_source",
        choices=["matches", "two_view_geometries"],
        default="matches",
        help="Pair correspondence source passed to InstantSfM.",
    )
    parser.add_argument(
        "--instantsfm_skip_view_graph_calibration",
        action="store_true",
        help="Pass --skip_view_graph_calibration to InstantSfM.",
    )
    parser.add_argument(
        "--instantsfm_enable_retriangulation",
        action="store_true",
        help="Pass --enable_retriangulation to InstantSfM.",
    )
    parser.add_argument(
        "--instantsfm_seed",
        type=int,
        default=None,
        help="Pass --seed to InstantSfM.",
    )
    parser.add_argument(
        "--instantsfm_visible_devices",
        type=str,
        default="",
        help=(
            "Comma-separated CUDA device indices to round-robin across InstantSfM "
            "workers. Defaults to auto-discovery via nvidia-smi."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset"),
        help="Directory for output CSV reports.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="eth3d_dslr_pose_auc",
        help="Filename prefix for output reports.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0, 10.0],
        help="Relative pose AUC thresholds (degrees).",
    )
    parser.add_argument(
        "--position_accuracy_gt",
        type=float,
        default=0.001,
        help="ETH3D GT position accuracy in meters.",
    )
    parser.add_argument(
        "--inject_gt_intrinsics",
        action="store_true",
        help=(
            "Inject GT intrinsics from *_calibration_undistorted into database.db "
            "before reconstruction/evaluation, while preserving features/matches."
        ),
    )
    parser.add_argument(
        "--force_reinject_gt_intrinsics",
        action="store_true",
        help="Force rebuilding injected database.db from database_orig.db if available.",
    )
    return parser.parse_args()


def discover_scenes(args: argparse.Namespace) -> list[str]:
    if args.scenes:
        return args.scenes

    if args.scene_source_csv.exists():
        scenes: list[str] = []
        with args.scene_source_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("scope") == "scene":
                    scenes.append(row["scene"])
        if scenes:
            return scenes

    return sorted(p.name for p in args.dslr_root.iterdir() if p.is_dir())


def find_gt_sparse_path(scene_dir: Path) -> Path:
    cands = sorted(scene_dir.glob("*_calibration_undistorted"))
    if not cands:
        raise FileNotFoundError(f"No *_calibration_undistorted found in {scene_dir}")
    return cands[0]


def map_camera_model(model_name: str) -> int:
    mapping = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
        "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8,
        "RADIAL_FISHEYE": 9,
        "THIN_PRISM_FISHEYE": 10,
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown camera model: {model_name}")
    return mapping[model_name]


def inject_gt_intrinsics_for_scene(
    scene_dir: Path,
    force_reinject: bool,
) -> tuple[str, bool, str]:
    scene = scene_dir.name
    database_path = scene_dir / "database.db"
    if not database_path.exists():
        return scene, False, f"missing database: {database_path}"

    gt_sparse_path = find_gt_sparse_path(scene_dir)
    gt_images_path = gt_sparse_path / "images.txt"
    gt_cameras_path = gt_sparse_path / "cameras.txt"
    if not gt_images_path.exists() or not gt_cameras_path.exists():
        return scene, False, f"missing GT cameras/images under {gt_sparse_path}"

    backup_db_path = scene_dir / "database_orig.db"
    if force_reinject and backup_db_path.exists():
        backup_db_path.unlink()
    if not backup_db_path.exists():
        shutil.copy2(database_path, backup_db_path)

    source_db_path = backup_db_path if backup_db_path.exists() else database_path

    injected_db_path = scene_dir / "database_gtintr.db"
    if injected_db_path.exists():
        injected_db_path.unlink()
    shutil.copy2(source_db_path, injected_db_path)

    images_gt = read_images_text(str(gt_images_path))
    cameras_gt = read_cameras_text(str(gt_cameras_path))

    conn = sqlite3.connect(str(injected_db_path))
    cur = conn.cursor()

    image_rows = list(cur.execute("SELECT image_id, name, camera_id FROM images"))
    image_name_to_row: dict[str, tuple[int, int]] = {}
    basename_to_rows: dict[str, list[tuple[int, int]]] = {}
    for image_id, name, camera_id in image_rows:
        image_name_to_row[name] = (image_id, camera_id)
        basename_to_rows.setdefault(Path(name).name, []).append((image_id, camera_id))

    existing_camera_ids = {row[0] for row in cur.execute("SELECT camera_id FROM cameras")}
    camera_updates: dict[int, tuple[int, int, int, bytes]] = {}

    for image in images_gt.values():
        gt_name = image.name
        if gt_name in image_name_to_row:
            image_id, existing_camera_id = image_name_to_row[gt_name]
        else:
            cands = basename_to_rows.get(Path(gt_name).name, [])
            if len(cands) != 1:
                conn.close()
                return scene, False, f"cannot map GT image to DB row: {gt_name}"
            image_id, existing_camera_id = cands[0]

        if image.camera_id not in cameras_gt:
            conn.close()
            return scene, False, f"GT camera id {image.camera_id} missing for image {gt_name}"
        gt_cam = cameras_gt[image.camera_id]
        camera_updates[existing_camera_id] = (
            map_camera_model(gt_cam.model),
            gt_cam.width,
            gt_cam.height,
            np.asarray(gt_cam.params, dtype=np.float64).tobytes(),
        )

        # Keep camera_id unchanged for modern COLMAP frame/frame_data consistency.
        cur.execute("UPDATE images SET name = ? WHERE image_id = ?", (gt_name, image_id))

    for camera_id, (model, width, height, params_blob) in camera_updates.items():
        if camera_id not in existing_camera_ids:
            conn.close()
            return scene, False, f"camera_id {camera_id} not found in DB"
        cur.execute(
            """
            UPDATE cameras
            SET model = ?, width = ?, height = ?, params = ?, prior_focal_length = 1
            WHERE camera_id = ?
            """,
            (model, width, height, params_blob, camera_id),
        )

    # Keep two-view info but avoid NULL epipolar blobs that break some InstantSfM builds.
    zero_mat = np.zeros((3, 3), dtype=np.float64).tobytes()
    cur.execute("UPDATE two_view_geometries SET E = ? WHERE E IS NULL", (zero_mat,))
    cur.execute("UPDATE two_view_geometries SET F = ? WHERE F IS NULL", (zero_mat,))
    cur.execute("UPDATE two_view_geometries SET H = ? WHERE H IS NULL", (zero_mat,))

    conn.commit()
    conn.close()

    os.replace(injected_db_path, database_path)
    return scene, True, f"injected_from={source_db_path.name}"


def inject_gt_intrinsics_batch(
    args: argparse.Namespace,
    scenes: Iterable[str],
) -> list[tuple[str, bool, str]]:
    records: list[tuple[str, bool, str]] = []
    for scene in scenes:
        records.append(
            inject_gt_intrinsics_for_scene(
                args.dslr_root / scene,
                force_reinject=args.force_reinject_gt_intrinsics,
            )
        )
    records.sort(key=lambda x: x[0])
    return records


def run_glomap_for_scene(
    scene_dir: Path,
    colmap_bin: str,
    sparse_name: str,
    mapper_threads: int,
    force_rerun: bool,
) -> tuple[str, int, bool, str]:
    scene = scene_dir.name
    database_path = scene_dir / "database.db"
    image_path = scene_dir / "images"
    output_path = scene_dir / sparse_name
    log_path = scene_dir / "glomap.log"

    if not database_path.exists():
        return scene, 1, False, f"missing database: {database_path}"
    if not image_path.exists():
        return scene, 1, False, f"missing images: {image_path}"

    existing_model = output_path / "0" / "images.bin"
    if existing_model.exists() and not force_rerun:
        return scene, 0, True, "skipped (already exists)"

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        colmap_bin,
        "global_mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--GlobalMapper.num_threads",
        str(mapper_threads),
        "--GlobalMapper.ba_refine_focal_length",
        "0",
        "--GlobalMapper.ba_refine_extra_params",
        "0",
    ]
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    model_ok = (output_path / "0" / "images.bin").exists()
    return scene, proc.returncode, model_ok, str(log_path)


def run_glomap_batch(args: argparse.Namespace, scenes: Iterable[str]) -> list[tuple[str, int, bool, str]]:
    scene_dirs = [args.dslr_root / s for s in scenes]
    records: list[tuple[str, int, bool, str]] = []

    with ThreadPoolExecutor(max_workers=args.glomap_workers) as executor:
        futures = [
            executor.submit(
                run_glomap_for_scene,
                scene_dir,
                args.colmap_bin,
                args.glomap_sparse_name,
                args.global_mapper_num_threads,
                args.force_rerun_glomap,
            )
            for scene_dir in scene_dirs
        ]
        for fut in as_completed(futures):
            records.append(fut.result())
    records.sort(key=lambda x: x[0])
    return records


def prepare_instantsfm_workspace(scene_dir: Path, workspace_dir: Path, force_rerun: bool) -> None:
    if force_rerun and workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    database_src = scene_dir / "database.db"
    database_dst = workspace_dir / "database.db"
    if force_rerun or not database_dst.exists():
        shutil.copy2(database_src, database_dst)
        zero_mat = np.zeros((3, 3), dtype=np.float64).tobytes()
        with sqlite3.connect(database_dst) as conn:
            conn.execute("UPDATE two_view_geometries SET E = ? WHERE E IS NULL", (zero_mat,))
            conn.execute("UPDATE two_view_geometries SET F = ? WHERE F IS NULL", (zero_mat,))
            conn.execute("UPDATE two_view_geometries SET H = ? WHERE H IS NULL", (zero_mat,))
            conn.commit()

    images_src = scene_dir / "images"
    images_dst = workspace_dir / "images"
    if images_dst.exists() or images_dst.is_symlink():
        if force_rerun:
            if images_dst.is_symlink() or images_dst.is_file():
                images_dst.unlink()
            else:
                shutil.rmtree(images_dst)
    if not images_dst.exists():
        images_dst.symlink_to(images_src.resolve(), target_is_directory=True)


def discover_cuda_devices(visible_devices_arg: str) -> list[str]:
    if visible_devices_arg.strip():
        return [d.strip() for d in visible_devices_arg.split(",") if d.strip()]

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        return [d.strip() for d in cuda_visible.split(",") if d.strip()]

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def run_instantsfm_for_scene(
    scene_dir: Path,
    workspace_root: Path,
    sparse_name: str,
    pair_correspondence_source: str,
    skip_view_graph_calibration: bool,
    enable_retriangulation: bool,
    seed: int | None,
    visible_device: str | None,
    force_rerun: bool,
) -> tuple[str, int, bool, str]:
    scene = scene_dir.name
    workspace_dir = workspace_root / scene
    log_path = workspace_dir / "instantsfm.log"
    output_path = workspace_dir / sparse_name

    if not (scene_dir / "database.db").exists():
        return scene, 1, False, f"missing database: {scene_dir / 'database.db'}"
    if not (scene_dir / "images").exists():
        return scene, 1, False, f"missing images: {scene_dir / 'images'}"

    existing_model = output_path / "0" / "images.bin"
    if existing_model.exists() and not force_rerun:
        return scene, 0, True, "skipped (already exists)"

    prepare_instantsfm_workspace(scene_dir, workspace_dir, force_rerun)

    runner = r"""
import sys
import time
from pathlib import Path

from instantsfm.controllers.config import Config
from instantsfm.controllers.data_reader import ReadColmapDatabase, ReadData, ReadDepthsIntoFeatures, ReadSemanticsIntoFeatures
from instantsfm.controllers.global_mapper import SolveGlobalMapper
from instantsfm.controllers.reconstruction_writer import WriteGlomapReconstruction
from instantsfm.processors.bundle_adjustment import TorchBA
import inspect

data_path = Path(sys.argv[1])
pair_correspondence_source = sys.argv[2]
skip_view_graph_calibration = sys.argv[3] == "1"
enable_retriangulation = sys.argv[4] == "1"
seed_arg = sys.argv[5]
sparse_name = sys.argv[6]
seed = None if seed_arg == "none" else int(seed_arg)

path_info = ReadData(str(data_path))
path_info.output_path = str(data_path / sparse_name)
if pair_correspondence_source != "matches":
    raise ValueError(f"Unsupported pair_correspondence_source for current InstantSfM tree: {pair_correspondence_source}")
view_graph, cameras, images, feature_name, multi_camera_rig = ReadColmapDatabase(
    path_info.database_path,
)

start_time = time.time()
config = Config(feature_name)
config.RUNTIME_OPTIONS["multi_camera_rig"] = multi_camera_rig
config.RUNTIME_OPTIONS["random_seed"] = seed
if skip_view_graph_calibration:
    config.OPTIONS["skip_view_graph_calibration"] = True
if enable_retriangulation:
    config.OPTIONS["skip_retriangulation"] = False

if multi_camera_rig and path_info.fixed_relative_poses_path:
    config.RUNTIME_OPTIONS["use_fixed_rel_poses"] = True
if path_info.depth_path:
    config.RUNTIME_OPTIONS["use_depths"] = True
    ReadDepthsIntoFeatures(path_info.depth_path, cameras, images)
if path_info.semantics_path:
    config.RUNTIME_OPTIONS["use_semantic_filtering"] = True
    ReadSemanticsIntoFeatures(path_info.semantics_path, cameras, images)

if "fix_rotation" not in inspect.signature(TorchBA.Solve).parameters:
    _orig_solve = TorchBA.Solve
    def _compat_solve(self, cameras, images, tracks, options, *args, **kwargs):
        kwargs.pop("fix_rotation", None)
        return _orig_solve(self, cameras, images, tracks, options, *args, **kwargs)
    TorchBA.Solve = _compat_solve

cameras, images, tracks = SolveGlobalMapper(view_graph, cameras, images, config)
print("Reconstruction done in", time.time() - start_time, "seconds")
WriteGlomapReconstruction(path_info.output_path, cameras, images, tracks, path_info.image_path)
print("Reconstruction written to", path_info.output_path)
"""
    cmd = [
        sys.executable,
        "-c",
        runner,
        str(workspace_dir),
        pair_correspondence_source,
        "1" if skip_view_graph_calibration else "0",
        "1" if enable_retriangulation else "0",
        "none" if seed is None else str(seed),
        sparse_name,
    ]

    env = os.environ.copy()
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_device
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False, env=env)

    model_ok = (output_path / "0" / "images.bin").exists()
    return scene, proc.returncode, model_ok, str(log_path)


def run_instantsfm_batch(
    args: argparse.Namespace,
    scenes: Iterable[str],
) -> tuple[Path, list[tuple[str, int, bool, str]]]:
    workspace_root = args.instantsfm_workspace_root
    if workspace_root is None:
        workspace_root = args.output_dir / f"{args.prefix}_instantsfm_workspaces"

    scene_dirs = [args.dslr_root / s for s in scenes]
    visible_devices = discover_cuda_devices(args.instantsfm_visible_devices)
    records: list[tuple[str, int, bool, str]] = []
    with ThreadPoolExecutor(max_workers=args.instantsfm_workers) as executor:
        futures = [
            executor.submit(
                run_instantsfm_for_scene,
                scene_dir,
                workspace_root,
                args.instantsfm_sparse_name,
                args.instantsfm_pair_correspondence_source,
                args.instantsfm_skip_view_graph_calibration,
                args.instantsfm_enable_retriangulation,
                args.instantsfm_seed,
                visible_devices[idx % len(visible_devices)] if visible_devices else None,
                args.force_rerun_instantsfm,
            )
            for idx, scene_dir in enumerate(scene_dirs)
        ]
        for fut in as_completed(futures):
            records.append(fut.result())
    records.sort(key=lambda x: x[0])
    return workspace_root, records


def merge_sparse_components(sparse_dir: Path) -> tuple[_MergedReconstruction | None, int]:
    if not sparse_dir.is_dir():
        return None, 0

    merged_by_name: dict[str, pycolmap.Image] = {}
    num_components = 0
    components: list[pycolmap.Reconstruction] = []

    for subdir in sorted(sparse_dir.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            recon = pycolmap.Reconstruction(subdir)
        except Exception:
            continue
        components.append(recon)  # keep alive while image objects are used
        num_components += 1
        for image in recon.images.values():
            key = image.name.split("/")[-1]
            if key not in merged_by_name:
                merged_by_name[key] = image

    if not merged_by_name:
        return None, num_components

    merged_images = {idx + 1: img for idx, img in enumerate(merged_by_name.values())}
    return _MergedReconstruction(merged_images), num_components


def evaluate_method(
    dslr_root: Path,
    recon_root: Path,
    scenes: Iterable[str],
    sparse_name: str,
    thresholds: np.ndarray,
    min_error: float,
) -> tuple[list[SceneEval], np.ndarray, np.ndarray]:
    rows: list[SceneEval] = []
    all_errors: list[np.ndarray] = []

    for scene in scenes:
        scene_dir = dslr_root / scene
        gt_path = find_gt_sparse_path(scene_dir)
        sparse_gt = pycolmap.Reconstruction(gt_path)

        sparse_eval, _ = merge_sparse_components((recon_root / scene) / sparse_name)
        dts, dRs = compute_rel_errors(
            sparse_gt=sparse_gt,
            sparse=sparse_eval,
            min_proj_center_dist=min_error,
        )
        errors = np.maximum(dts, dRs)
        aucs = compute_auc(errors, thresholds, min_error=min_error)

        num_reg_images = 0 if sparse_eval is None else sparse_eval.num_images()
        rows.append(
            SceneEval(
                scene=scene,
                aucs=aucs,
                num_reg_images=num_reg_images,
                num_images=sparse_gt.num_images(),
                errors=errors,
            )
        )
        all_errors.append(errors)

    summary_all = compute_auc(np.concatenate(all_errors), thresholds, min_error=min_error)
    summary_avg = np.mean(np.stack([row.aucs for row in rows], axis=0), axis=0)
    return rows, summary_all, summary_avg


def write_method_csv(
    out_path: Path,
    rows: list[SceneEval],
    summary_all: np.ndarray,
    summary_avg: np.ndarray,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scope", "scene", "AUC@1", "AUC@3", "AUC@5", "AUC@10", "num_reg_images", "num_images"]
        )
        writer.writerow(["summary_all", "__all__", *summary_all.tolist(), "", ""])
        writer.writerow(["summary_avg", "__avg__", *summary_avg.tolist(), "", ""])
        for row in rows:
            writer.writerow(
                ["scene", row.scene, *row.aucs.tolist(), row.num_reg_images, row.num_images]
            )


def write_compare_csv(
    out_path: Path,
    inst_rows: list[SceneEval],
    glo_rows: list[SceneEval],
) -> None:
    inst = {r.scene: r for r in inst_rows}
    glo = {r.scene: r for r in glo_rows}
    scenes = sorted(set(inst) & set(glo))

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
                "instantsfm_AUC@1",
                "glomap_AUC@1",
                "delta_AUC@1",
                "instantsfm_AUC@3",
                "glomap_AUC@3",
                "delta_AUC@3",
                "instantsfm_AUC@5",
                "glomap_AUC@5",
                "delta_AUC@5",
                "instantsfm_AUC@10",
                "glomap_AUC@10",
                "delta_AUC@10",
                "instantsfm_num_reg_images",
                "glomap_num_reg_images",
                "num_images",
            ]
        )
        for scene in scenes:
            i = inst[scene]
            g = glo[scene]
            writer.writerow(
                [
                    scene,
                    f"{i.aucs[0]:.6f}",
                    f"{g.aucs[0]:.6f}",
                    f"{(g.aucs[0] - i.aucs[0]):.6f}",
                    f"{i.aucs[1]:.6f}",
                    f"{g.aucs[1]:.6f}",
                    f"{(g.aucs[1] - i.aucs[1]):.6f}",
                    f"{i.aucs[2]:.6f}",
                    f"{g.aucs[2]:.6f}",
                    f"{(g.aucs[2] - i.aucs[2]):.6f}",
                    f"{i.aucs[3]:.6f}",
                    f"{g.aucs[3]:.6f}",
                    f"{(g.aucs[3] - i.aucs[3]):.6f}",
                    i.num_reg_images,
                    g.num_reg_images,
                    i.num_images,
                ]
            )


def main() -> None:
    args = parse_args()
    scenes = discover_scenes(args)
    if not scenes:
        raise SystemExit("No scenes found to evaluate.")

    if args.inject_gt_intrinsics:
        records = inject_gt_intrinsics_batch(args, scenes)
        status_path = args.output_dir / f"{args.prefix}_gtintrinsics_status.tsv"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("w") as f:
            f.write("scene\tok\tdetail\n")
            for scene, ok, detail in records:
                f.write(f"{scene}\t{int(ok)}\t{detail}\n")
        num_ok = sum(1 for _, ok, _ in records if ok)
        print(f"GT-intrinsics injection status: {num_ok}/{len(records)} scenes successful.")
        print(f"GT-intrinsics status file: {status_path}")

    if args.run_glomap:
        records = run_glomap_batch(args, scenes)
        status_path = args.output_dir / f"{args.prefix}_glomap_status.tsv"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("w") as f:
            f.write("scene\treturn_code\tmodel_ok\tlog_or_reason\n")
            for scene, return_code, model_ok, detail in records:
                f.write(f"{scene}\t{return_code}\t{int(model_ok)}\t{detail}\n")
        num_ok = sum(1 for _, _, ok, _ in records if ok)
        print(f"GLOMAP run status: {num_ok}/{len(records)} scenes have valid models.")
        print(f"GLOMAP status file: {status_path}")

    instantsfm_recon_root = args.dslr_root
    if args.run_instantsfm:
        instantsfm_recon_root, records = run_instantsfm_batch(args, scenes)
        status_path = args.output_dir / f"{args.prefix}_instantsfm_status.tsv"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("w") as f:
            f.write("scene\treturn_code\tmodel_ok\tlog_or_reason\n")
            for scene, return_code, model_ok, detail in records:
                f.write(f"{scene}\t{return_code}\t{int(model_ok)}\t{detail}\n")
        num_ok = sum(1 for _, _, ok, _ in records if ok)
        print(f"InstantSfM run status: {num_ok}/{len(records)} scenes have valid models.")
        print(f"InstantSfM status file: {status_path}")

    thresholds = np.array(args.thresholds, dtype=float)
    min_error = float(args.position_accuracy_gt)

    inst_rows, inst_all, inst_avg = evaluate_method(
        dslr_root=args.dslr_root,
        recon_root=instantsfm_recon_root,
        scenes=scenes,
        sparse_name=args.instantsfm_sparse_name,
        thresholds=thresholds,
        min_error=min_error,
    )
    glo_rows, glo_all, glo_avg = evaluate_method(
        dslr_root=args.dslr_root,
        recon_root=args.dslr_root,
        scenes=scenes,
        sparse_name=args.glomap_sparse_name,
        thresholds=thresholds,
        min_error=min_error,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inst_csv = args.output_dir / f"{args.prefix}_instantsfm.csv"
    glo_csv = args.output_dir / f"{args.prefix}_glomap.csv"
    cmp_csv = args.output_dir / f"{args.prefix}_compare.csv"

    write_method_csv(inst_csv, inst_rows, inst_all, inst_avg)
    write_method_csv(glo_csv, glo_rows, glo_all, glo_avg)
    write_compare_csv(cmp_csv, inst_rows, glo_rows)

    print(f"InstantSfM report: {inst_csv}")
    print(f"GLOMAP report:     {glo_csv}")
    print(f"Comparison report: {cmp_csv}")
    print("Summary (AUC@1/3/5/10):")
    print("  InstantSfM __all__:", ", ".join(f"{v:.6f}" for v in inst_all))
    print("  GLOMAP    __all__:", ", ".join(f"{v:.6f}" for v in glo_all))
    print("  Delta     __all__:", ", ".join(f"{(g - i):.6f}" for i, g in zip(inst_all, glo_all)))
    print("  InstantSfM __avg__:", ", ".join(f"{v:.6f}" for v in inst_avg))
    print("  GLOMAP    __avg__:", ", ".join(f"{v:.6f}" for v in glo_avg))
    print("  Delta     __avg__:", ", ".join(f"{(g - i):.6f}" for i, g in zip(inst_avg, glo_avg)))


if __name__ == "__main__":
    main()
