#!/usr/bin/env python3
"""Place several trained IRGS objects on a trained IRGS table.

The first-stage cleaned meshes determine the tabletop plane, object principal
axes, and contact height.  The final output is still a normal IRGS Gaussian
model, so base color, roughness, opacity, and the other learned attributes are
preserved for PBR rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "IRGS" / "outputs"


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    output_name: str
    scale: float
    u_fraction: float
    v_fraction: float
    yaw_degrees: float
    vertical_axis: str = "thin"
    vertical_sign: float = 1.0
    dark_end_up: bool = False


DEFAULT_OBJECTS = (
    ObjectSpec("penbox", "input_video_frames_penboxv1_12_mvinverse",
               0.35, -0.27, -0.19, -12.0),
    ObjectSpec("paper_cup", "input_video_frames_paper_cup_v2_21_mvinverse",
               0.40, 0.17, -0.18, 8.0, vertical_axis="long",
               dark_end_up=True),
    ObjectSpec("mouse", "input_video_frames_mousev1_12_mvinverse_image_features",
               0.38, -0.30, 0.20, 18.0, vertical_sign=-1.0),
    ObjectSpec("milk", "input_video_frames_milkv2_12_mvinverse_image_features",
               0.40, 0.00, 0.19, -8.0),
    ObjectSpec("tea", "input_video_frames_chayev1_12_mvinverse_image_features",
               0.36, 0.29, 0.20, 14.0),
)

TOY_OBJECT = ObjectSpec(
    "toy", "input_video_frames_toyv1_12_mvinverse",
    0.35, -0.30, 0.20, 5.0, vertical_axis="long",
)
ALL_OBJECTS = DEFAULT_OBJECTS + (TOY_OBJECT,)


def model_paths(output_name: str, iteration: int, model_stage: str = "irgs"):
    root = OUTPUTS / output_name
    return {
        "root": root,
        "model": root / model_stage,
        "mesh": root / "refgs" / "mesh_stage1" / "stage1_mesh_clean_largest.ply",
        "ply": root / model_stage / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def robust_pca(points: np.ndarray):
    center = np.median(points, axis=0)
    covariance = np.cov((points - center).T)
    values, axes = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    coordinates = (points - center) @ axes
    low, high = np.percentile(coordinates, [1.0, 99.0], axis=0)
    robust_center = center + axes @ ((low + high) * 0.5)
    extents = high - low
    return robust_center, axes, extents


def load_camera_positions(model_path: Path) -> np.ndarray:
    cameras = json.loads((model_path / "cameras.json").read_text(encoding="utf-8"))
    return np.asarray([camera["position"] for camera in cameras], dtype=np.float64)


def fit_tabletop(mesh: o3d.geometry.TriangleMesh, model_path: Path):
    vertices = np.asarray(mesh.vertices)
    diagonal = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    cloud = mesh.sample_points_uniformly(number_of_points=min(350_000, max(100_000, len(vertices) * 2)))
    equation, inliers = cloud.segment_plane(
        distance_threshold=diagonal * 0.006,
        ransac_n=3,
        num_iterations=3000,
    )
    normal = normalize(np.asarray(equation[:3], dtype=np.float64))
    d = float(equation[3]) / np.linalg.norm(equation[:3])
    plane_points = np.asarray(cloud.points)[np.asarray(inliers)]
    plane_center = np.median(plane_points, axis=0)
    plane_center -= (np.dot(normal, plane_center) + d) * normal

    camera_center = np.median(load_camera_positions(model_path), axis=0)
    if np.dot(normal, camera_center - plane_center) < 0:
        normal = -normal
        d = -d

    projected = plane_points - np.outer((plane_points - plane_center) @ normal, normal)
    covariance = np.cov((projected - plane_center).T)
    values, vectors = np.linalg.eigh(covariance)
    u = vectors[:, np.argmax(values)]
    u = normalize(u - np.dot(u, normal) * normal)
    v = normalize(np.cross(normal, u))
    if np.dot(np.cross(u, v), normal) < 0:
        v = -v

    uv = np.column_stack(((projected - plane_center) @ u, (projected - plane_center) @ v))
    uv_low, uv_high = np.percentile(uv, [2.0, 98.0], axis=0)
    plane_center += u * ((uv_low[0] + uv_high[0]) * 0.5)
    plane_center += v * ((uv_low[1] + uv_high[1]) * 0.5)
    half_extents = (uv_high - uv_low) * 0.5
    return plane_center, u, v, normal, half_extents, len(inliers), len(cloud.points)


def read_gaussians(path: Path):
    ply = PlyData.read(path)
    return ply, ply["vertex"].data.copy()


def quaternion_left_multiply(records: np.ndarray, rotation: np.ndarray):
    local_wxyz = np.column_stack(tuple(records[f"rot_{index}"] for index in range(4)))
    norms = np.linalg.norm(local_wxyz, axis=1, keepdims=True)
    local_wxyz = local_wxyz / np.maximum(norms, 1e-12)
    local_xyzw = local_wxyz[:, [1, 2, 3, 0]]
    world_rotation = Rotation.from_matrix(rotation) * Rotation.from_quat(local_xyzw)
    world_xyzw = world_rotation.as_quat()
    world_wxyz = world_xyzw[:, [3, 0, 1, 2]]
    for index in range(4):
        records[f"rot_{index}"] = world_wxyz[:, index]


def choose_vertical_sign(
    spec: ObjectSpec,
    axis: np.ndarray,
    center: np.ndarray,
    gaussian_records: np.ndarray,
) -> float:
    sign = float(spec.vertical_sign)
    if not spec.dark_end_up:
        return sign

    xyz = np.column_stack((gaussian_records["x"], gaussian_records["y"], gaussian_records["z"]))
    projection = (xyz - center) @ axis
    low_cut, high_cut = np.percentile(projection, [15.0, 85.0])
    colors = np.column_stack(tuple(gaussian_records[f"base_color_{index}"] for index in range(3)))
    # Stored base-color parameters are logits; sigmoid is monotonic and makes
    # the diagnostic luminance easier to interpret.
    colors = 1.0 / (1.0 + np.exp(-np.clip(colors, -20.0, 20.0)))
    low_luminance = float(colors[projection <= low_cut].mean())
    high_luminance = float(colors[projection >= high_cut].mean())
    # The paper-cup lid is the dark end and should point away from the table.
    sign = -1.0 if low_luminance < high_luminance else 1.0
    print(f"[{spec.name}] end luminance low={low_luminance:.3f}, high={high_luminance:.3f}; dark end up")
    return sign


def object_transform(
    spec: ObjectSpec,
    mesh: o3d.geometry.TriangleMesh,
    gaussian_records: np.ndarray,
    table_center: np.ndarray,
    table_u: np.ndarray,
    table_v: np.ndarray,
    table_up: np.ndarray,
    table_half_extents: np.ndarray,
    clearance_override: float | None = 0.0,
):
    vertices = np.asarray(mesh.vertices)
    center, axes, extents = robust_pca(vertices)
    vertical_index = 0 if spec.vertical_axis == "long" else 2
    z_source = axes[:, vertical_index]
    z_source *= choose_vertical_sign(spec, z_source, center, gaussian_records)
    remaining = [index for index in range(3) if index != vertical_index]
    x_index = max(remaining, key=lambda index: extents[index])
    x_source = axes[:, x_index]
    x_source = normalize(x_source - np.dot(x_source, z_source) * z_source)
    y_source = normalize(np.cross(z_source, x_source))
    source_basis = np.column_stack((x_source, y_source, z_source))

    yaw = math.radians(spec.yaw_degrees)
    x_target = normalize(math.cos(yaw) * table_u + math.sin(yaw) * table_v)
    y_target = normalize(np.cross(table_up, x_target))
    target_basis = np.column_stack((x_target, y_target, table_up))
    rotation = target_basis @ source_basis.T
    if np.linalg.det(rotation) < 0.999:
        raise RuntimeError(f"Invalid placement rotation for {spec.name}")

    relative = ((vertices - center) @ rotation.T) * spec.scale
    contact_height = float(np.percentile(relative @ table_up, 0.25))
    target_center = (
        table_center
        + table_u * (spec.u_fraction * 2.0 * table_half_extents[0])
        + table_v * (spec.v_fraction * 2.0 * table_half_extents[1])
    )
    # By default the robust lower mesh surface touches the fitted tabletop.
    # A positive gap is only introduced when explicitly requested.
    clearance = float(clearance_override) if clearance_override is not None else 0.0
    translation = target_center + table_up * (clearance - contact_height)
    transformed_vertices = ((vertices - center) @ rotation.T) * spec.scale + translation
    return rotation, center, translation, transformed_vertices, extents


def transform_gaussians(records, rotation, center, translation, scale):
    transformed = records.copy()
    xyz = np.column_stack((records["x"], records["y"], records["z"]))
    xyz = ((xyz - center) @ rotation.T) * scale + translation
    transformed["x"], transformed["y"], transformed["z"] = xyz.T
    log_scale = math.log(scale)
    transformed["scale_0"] += log_scale
    transformed["scale_1"] += log_scale
    quaternion_left_multiply(transformed, rotation)
    return transformed


def mesh_footprint(vertices, table_center, table_u, table_v):
    relative = vertices - table_center
    uv = np.column_stack((relative @ table_u, relative @ table_v))
    hull = ConvexHull(uv)
    polygon = uv[hull.vertices]
    signed_area = 0.5 * np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    if signed_area < 0:
        polygon = polygon[::-1]
    return polygon


def signed_distance_inside_convex_polygon(points, polygon, chunk_size=20_000):
    edge_start = polygon
    edges = np.roll(polygon, -1, axis=0) - polygon
    edge_length = np.maximum(np.linalg.norm(edges, axis=1), 1e-12)
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        relative = points[start:stop, None, :] - edge_start[None, :, :]
        cross = (
            edges[None, :, 0] * relative[:, :, 1]
            - edges[None, :, 1] * relative[:, :, 0]
        )
        result[start:stop] = np.min(cross / edge_length[None, :], axis=1)
    return result


def carve_table_gaussians(
    records, footprints, table_center, table_u, table_v, table_up,
    edge_width, slab_below, slab_above,
):
    xyz = np.column_stack((records["x"], records["y"], records["z"]))
    relative = xyz - table_center
    height = relative @ table_up
    slab = (height >= -slab_below) & (height <= slab_above)
    candidate_indices = np.flatnonzero(slab)
    candidate_uv = np.column_stack(
        (relative[candidate_indices] @ table_u, relative[candidate_indices] @ table_v)
    )
    delete = np.zeros(len(records), dtype=bool)
    opacity_factor = np.ones(len(records), dtype=np.float64)

    for polygon in footprints:
        signed_distance = signed_distance_inside_convex_polygon(candidate_uv, polygon)
        core = signed_distance >= edge_width
        transition = (signed_distance >= 0.0) & (signed_distance < edge_width)
        delete[candidate_indices[core]] = True
        t = np.clip(signed_distance[transition] / edge_width, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        factors = 1.0 - smooth
        transition_indices = candidate_indices[transition]
        opacity_factor[transition_indices] = np.minimum(
            opacity_factor[transition_indices], factors
        )

    transition = (~delete) & (opacity_factor < 1.0)
    adjusted = records.copy()
    if np.any(transition):
        raw = adjusted["opacity"][transition].astype(np.float64)
        alpha = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
        alpha = np.clip(alpha * opacity_factor[transition], 1e-6, 1.0 - 1e-6)
        adjusted["opacity"][transition] = np.log(alpha / (1.0 - alpha))
    return adjusted[~delete], int(delete.sum()), int(transition.sum())


def color_mesh(mesh: o3d.geometry.TriangleMesh, color):
    result = o3d.geometry.TriangleMesh(mesh)
    result.paint_uniform_color(color)
    result.compute_vertex_normals()
    return result


def write_layout_preview(path: Path, meshes, colors, table_center, u, v, up):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260901)
    samples = []
    for index, mesh in enumerate(meshes):
        vertices = np.asarray(mesh.vertices)
        count = min(18_000 if index == 0 else 9_000, len(vertices))
        indices = rng.choice(len(vertices), count, replace=False)
        points = vertices[indices] - table_center
        samples.append(np.column_stack((points @ u, points @ v, points @ up)))

    fig = plt.figure(figsize=(18, 6), dpi=180)
    views = ((28, -62, "Oblique mesh view"), (90, -90, "Top mesh view"),
             (2, -90, "Side/contact mesh view"))
    for panel, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = fig.add_subplot(1, 3, panel, projection="3d")
        for points, color in zip(samples, colors):
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25,
                         c=[color], alpha=0.48, depthshade=False)
        axis.set_title(title)
        axis.set_xlabel("table u")
        axis.set_ylabel("table v")
        axis.set_zlabel("table up")
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_box_aspect((4.0, 2.7, 2.0))
    fig.suptitle("First-stage mesh placement QA: table + five IRGS objects", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def update_cfg_args(source: Path, destination: Path, model_path: Path):
    text = source.read_text(encoding="utf-8")
    prefix = "Namespace("
    if not text.startswith(prefix) or not text.rstrip().endswith(")"):
        raise RuntimeError(f"Unsupported cfg_args format: {source}")
    # cfg_args is a Python Namespace repr rather than JSON.  Only replace the
    # model path while retaining every original table dataset/render option.
    old = str(source.parent.resolve())
    text = text.replace(repr(old), repr(str(model_path.resolve())), 1)
    destination.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="input_video_frames_basev11")
    parser.add_argument(
        "--table-stage", default="irgs",
        help="Table model directory, for example irgs_stage2_no_albedo.",
    )
    parser.add_argument("--output", default=str(OUTPUTS / "composed_basev11_five_objects" / "irgs"))
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument(
        "--object-scale-multiplier", type=float, default=1.0,
        help="Uniformly resize all placed objects without changing their layout.",
    )
    parser.add_argument(
        "--object-clearance", type=float, default=0.0,
        help=("Absolute distance above the fitted tabletop; defaults to 0 "
              "(no vertical offset)."),
    )
    parser.add_argument(
        "--object-clearance-override", action="append", default=[],
        metavar="NAME=VALUE",
        help=("Override tabletop clearance for one object; repeat for multiple "
              "objects, for example toy=0.025."),
    )
    parser.add_argument(
        "--exclude-object", action="append", default=[],
        choices=[spec.name for spec in ALL_OBJECTS],
        help="Skip one object; repeat the option to skip multiple objects.",
    )
    parser.add_argument(
        "--include-toy", action="store_true",
        help="Add the optional toyv1 IRGS model at the spare upper-left position.",
    )
    parser.add_argument(
        "--carve-table-under-objects", action="store_true",
        help="Remove guaranteed-hidden tabletop Gaussians under object mesh footprints.",
    )
    parser.add_argument("--carve-edge-width", type=float, default=0.02)
    parser.add_argument("--carve-slab-below", type=float, default=0.08)
    parser.add_argument("--carve-slab-above", type=float, default=0.06)
    args = parser.parse_args()
    # Keep tabletop RANSAC deterministic so placement ablations differ only
    # in the explicitly requested parameters.
    o3d.utility.random.seed(0)
    if args.object_scale_multiplier <= 0:
        parser.error("--object-scale-multiplier must be positive")
    if args.object_clearance is not None and args.object_clearance < 0:
        parser.error("--object-clearance must be non-negative")
    clearance_overrides = {}
    valid_object_names = {spec.name for spec in ALL_OBJECTS}
    for item in args.object_clearance_override:
        try:
            name, value_text = item.split("=", 1)
            value = float(value_text)
        except ValueError:
            parser.error(
                "--object-clearance-override must use NAME=VALUE syntax"
            )
        if name not in valid_object_names:
            parser.error(f"Unknown object in clearance override: {name}")
        if value < 0:
            parser.error("object clearance overrides must be non-negative")
        clearance_overrides[name] = value
    if args.carve_edge_width <= 0 or args.carve_slab_below < 0 or args.carve_slab_above < 0:
        parser.error("carve edge width must be positive and slab distances non-negative")

    output_model = Path(args.output).expanduser().resolve()
    iteration_dir = output_model / "point_cloud" / f"iteration_{args.iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    table = model_paths(args.table, args.iteration, args.table_stage)
    table_mesh = o3d.io.read_triangle_mesh(str(table["mesh"]))
    if table_mesh.is_empty():
        raise RuntimeError(f"Cannot read table mesh: {table['mesh']}")
    table_center, table_u, table_v, table_up, table_half, plane_inliers, plane_samples = fit_tabletop(
        table_mesh, table["model"]
    )
    print(f"[table] center={table_center.tolist()}")
    print(f"[table] up={table_up.tolist()}, half_extents={table_half.tolist()}, plane support={plane_inliers}/{plane_samples}")

    table_ply, table_records = read_gaussians(table["ply"])
    all_records = [table_records]
    object_footprints = []
    qa_meshes = [color_mesh(table_mesh, (0.55, 0.55, 0.55))]
    qa_colors = ((0.95, 0.25, 0.20), (0.95, 0.75, 0.15), (0.15, 0.75, 0.95),
                 (0.25, 0.90, 0.35), (0.75, 0.30, 0.95), (0.10, 0.45, 1.00))
    report = {
        "table": str(table["root"]),
        "table_model_stage": args.table_stage,
        "iteration": args.iteration,
        "table_plane": {
            "center": table_center.tolist(), "u": table_u.tolist(), "v": table_v.tolist(),
            "up": table_up.tolist(), "half_extents": table_half.tolist(),
            "mesh_ransac_support": [plane_inliers, plane_samples],
        },
        "objects": [],
        "notes": [
            "Table plane and object axes/contact heights come from first-stage cleaned meshes.",
            "IRGS Gaussian xyz, log-scale, and wxyz quaternion are transformed; learned material attributes are preserved.",
            "Higher-order SH coefficients are retained without directional rotation for this initial PBR composition test.",
        ],
    }

    color_by_name = {
        spec.name: color for spec, color in zip(ALL_OBJECTS, qa_colors)
    }
    requested_objects = list(DEFAULT_OBJECTS)
    if args.include_toy:
        requested_objects.append(TOY_OBJECT)
    selected_objects = [
        spec for spec in requested_objects if spec.name not in args.exclude_object
    ]
    if not selected_objects:
        parser.error("At least one object must remain")

    for spec in selected_objects:
        color = color_by_name[spec.name]
        paths = model_paths(spec.output_name, args.iteration)
        mesh = o3d.io.read_triangle_mesh(str(paths["mesh"]))
        object_ply, records = read_gaussians(paths["ply"])
        if records.dtype != table_records.dtype:
            raise RuntimeError(f"PLY property mismatch: {spec.name}")
        effective_spec = ObjectSpec(
            spec.name, spec.output_name, spec.scale * args.object_scale_multiplier,
            spec.u_fraction, spec.v_fraction, spec.yaw_degrees,
            spec.vertical_axis, spec.vertical_sign, spec.dark_end_up,
        )
        object_clearance = clearance_overrides.get(
            spec.name, args.object_clearance
        )
        rotation, center, translation, transformed_vertices, extents = object_transform(
            effective_spec, mesh, records, table_center, table_u, table_v, table_up,
            table_half, object_clearance,
        )
        all_records.append(transform_gaussians(
            records, rotation, center, translation, effective_spec.scale
        ))
        transformed_mesh = color_mesh(mesh, color)
        transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        transformed_mesh.compute_vertex_normals()
        qa_meshes.append(transformed_mesh)
        object_footprints.append(
            mesh_footprint(transformed_vertices, table_center, table_u, table_v)
        )
        report["objects"].append({
            "name": spec.name,
            "source": str(paths["root"]),
            "scale": effective_spec.scale,
            "table_uv_fraction": [spec.u_fraction, spec.v_fraction],
            "yaw_degrees": spec.yaw_degrees,
            "clearance": object_clearance,
            "source_mesh_robust_extents": extents.tolist(),
            "source_center": center.tolist(),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "gaussian_count": int(len(records)),
        })
        print(f"[{spec.name}] gaussians={len(records)}, scale={effective_spec.scale}, uv=({spec.u_fraction}, {spec.v_fraction})")

    if args.carve_table_under_objects:
        carved_table, deleted_count, faded_count = carve_table_gaussians(
            table_records, object_footprints, table_center, table_u, table_v, table_up,
            args.carve_edge_width, args.carve_slab_below, args.carve_slab_above,
        )
        all_records[0] = carved_table
        report["table_carving"] = {
            "enabled": True,
            "original_gaussians": int(len(table_records)),
            "kept_gaussians": int(len(carved_table)),
            "deleted_core_gaussians": deleted_count,
            "opacity_faded_edge_gaussians": faded_count,
            "edge_width": args.carve_edge_width,
            "slab_below": args.carve_slab_below,
            "slab_above": args.carve_slab_above,
        }
        print(
            f"[table carve] deleted={deleted_count}, opacity_faded={faded_count}, "
            f"kept={len(carved_table)}/{len(table_records)}"
        )
    else:
        report["table_carving"] = {"enabled": False}

    combined = np.concatenate(all_records)
    vertex = PlyElement.describe(combined, "vertex")
    PlyData([vertex], text=table_ply.text, byte_order=table_ply.byte_order).write(iteration_dir / "point_cloud.ply")

    combined_mesh = qa_meshes[0]
    for mesh in qa_meshes[1:]:
        combined_mesh += mesh
    o3d.io.write_triangle_mesh(str(output_model / "layout_mesh_colored.ply"), combined_mesh, write_ascii=False)
    write_layout_preview(
        output_model / "layout_mesh_preview.png", qa_meshes,
        ((0.55, 0.55, 0.55),) + qa_colors,
        table_center, table_u, table_v, table_up,
    )

    update_cfg_args(table["model"] / "cfg_args", output_model / "cfg_args", output_model)
    shutil.copy2(table["model"] / "cameras.json", output_model / "cameras.json")
    for filename in ("point_cloud1.map",):
        source = table["ply"].parent / filename
        if source.is_file():
            shutil.copy2(source, iteration_dir / filename)
    input_ply = table["model"] / "input.ply"
    if input_ply.is_file():
        shutil.copy2(input_ply, output_model / "input.ply")

    report["total_gaussians"] = int(len(combined))
    report["object_scale_multiplier"] = args.object_scale_multiplier
    report["object_clearance"] = args.object_clearance
    report["object_clearance_overrides"] = clearance_overrides
    report["excluded_objects"] = args.exclude_object
    report["output_model"] = str(output_model)
    (output_model / "layout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Combined model: {iteration_dir / 'point_cloud.ply'}")
    print(f"Mesh QA: {output_model / 'layout_mesh_colored.ply'}")
    print(f"Mesh preview: {output_model / 'layout_mesh_preview.png'}")
    print(f"Layout report: {output_model / 'layout.json'}")


if __name__ == "__main__":
    main()
