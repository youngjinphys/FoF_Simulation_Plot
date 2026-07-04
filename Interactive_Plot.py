#!/usr/bin/env python3
"""
Interactive_Plot.py

Generates an interactive WebGL-powered 3D cosmology visualization using Plotly & Dash.
Optimized for high-performance browser rendering (Hovertext compression, NaN line breaks, PBR Lighting).

Author: Antigravity Agent
"""

import os
import base64
import tempfile
import threading
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from functools import lru_cache
import copy
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

from lib.pipeline_config import setup_common_argparse, get_global_config
from lib.common_io import CATALOG_NAME, read_catalog_data, read_catalog_path
from lib.filament_components import build_periodic_filament_catalog
from lib.tensor_physics import (
    ellipsoid_axis_aligned_extents,
    periodic_image_indices_and_shifts,
    principal_rms_lengths,
    principal_rms_scale_factors,
)


@dataclass(frozen=True)
class InteractiveSizeModel:
    rms_axes: np.ndarray
    physical_axes: np.ndarray
    rendered_axes: np.ndarray
    mesh_scale_factors: np.ndarray
    size_label: str
    display_transform_label: str


def halo_preview_geometry(rendered_axes, eigenvectors):
    """Return the exact descending axes and basis used by the rendered mesh."""
    axes = np.asarray(rendered_axes, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("rendered_axes must have shape (N, 3)")
    if vectors.shape != (len(axes), 3, 3):
        raise ValueError("eigenvectors must have shape (N, 3, 3)")
    if np.any(~np.isfinite(axes) | (axes <= 0.0)):
        raise ValueError("rendered_axes must be finite and positive")
    if np.any(~np.isfinite(vectors)):
        raise ValueError("eigenvectors must be finite")

    basis = vectors[:, :, ::-1].copy()
    gram = np.matmul(np.swapaxes(basis, 1, 2), basis)
    target = np.broadcast_to(np.eye(3, dtype=np.float64), gram.shape)
    if not np.allclose(gram, target, rtol=1.0e-7, atol=1.0e-9):
        raise ValueError("eigenvectors must form an orthonormal basis")
    return axes[:, ::-1].copy(), basis


def interactive_size_model(
    eigenvalues,
    particles,
    tensor_type,
    box_size,
    *,
    boundary_mode="rms",
    log_scale=False,
):
    """Separate COM-centered catalog RMS axes from assumptions and display transforms."""
    if boundary_mode not in {"rms", "uniform"}:
        raise ValueError("boundary_mode must be 'rms' or 'uniform'")

    rms_axes = principal_rms_lengths(
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(particles, dtype=np.float64),
        tensor_type,
        box_size,
    )
    base_scale = principal_rms_scale_factors(
        np.asarray(particles, dtype=np.float64),
        tensor_type,
        box_size,
    )
    boundary_factor = np.sqrt(5.0) if boundary_mode == "uniform" else 1.0
    physical_axes = rms_axes * boundary_factor
    rendered_axes = physical_axes.copy()
    display_ratio = np.ones(len(physical_axes), dtype=np.float64)

    if log_scale and len(physical_axes) > 0:
        major = physical_axes[:, 2]
        finite = np.isfinite(major) & (major > 1.0e-12)
        if np.any(finite):
            maximum = float(np.max(major[finite]))
            normalized = major[finite] / maximum
            visual_major = (
                np.log1p(100.0 * normalized) / np.log1p(100.0) * maximum
            )
            display_ratio[finite] = visual_major / major[finite]
            rendered_axes[finite] *= display_ratio[finite, None]

    coordinate_label = (
        "COM-centered Lagrangian qq-tensor"
        if tensor_type == "qq"
        else "COM-centered Eulerian xx-tensor"
    )
    size_label = (
        f"{coordinate_label} uniform-density boundary half-axes "
        "(√5σ; uniform-density assumption)"
        if boundary_mode == "uniform"
        else f"{coordinate_label} principal RMS half-axes (1σ)"
    )
    display_label = (
        "Nonlinear visual exaggeration" if log_scale else "Physical linear scale"
    )
    return InteractiveSizeModel(
        rms_axes=rms_axes,
        physical_axes=physical_axes,
        rendered_axes=rendered_axes,
        mesh_scale_factors=base_scale * boundary_factor * display_ratio,
        size_label=size_label,
        display_transform_label=display_label,
    )


def _periodic_edge_segments(start, end, box_size):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    displacement = end - start
    displacement -= np.round(displacement / box_size) * box_size
    unwrapped_end = start + displacement

    events = [0.0, 1.0]
    for dimension in range(3):
        delta = displacement[dimension]
        if delta == 0.0:
            continue
        if unwrapped_end[dimension] < 0.0:
            event = -start[dimension] / delta
        elif unwrapped_end[dimension] > box_size:
            event = (box_size - start[dimension]) / delta
        else:
            continue
        if 0.0 < event < 1.0:
            events.append(float(event))

    events = sorted(set(events))
    segments = []
    for left, right in zip(events[:-1], events[1:]):
        point_a = start + left * displacement
        point_b = start + right * displacement
        midpoint = 0.5 * (point_a + point_b)
        shift = -np.floor(midpoint / box_size) * box_size
        segment = np.stack([point_a + shift, point_b + shift])
        segments.append(np.clip(segment, 0.0, box_size))
    return segments


def _edge_segments(start, end, box_size, *, periodic=True):
    if periodic:
        return _periodic_edge_segments(start, end, box_size)
    segment = np.stack([
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    ])
    return [np.clip(segment, 0.0, box_size)]


def _component_focus(component, catalog, box_size=None, *, periodic=True):
    positions = np.asarray(catalog.wrapped_positions[component.member_indices], dtype=np.float64)
    if len(positions) == 0:
        center = np.zeros(3, dtype=np.float64)
        return center, 1.0

    if periodic and box_size is not None:
        anchor = positions[0]
        deltas = positions - anchor
        deltas -= np.round(deltas / box_size) * box_size
        unwrapped = anchor + deltas
        center_unwrapped = np.mean(unwrapped, axis=0)
        center = np.mod(center_unwrapped, box_size)
        distances = np.linalg.norm(unwrapped - center_unwrapped, axis=1)
        floor = 0.02 * float(box_size)
    else:
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        floor = max(float(np.ptp(positions, axis=0).max()) * 0.1, 1.0)

    radius = max(
        float(np.max(distances)) if len(distances) else 0.0,
        0.5 * float(component.bounding_box_diagonal),
        floor,
    )
    return center.astype(np.float64), radius


# Filament skeleton and member-marker base appearance. Selection emphasis is
# applied to one skeleton trace at a time by assets/filament_hover.js.
FILAMENT_BASE_WIDTH = 4
FILAMENT_MEMBER_REST_SIZE = 4.0
FILAMENT_MEMBER_REST_ALPHA = 0.65
# Mesh3d batches all ellipsoids into one trace, so a near-transparent center
# marker is the per-halo picking surface. A fully transparent WebGL marker is
# not picked consistently across browsers/GPUs, so keep a tiny alpha and a
# slightly larger target while remaining visually unobtrusive.
HALO_HIT_TARGET_SIZE = 12.0
MAX_MESH_IMAGES = 25_000
MAX_SCENE_HALOS = 25_000
MAX_CUSTOM_CATALOG_BYTES = 128 * 1024**2
MAX_FILAMENT_CANDIDATE_EDGES = 2_000_000
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep ownership server-side: dcc.Store is client-controlled and must never be
# trusted as authority to delete a filesystem path.
_CUSTOM_TEMP_DIRS = {}
_CUSTOM_TEMP_DIRS_LOCK = threading.Lock()


def _prepare_custom_catalog_dir():
    holder = tempfile.TemporaryDirectory(prefix="fof-interactive-")
    tmp_dir = holder.name
    with _CUSTOM_TEMP_DIRS_LOCK:
        _CUSTOM_TEMP_DIRS[tmp_dir] = holder
    zero_dir = os.path.join(tmp_dir, "0")
    os.makedirs(zero_dir, exist_ok=True)
    return tmp_dir, os.path.join(zero_dir, CATALOG_NAME)


def _cleanup_custom_catalog_dir(store_data):
    if not isinstance(store_data, dict):
        return False
    tmp_dir = store_data.get("temporary_dir")
    if not isinstance(tmp_dir, str):
        return False
    with _CUSTOM_TEMP_DIRS_LOCK:
        holder = _CUSTOM_TEMP_DIRS.pop(tmp_dir, None)
    if holder is None:
        return False
    holder.cleanup()
    return True


def _custom_catalog_too_large(file_path):
    return os.path.getsize(file_path) > MAX_CUSTOM_CATALOG_BYTES


def _is_bin_catalog_name(path_or_name):
    return (
        isinstance(path_or_name, str)
        and path_or_name.strip().lower().endswith(".bin")
    )


def _resolve_existing_custom_input_path(path_text):
    """Resolve trusted local custom input paths.

    Users commonly paste either an absolute path or a project-relative path
    such as ``data/FoF_E/resE499/...bin``.  Resolve relative paths against both
    the server working directory and this script's directory; return the first
    existing path as an absolute normalized path.
    """
    candidate = os.path.expanduser(os.fspath(path_text))
    if os.path.isabs(candidate):
        candidates = [candidate]
    else:
        candidates = [
            os.path.abspath(candidate),
            os.path.join(APP_ROOT, candidate),
        ]
    seen = set()
    normalized = []
    for item in candidates:
        resolved = os.path.abspath(os.path.normpath(item))
        if resolved not in seen:
            seen.add(resolved)
            normalized.append(resolved)
    for item in normalized:
        if os.path.exists(item):
            return item, normalized
    return normalized[0] if normalized else candidate, normalized


def _resolve_custom_catalog_path(base_dir, file_index):
    """Resolve common trusted local directory layouts without guessing data."""
    base = os.fspath(base_dir)
    index = int(file_index)
    candidates = [
        os.path.join(base, CATALOG_NAME),
        os.path.join(base, str(index), CATALOG_NAME),
        os.path.join(base, f"resE{index}", CATALOG_NAME),
        os.path.join(base, f"resR{index}", CATALOG_NAME),
        os.path.join(f"{base}{index}", CATALOG_NAME),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"no {CATALOG_NAME} found for index {index} under {base}"
    )


def _validate_custom_catalog_path(file_path):
    """Validate schema and exact payload length before accepting local input."""
    data = read_catalog_path(os.fspath(file_path))
    if data is None or len(data) == 0:
        raise ValueError("catalog contains no halo rows")
    return int(len(data))


def mesh_resolution_for_count(image_count):
    if isinstance(image_count, bool) or not isinstance(image_count, (int, np.integer)):
        raise ValueError("image_count must be a non-negative integer")
    if image_count < 0:
        raise ValueError("image_count must be a non-negative integer")
    if image_count <= 1_000:
        return 10
    if image_count <= 4_000:
        return 8
    return 6


def select_mesh_images(particles, max_images=MAX_MESH_IMAGES):
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 1:
        raise ValueError("particles must be one-dimensional")
    if isinstance(max_images, bool) or not isinstance(max_images, (int, np.integer)):
        raise ValueError("max_images must be a positive integer")
    if max_images <= 0:
        raise ValueError("max_images must be a positive integer")
    selected = np.ones(len(particles), dtype=bool)
    if len(particles) <= max_images:
        return selected
    selected[:] = False
    order = np.argsort(-particles, kind="stable")[:max_images]
    selected[order] = True
    return selected


def select_scene_halos(valid_mask, particles, max_halos=MAX_SCENE_HALOS):
    """Bound display payload using the highest-particle valid halos."""
    valid = np.asarray(valid_mask, dtype=bool)
    particles = np.asarray(particles, dtype=np.float64)
    if valid.ndim != 1 or particles.ndim != 1 or valid.shape != particles.shape:
        raise ValueError("valid_mask and particles must be matching 1D arrays")
    if isinstance(max_halos, bool) or not isinstance(max_halos, (int, np.integer)):
        raise ValueError("max_halos must be a positive integer")
    if max_halos <= 0:
        raise ValueError("max_halos must be a positive integer")

    candidate_indices = np.flatnonzero(valid)
    if len(candidate_indices) <= max_halos:
        return valid.copy(), len(candidate_indices)
    order = np.argsort(-particles[candidate_indices], kind="stable")[:max_halos]
    selected = np.zeros_like(valid)
    selected[candidate_indices[order]] = True
    return selected, len(candidate_indices)


def _rgb_to_rgba(rgb_string, alpha):
    """Convert a Plotly 'rgb(r, g, b)' string into an 'rgba(r, g, b, a)' one."""
    inside = rgb_string[rgb_string.find("(") + 1 : rgb_string.find(")")]
    red, green, blue = (part.strip() for part in inside.split(",")[:3])
    return f"rgba({red}, {green}, {blue}, {alpha})"


# Cyan-magenta palette for filament skeletons.  Chosen to contrast with the
# T-web environment colorbar (grey/limegreen/dodgerblue/red) so that the
# filament network does not visually merge with environment-coloured halos.
_FILAMENT_COLORSCALE = [
    [0.0, "rgb(0, 200, 200)"],    # Cyan (small components)
    [0.5, "rgb(120, 100, 220)"],  # Purple (mid)
    [1.0, "rgb(220, 60, 180)"],   # Magenta (large components)
]


@dataclass(frozen=True)
class FilamentCountScale:
    mode: str
    fractions: np.ndarray
    tick_values: tuple[int, ...]


def filament_count_scale(counts):
    """Normalize component member counts and describe the displayed scale."""
    values = np.asarray(counts, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) == 0
        or np.any(~np.isfinite(values) | (values < 1.0))
    ):
        raise ValueError(
            "counts must be a non-empty vector of positive finite values"
        )

    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        return FilamentCountScale(
            mode="constant",
            fractions=np.full(len(values), 0.5, dtype=np.float64),
            tick_values=(int(low),),
        )

    mode = "log" if high / low >= 10.0 else "linear"
    transformed = np.log10(values) if mode == "log" else values
    fractions = (transformed - np.min(transformed)) / np.ptp(transformed)
    midpoint = np.sqrt(low * high) if mode == "log" else 0.5 * (low + high)
    ticks = tuple(dict.fromkeys(int(round(value)) for value in (low, midpoint, high)))
    return FilamentCountScale(mode, fractions, ticks)


def filament_component_colors(catalog):
    """One base colour per component using the shared adaptive count scale.

    Uses a cyan-to-magenta palette that visually contrasts with every
    T-web environment class colour.
    """
    if not catalog.components:
        return []
    counts = np.array(
        [component.halo_count for component in catalog.components], dtype=np.float64
    )
    scale = filament_count_scale(counts)
    return [
        sample_colorscale(_FILAMENT_COLORSCALE, [fraction])[0]
        for fraction in scale.fractions
    ]


def _filament_custom_row(component, catalog, method_label, *, box_size=None, periodic=True):
    focus_center, focus_radius = _component_focus(
        component, catalog, box_size, periodic=periodic
    )
    return [
        "filament",
        component.component_id,
        component.halo_count,
        component.total_mass,
        component.median_mass,
        component.min_mass,
        component.max_mass,
        component.total_edge_length,
        component.bounding_box_diagonal,
        catalog.max_distance,
        method_label,
        float(focus_center[0]),
        float(focus_center[1]),
        float(focus_center[2]),
        float(focus_radius),
    ]


def _minimum_image_delta(start, end, box_size, *, periodic=True):
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    if periodic:
        delta -= np.round(delta / float(box_size)) * float(box_size)
    return delta


def _halo_skeleton_angle_summaries(
    catalog,
    *,
    positions,
    major_axes,
    row_indices,
    box_size,
    periodic=True,
):
    """Return per-scene-halo MST-edge/major-axis angle summaries.

    Angles are local to the current interactive scene: the skeleton edges come
    from the retained thresholded MST forest and the halo major axis comes from
    the currently selected tensor (XX or QQ). Both the skeleton segment and the
    major axis are undirected, so the displayed angle is acute in [0, 90] deg.
    """
    count = len(positions)
    summaries = [
        {
            "component_id": -1,
            "edge_count": 0,
            "summary": "No adjacent retained MST skeleton edge at the current threshold/method.",
        }
        for _ in range(count)
    ]
    if catalog is None or len(getattr(catalog, "edge_source_indices", ())) == 0:
        return summaries

    position_array = np.asarray(positions, dtype=np.float64)
    axis_array = np.asarray(major_axes, dtype=np.float64)
    row_array = np.asarray(row_indices)
    per_halo = {index: [] for index in range(count)}
    component_ids = np.asarray(catalog.edge_component_ids, dtype=np.int64)
    source_pairs = np.asarray(catalog.edge_source_indices, dtype=np.int64)
    lengths = np.asarray(catalog.edge_lengths, dtype=np.float64)

    for (source_a, source_b), component_id, edge_length in zip(
        source_pairs, component_ids, lengths
    ):
        source_a = int(source_a)
        source_b = int(source_b)
        if not (0 <= source_a < count and 0 <= source_b < count):
            continue
        for center, neighbor in ((source_a, source_b), (source_b, source_a)):
            segment = _minimum_image_delta(
                position_array[center],
                position_array[neighbor],
                box_size,
                periodic=periodic,
            )
            segment_norm = float(np.linalg.norm(segment))
            axis = axis_array[center]
            axis_norm = float(np.linalg.norm(axis))
            if segment_norm <= 0.0 or axis_norm <= 0.0:
                continue
            cos_angle = float(
                abs(np.dot(axis, segment) / (axis_norm * segment_norm))
            )
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            per_halo[center].append(
                {
                    "component_id": int(component_id),
                    "neighbor_row": int(row_array[neighbor]),
                    "length": float(edge_length),
                    "angle": angle_deg,
                }
            )

    for halo_index, rows in per_halo.items():
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: (row["component_id"], row["neighbor_row"]))
        component_labels = sorted({row["component_id"] for row in rows})
        angle_text = "; ".join(
            (
                f"row {row['neighbor_row']}: {row['angle']:.1f}° "
                f"(edge {row['length']:.3f} Mpc/h)"
            )
            for row in rows[:6]
        )
        if len(rows) > 6:
            angle_text += f"; +{len(rows) - 6} more"
        summaries[halo_index] = {
            "component_id": component_labels[0] if len(component_labels) == 1 else -1,
            "edge_count": len(rows),
            "summary": angle_text,
        }
    return summaries


def build_filament_traces(catalog, colors=None, *, box_size, method_label, periodic_edges=True):
    """Return one selectable Plotly line trace per MST component."""
    if not catalog.components:
        return []
    if colors is None:
        colors = filament_component_colors(catalog)
    if len(colors) != len(catalog.components):
        raise ValueError("colors must contain one entry per filament component")

    traces = []
    for component, base_color in zip(catalog.components, colors):
        edge_mask = catalog.edge_component_ids == component.component_id
        x_values, y_values, z_values, custom_rows = [], [], [], []
        focus_center, focus_radius = _component_focus(
            component, catalog, box_size, periodic=periodic_edges
        )
        custom_row = _filament_custom_row(
            component,
            catalog,
            method_label,
            box_size=box_size,
            periodic=periodic_edges,
        )

        for edge in catalog.edges[edge_mask]:
            for segment in _edge_segments(
                edge[0], edge[1], box_size, periodic=periodic_edges
            ):
                x_values.extend([segment[0, 0], segment[1, 0], None])
                y_values.extend([segment[0, 1], segment[1, 1], None])
                z_values.extend([segment[0, 2], segment[1, 2], None])
                custom_rows.extend([custom_row, custom_row, None])

        traces.append(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                line=dict(color=base_color, width=FILAMENT_BASE_WIDTH),
                # Keep hover/click events alive without drawing native tooltip
                # labels; the OSD inspector owns the actual information display.
                text="",
                customdata=custom_rows,
                hoverinfo="text",
                name=f"Filament #{component.component_id}",
                legendgroup="filaments",
                showlegend=False,
                meta={
                    "role": "filament",
                    "component_id": component.component_id,
                    "base_color": base_color,
                    "base_width": FILAMENT_BASE_WIDTH,
                    "focus_center": tuple(float(value) for value in focus_center),
                    "focus_radius": float(focus_radius),
                },
            )
        )
    return traces


def count_periodic_crossing_edges(catalog, box_size):
    """Count MST skeleton edges whose minimum-image path crosses a box face."""
    if catalog is None or len(catalog.edges) == 0:
        return 0
    raw_delta = np.abs(catalog.edges[:, 1] - catalog.edges[:, 0])
    return int(np.sum(np.any(raw_delta > 0.5 * float(box_size), axis=1)))


def build_pbc_continuation_trace(catalog, colors, *, box_size, method_label):
    """Mark paired face endpoints for PBC-crossing skeleton edges."""
    if catalog is None or not catalog.components:
        return None
    xs, ys, zs, marker_colors, custom_rows, component_ids = [], [], [], [], [], []
    components = {component.component_id: component for component in catalog.components}
    for edge, component_id in zip(catalog.edges, catalog.edge_component_ids):
        if not np.any(np.abs(edge[1] - edge[0]) > 0.5 * float(box_size)):
            continue
        segments = _periodic_edge_segments(edge[0], edge[1], box_size)
        if len(segments) < 2:
            continue
        component_id = int(component_id)
        component = components[component_id]
        row = _filament_custom_row(
            component,
            catalog,
            method_label,
            box_size=box_size,
            periodic=True,
        )
        # The first segment exits one face and the last segment enters its
        # periodic partner. Plotly renders these as a paired continuation cue.
        for point in (segments[0][1], segments[-1][0]):
            xs.append(float(point[0]))
            ys.append(float(point[1]))
            zs.append(float(point[2]))
            marker_colors.append(colors[component_id])
            custom_rows.append(row)
            component_ids.append(component_id)
    if not xs:
        return None
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=5, symbol="diamond", color=marker_colors, opacity=0.95),
        customdata=custom_rows,
        text="",
        hoverinfo="text",
        name="PBC continuations",
        showlegend=False,
        meta={"role": "pbc_continuations", "component_ids": component_ids},
    )


def build_filament_member_trace(
    catalog,
    colors,
    *,
    f_mass,
    f_particles,
    method_label,
    box_size=None,
    periodic_edges=True,
):
    """Return one lightweight picking trace over all filament-member halos."""
    if not catalog.components:
        return None

    xs, ys, zs = [], [], []
    component_ids = []
    rest_colors = []
    custom_rows = []
    for component, base_color in zip(catalog.components, colors):
        member_positions = catalog.wrapped_positions[component.member_indices]
        rest_rgba = _rgb_to_rgba(base_color, FILAMENT_MEMBER_REST_ALPHA)
        custom_row = _filament_custom_row(
            component,
            catalog,
            method_label,
            box_size=box_size,
            periodic=periodic_edges,
        )
        for source_index, position in zip(component.source_indices, member_positions):
            xs.append(position[0])
            ys.append(position[1])
            zs.append(position[2])
            component_ids.append(int(component.component_id))
            rest_colors.append(rest_rgba)
            custom_rows.append(
                custom_row
                + [float(f_mass[source_index]), int(f_particles[source_index])]
            )

    if not xs:
        return None

    rest_size = [FILAMENT_MEMBER_REST_SIZE] * len(xs)
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=rest_size,
            color=rest_colors,
            line=dict(width=0),
        ),
        customdata=custom_rows,
        text="",
        hoverinfo="text",
        name="Filament members",
        legendgroup="filaments",
        showlegend=False,
        meta={
            "role": "filament_members",
            "component_ids": component_ids,
        },
    )


# ==========================================
# Pre-computed Mesh Geometry
# ==========================================
@lru_cache(maxsize=4)
def generate_unit_sphere(res=12):
    if isinstance(res, bool) or not isinstance(res, (int, np.integer)) or res < 4:
        raise ValueError("sphere resolution must be an integer >= 4")
    u = np.linspace(0, 2.0 * np.pi, res, dtype=np.float32)
    v = np.linspace(0, np.pi, res, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)
    
    x = np.cos(u_grid) * np.sin(v_grid)
    y = np.sin(u_grid) * np.sin(v_grid)
    z = np.cos(v_grid)
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()]).astype(
        np.float32, copy=False
    )
    
    i_list, j_list, k_list = [], [], []
    for r in range(res - 1):
        for c in range(res - 1):
            p1 = r * res + c
            p2 = p1 + 1
            p3 = (r + 1) * res + c
            p4 = p3 + 1
            i_list.extend([p1, p3])
            j_list.extend([p3, p4])
            k_list.extend([p2, p2])
            
    faces = np.asarray([i_list, j_list, k_list], dtype=np.int32)
    return vertices, faces


def build_ellipsoids_mesh(
    pos,
    eigenvalues,
    eigenvectors,
    T_values,
    K_scale=1.0,
    *,
    resolution=None,
):
    """Builds a single Mesh3d trace for N ellipsoids using cached geometry (fully vectorized)."""
    N = len(pos)
    if resolution is None:
        resolution = mesh_resolution_for_count(N)
    unit_vertices, unit_faces = generate_unit_sphere(resolution)
    vertices_per_ellipsoid = unit_vertices.shape[1]
    if N == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
        
    K_scale = np.atleast_1d(K_scale).astype(np.float32, copy=False)
    if len(K_scale) == 1:
        K_scale = np.repeat(K_scale[0], N)

    eigenvalues = np.asarray(eigenvalues, dtype=np.float32)
    a_major = np.sqrt(np.maximum(eigenvalues[:, 2], 1e-12)) * K_scale
    b_inter = np.sqrt(np.maximum(eigenvalues[:, 1], 1e-12)) * K_scale
    c_minor = np.sqrt(np.maximum(eigenvalues[:, 0], 1e-12)) * K_scale
    
    # (N, 3, 1) * (1, 3, V) -> (N, 3, V)
    abc = np.stack([a_major, b_inter, c_minor], axis=1)[:, :, None]
    scaled_v = abc * unit_vertices[None, :, :]
    
    # Rotation (reverse last axis to match eigenvectors order)
    v_rot = np.asarray(eigenvectors[:, :, ::-1], dtype=np.float32)
    
    # (N, 3, 3) @ (N, 3, V) -> (N, 3, V)
    rot_v = np.matmul(v_rot, scaled_v)
    
    # Translation
    rot_v += np.asarray(pos, dtype=np.float32)[:, :, None]
    
    all_x = rot_v[:, 0, :].ravel()
    all_y = rot_v[:, 1, :].ravel()
    all_z = rot_v[:, 2, :].ravel()
    
    # Faces: add offset for each halo
    offsets = np.arange(N, dtype=np.int32)[:, None, None] * vertices_per_ellipsoid
    faces = unit_faces[None, :, :] + offsets # (N, 3, F)
    
    all_i = faces[:, 0, :].ravel()
    all_j = faces[:, 1, :].ravel()
    all_k = faces[:, 2, :].ravel()
    
    all_intensity = np.repeat(
        np.asarray(T_values, dtype=np.float32), vertices_per_ellipsoid
    )
    
    return all_x, all_y, all_z, all_i, all_j, all_k, all_intensity


# ==========================================
# Data Processing Global State
# ==========================================
def get_environment(lambda_tide, threshold=5e-12):
    """
    Returns T-web environment classification:
    3: Knot, 2: Filament, 1: Sheet, 0: Void
    """
    return np.sum(lambda_tide > threshold, axis=1)

def get_filament_probability(lambda_tide, threshold=5e-12, k_sig=1.0e12):
    """
    Compute a heuristic continuous filament weight in [0, 1].

    This sigmoid partition is useful for visualization but is not a calibrated
    posterior probability without a statistical model for the eigenvalues.
    Filament condition: 2 eigenvalues > threshold, 1 eigenvalue < threshold.
    """
    w_tide = np.sort(lambda_tide, axis=1) # Ascending order: l3 <= l2 <= l1
    l3, l2, l1 = w_tide[:, 0], w_tide[:, 1], w_tide[:, 2]
    
    def sigmoid(x):
        # Clip to avoid overflow/underflow in exp
        return 1.0 / (1.0 + np.exp(-np.clip(k_sig * x, -20, 20)))
    
    s1 = sigmoid(l1 - threshold)
    s2 = sigmoid(l2 - threshold)
    s3 = sigmoid(l3 - threshold)
    
    # CRITICAL MATH FIX:
    # Eigenvalues are strictly ordered: l1 >= l2 >= l3.
    # Therefore, the events of exceeding the threshold are nested:
    # (l3 > th) implies (l2 > th) implies (l1 > th).
    # The mutually exclusive classes form a partition:
    # P(Filament) = P(l2 > th) - P(l3 > th) = s2 - s3
    return s2 - s3

def get_cic_grid(pos, weights, box_size, grid_res, *, periodic=True):
    """Cloud-In-Cell assignment onto a 3D grid."""
    pos = np.asarray(pos, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    grid = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    cell_size = box_size / grid_res
    p = pos / cell_size
    idx = np.floor(p).astype(int)
    d = p - idx
    t = 1.0 - d

    if not periodic:
        contributions = []
        normalizer = np.zeros(len(pos), dtype=np.float64)
        for ox in (0, 1):
            wx = d[:, 0] if ox else t[:, 0]
            ix = idx[:, 0] + ox
            for oy in (0, 1):
                wy = d[:, 1] if oy else t[:, 1]
                iy = idx[:, 1] + oy
                for oz in (0, 1):
                    wz = d[:, 2] if oz else t[:, 2]
                    iz = idx[:, 2] + oz
                    frac = wx * wy * wz
                    valid = (
                        (ix >= 0)
                        & (ix < grid_res)
                        & (iy >= 0)
                        & (iy < grid_res)
                        & (iz >= 0)
                        & (iz < grid_res)
                    )
                    normalizer[valid] += frac[valid]
                    contributions.append((ix, iy, iz, frac, valid))

        nonzero = normalizer > 0.0
        for ix, iy, iz, frac, valid in contributions:
            use = valid & nonzero
            if np.any(use):
                np.add.at(
                    grid,
                    (ix[use], iy[use], iz[use]),
                    weights[use] * frac[use] / normalizer[use],
                )
        return grid

    i0 = idx[:, 0] % grid_res
    j0 = idx[:, 1] % grid_res
    k0 = idx[:, 2] % grid_res
    i1 = (i0 + 1) % grid_res
    j1 = (j0 + 1) % grid_res
    k1 = (k0 + 1) % grid_res
    
    np.add.at(grid, (i0, j0, k0), weights * t[:,0] * t[:,1] * t[:,2])
    np.add.at(grid, (i1, j0, k0), weights * d[:,0] * t[:,1] * t[:,2])
    np.add.at(grid, (i0, j1, k0), weights * t[:,0] * d[:,1] * t[:,2])
    np.add.at(grid, (i1, j1, k0), weights * d[:,0] * d[:,1] * t[:,2])
    np.add.at(grid, (i0, j0, k1), weights * t[:,0] * t[:,1] * d[:,2])
    np.add.at(grid, (i1, j0, k1), weights * d[:,0] * t[:,1] * d[:,2])
    np.add.at(grid, (i0, j1, k1), weights * t[:,0] * d[:,1] * d[:,2])
    np.add.at(grid, (i1, j1, k1), weights * d[:,0] * d[:,1] * d[:,2])
    return grid

def parse_tensor_symmetrize(flat):
    """
    Parses a flattened 9-element array into a 3x3 symmetric tensor.
    The binary catalog stores the tensor in C-order (row-major).
    reshape(-1, 3, 3) correctly maps this layout to C-order 3x3 matrices.
    """
    t = flat.astype(np.float64, copy=False).reshape((-1, 3, 3))
    return 0.5 * (t + np.swapaxes(t, 1, 2))

def get_shape_properties(xx_flat):
    xx = parse_tensor_symmetrize(xx_flat)
    w, v = np.linalg.eigh(xx)
    
    finite = np.all(np.isfinite(w), axis=1) & np.all(np.isfinite(v), axis=(1, 2))
    pos_def = finite & (w[:, 0] > 0) & (w[:, 1] > 0) & (w[:, 2] > 0)
    
    s = np.zeros(len(w))
    q = np.zeros(len(w))
    T = np.zeros(len(w))
    
    ok = pos_def & (w[:, 2] > w[:, 0])
    s[ok] = np.sqrt(np.clip(w[ok, 0] / w[ok, 2], 0, None))
    q[ok] = np.sqrt(np.clip(w[ok, 1] / w[ok, 2], 0, None))
    T[ok] = (w[ok, 2] - w[ok, 1]) / (w[ok, 2] - w[ok, 0])
    
    valid = ok & np.isfinite(s) & np.isfinite(q) & np.isfinite(T) & (s <= q) & (q <= 1.0) & (T >= 0.0) & (T <= 1.0)
    return valid, w, v, s, q, T

class GlobalDataManager:
    def __init__(self, config, file_index=0, custom_path=None):
        self.config = config
        data = read_catalog_data(file_index, config, custom_path=custom_path)
        if data is None:
            raise FileNotFoundError("Catalog data not found.")
            
        print("Data loaded. Extracting columns and clearing original buffer to save memory...")
        self.row_index = np.arange(len(data), dtype=np.int64)
        self.particles = data[:, 0].astype(np.float64, copy=True)
        self.mass = self.particles * config.particle_mass
        self.pos = (
            data[:, 1:4].astype(np.float64, copy=True) * config.coord_scale
        ) % config.box_size
        self.pos_q = (
            data[:, 7:10].astype(np.float64, copy=True) * config.coord_scale
        ) % config.box_size
        
        # Cosmic Web (T-web)
        self.lambda_tide = data[:, 82:85].astype(np.float64, copy=True)
        
        xx_flat = data[:, 34:43].astype(np.float64, copy=True)
        qq_flat = data[:, 43:52].astype(np.float64, copy=True)
        del data # Free memory

        # Only the eigensystem (axis ratios + eigenvectors) is consumed
        # downstream; the symmetrized 3x3 tensors themselves were stored but
        # never read. Dropping them saves 2 * N * 9 float64 per cached dataset
        # and one redundant symmetrization pass per tensor.
        self.val_x, self.w_x, self.v_x, self.s_x, self.q_x, self.T_x = get_shape_properties(xx_flat)
        self.val_q, self.w_q, self.v_q, self.s_q, self.q_q, self.T_q = get_shape_properties(qq_flat)
        del xx_flat, qq_flat

        self.max_particles = float(np.max(self.particles))
        self.min_particles = float(np.min(self.particles))
        self.box_size = config.box_size

GLOBAL_CONFIG = None

@lru_cache(maxsize=4)
def load_cached_data(data_prefix, file_index, custom_path=None, custom_dir=None):
    if GLOBAL_CONFIG is None:
        return None
    cfg = copy.copy(GLOBAL_CONFIG)
    if custom_path:
        cfg.data_dir = os.path.dirname(custom_path)
        cfg.data_prefix = ""
        if os.path.isfile(custom_path) and _custom_catalog_too_large(custom_path):
            size_mib = os.path.getsize(custom_path) / 1024**2
            raise ValueError(
                f"custom catalog is {size_mib:.1f} MiB; the interactive "
                f"safety limit is {MAX_CUSTOM_CATALOG_BYTES / 1024**2:.0f} MiB"
            )
    elif custom_dir:
        cfg.data_dir = custom_dir
        cfg.data_prefix = ""
        custom_path = _resolve_custom_catalog_path(cfg.data_dir, file_index)
    else:
        cfg.data_prefix = data_prefix
        if data_prefix == "resE":
            cfg.data_dir = cfg.data_dir.replace("FoF_R", "FoF_E")
        else:
            cfg.data_dir = cfg.data_dir.replace("FoF_E", "FoF_R")
    try:
        return GlobalDataManager(cfg, file_index=file_index, custom_path=custom_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading {data_prefix} index {file_index}: {e}")
        return None





# ==========================================
# Dash App Layout
# ==========================================
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
# Optional: suppress callback exceptions if layout loads dynamically
app.config.suppress_callback_exceptions = True

def create_layout():
    controls = dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    dbc.Label("Data Type"),
                    dbc.RadioItems(
                        id="data-prefix",
                        options=[
                            {"label": "resE", "value": "resE"},
                            {"label": "resR", "value": "resR"}
                        ],
                        value="resR", inline=True, className="mb-3"
                    ),
                    dbc.Label("File Index (0-499)"),
                    dbc.Input(id="file-index", type="number", min=0, max=499, step=1, value=0),
                    html.Hr(className="my-3"),
                    dbc.Label("Custom Input Path (or drag file here):"),
                    dbc.Input(
                        id="custom-path-input",
                        type="text",
                        placeholder="/absolute/path/to/catalog.bin or data/.../catalog.bin",
                        className="mb-2",
                    ),
                    dbc.Label("Or Upload Binary File:"),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Drag and Drop or ", html.A("Select File")]),
                        style={
                            'width': '100%', 'height': '40px', 'lineHeight': '40px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'border-color': 'rgba(0, 242, 254, 0.4)',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0',
                            'color': '#c6d3df'
                        },
                        multiple=False,
                        accept=".bin,application/octet-stream",
                    ),
                    dbc.Button("Clear Custom Data", id="clear-custom-btn", color="warning", outline=True, size="sm", className="mt-2 w-100"),
                    html.Div(id="custom-data-status", className="text-info small mt-2")
                ],
                title="0. Data Source",
                item_id="data_source"
            ),
            dbc.AccordionItem(
                [
                    dbc.Label("Minimum Particle Count"),
                    dcc.Slider(
                        id="particle-min",
                        min=0, max=5000, step=100, value=1000,
                        marks={0: '0', 1000: '1k', 5000: '5k'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("Use slider to filter out small noisy halos.", className="text-muted mt-2")
                ],
                title="1. Population Filter",
                item_id="filter"
            ),
            dbc.AccordionItem(
                [
                    dbc.Label("Tensor Type"),
                    dbc.RadioItems(
                        id="tensor-type",
                        options=[
                            {"label": "Eulerian Shape Tensor (xx)", "value": "xx"},
                            {"label": "Lagrangian Shape Tensor (qq)", "value": "qq"}
                        ],
                        value="xx", className="mb-3"
                    ),
                    html.Small(
                        "XX uses the z=0 COM-centered Eulerian second-moment tensor. QQ uses the COM-centered Lagrangian second-moment tensor in q coordinates; its RMS axes are tensor-derived display scales, not z=0 halo radii.",
                        className="scientific-note",
                    ),
                    dbc.Label("Mesh Color Variable"),
                    dcc.Dropdown(
                        id="color-map-var",
                        options=[
                            {"label": "Cosmic Web Environment (T-web)", "value": "env"},
                            {"label": "Triaxiality (T)", "value": "T"},
                            {"label": "Axis Ratio (s = c/a)", "value": "s"},
                            {"label": "Solid Color (No Map)", "value": "none"}
                        ],
                        value="env", clearable=False, className="text-dark mb-3"
                    ),
                    dbc.Label("T-web Threshold (1e-12)"),
                    dcc.Slider(
                        id="lambda-th-slider",
                        min=0.0, max=20.0, step=0.5, value=5.0,
                        marks={0: '0', 5: '5', 10: '10', 20: '20'}
                    ),
                    html.Br(),
                    dbc.Label("Prob Cloud Sharpness (k_sig) [x 1e12]"),
                    dcc.Slider(
                        id="k-sig-slider",
                        min=0.1, max=5.0, step=0.1, value=1.0,
                        marks={0.1: '0.1', 1.0: '1', 5.0: '5'},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ],
                title="2. Physical Properties",
                item_id="props"
            ),
            dbc.AccordionItem(
                [
                    dbc.Checklist(
                        options=[
                            {"label": "Show Ellipsoids (Mesh)", "value": "mesh"},
                            {"label": "Show Filament Prob Cloud", "value": "cloud"}
                        ],
                        value=["mesh"], id="render-layers", inline=True, className="mb-3"
                    ),
                    dbc.Checklist(
                        options=[
                            {
                                "label": "Periodic Boundary Condition: wrap links, images, and cloud smoothing",
                                "value": 1,
                            }
                        ],
                        value=[1],
                        id="pbc-toggle",
                        switch=True,
                        className="mb-3",
                    ),
                    dbc.Label("Cloud Smoothing Sigma (Mpc/h)"),
                    dcc.Slider(
                        id="cloud-sigma-slider",
                        min=0.0, max=5.0, step=0.5, value=1.5,
                        marks={0: "0", 1.5: "1.5", 5: "5"},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    dbc.Label("Cloud Iso-Min Density"),
                    dcc.Slider(
                        id="cloud-isomin-slider",
                        min=0.01, max=0.5, step=0.01, value=0.05,
                        marks={0.01: "0.01", 0.1: "0.1", 0.5: "0.5"},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    dbc.Label("Filament Rendering Method"),
                    dbc.RadioItems(
                        id="filament-method",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "MST Only", "value": "MST"},
                            {"label": "T-web Nodes Only", "value": "T-web"},
                            {"label": "MST + T-web (Hybrid)", "value": "all"}
                        ],
                        value="MST", inline=True, className="mb-3"
                    ),
                    dbc.Label("MST Edge Max Dist (Mpc/h)"),
                    dcc.Slider(
                        id="mst-th-slider",
                        min=0.5, max=10.0, step=0.1, value=4.0,
                        marks={2: "2", 4: "4", 6: "6"},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    dbc.Checklist(
                        options=[{"label": "Nonlinear visual size exaggeration", "value": 1}],
                        value=[], id="log-scale-toggle", switch=True, className="mb-3"
                    ),
                    dbc.Checklist(
                        options=[{"label": "Draw Major Axis Line", "value": 1}],
                        value=[], id="draw-line-toggle", switch=True, className="mb-2"
                    ),
                    dbc.Checklist(
                        options=[{"label": "Highlight MST Connected Axis (Red)", "value": 1}],
                        value=[], id="highlight-mst-axis-toggle", switch=True, className="mb-2"
                    ),
                    dbc.Label("Shape Physical Scale (Mpc/h)"),
                    dbc.RadioItems(
                        id="shape-scale-mode",
                        options=[
                            {"label": "Principal RMS half-axes (1σ)", "value": "rms"},
                            {"label": "Uniform-density boundary assumption (√5σ)", "value": "uniform"},
                            {"label": "Fixed display length (major-axis lines only)", "value": "fixed"}
                        ],
                        value="rms", className="mb-2"
                    ),
                    html.Small(
                        "RMS is catalog-derived about the halo COM. √5σ assumes a uniform-density ellipsoid and is not a measured FoF boundary.",
                        className="scientific-note",
                    ),
                    dbc.Collapse(
                        [
                            dbc.Label("Fixed Line Length (Mpc/h)"),
                            dbc.Input(id="norm-length-input", type="number", value=2.0, min=0.1, step=0.1, className="mb-3")
                        ],
                        id="fixed-length-collapse", is_open=False
                    )
                ],
                title="3. Visual Options",
                item_id="visuals"
            )
        ],
        always_open=True,
        active_item=["data_source", "filter", "props", "visuals"]
    )

    inspector = html.Details(
        [
            html.Summary(
                "SELECTION / NONE",
                id="selection-summary-label",
                className="osd-summary-label",
            ),
            html.Div(
                [
                    html.Span("Selection inspector", className="inspector-title"),
                    html.Span("hover or click", className="inspector-hint"),
                ],
                className="inspector-heading",
            ),
            html.Div(
                "Hover a halo or filament for details. Click to select it; selecting a filament highlights its skeleton and opens the mini target view. Filament identity is an MST connected component; T-web alone is only a local class.",
                id="selection-inspector",
                className="inspector-content",
            ),
        ],
        id="selection-inspector-card",
        className="selection-inspector",
        open=False,
    )

    return html.Div([
        html.Div(
            dcc.Loading(
                dcc.Graph(
                    id="halo-plot",
                    className="halo-plot",
                    config={"displaylogo": False, "responsive": True},
                ),
                type="cube", color="#39d9c5",
                className="fullscreen-loader"
            ),
            className="plot-accessible-frame",
            role="application",
            **{
                "aria-label": (
                    "Interactive three-dimensional halo shapes and filament "
                    "components. Use pointer or touch to rotate and select."
                )
            },
        ),
        html.Div(
            [
                html.Div("Halo Structure Explorer", className="navbar-brand"),
                html.Div(
                    id="halo-count-badge",
                    className="font-weight-bold text-info",
                    role="status",
                    **{"aria-live": "polite"},
                ),
            ],
            className="hud-top-bar"
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "",
                            id="parameter-title",
                            className="parameter-title font-weight-bold ml-2",
                            style={"color": "var(--accent)"},
                        ),
                        html.Button(
                            "⚙",
                            id="toggle-sidebar-btn",
                            type="button",
                            className="btn btn-outline-info settings-icon-button-inline",
                            title="Toggle parameter controls",
                            style={"border": "none", "fontSize": "1.2rem", "padding": "0 8px"},
                            **{
                                "aria-label": "Toggle parameter controls",
                                "aria-controls": "sidebar-collapse",
                                "aria-expanded": "false",
                            },
                        ),
                    ],
                    className="parameter-header",
                ),
                dbc.Collapse(
                    controls,
                    id="sidebar-collapse",
                    is_open=False,
                    className="parameter-collapse",
                ),
                inspector,
            ],
            className="hud-left-panel",
        ),
        html.Div(
            [
                html.Details(
                    [
                        html.Summary("SCENE SUMMARY", className="summary-title", style={"cursor": "pointer", "outline": "none"}),
                        html.Div(
                            id="scene-summary",
                            role="region",
                            **{"aria-label": "Current scene summary"},
                        )
                    ],
                    className="scene-summary",
                    open=True,
                ),
                html.Details(
                    [
                        html.Summary("ACTIVE ENCODINGS", className="summary-title"),
                        html.Div(
                            id="colorbar-panel",
                            className="colorbar-panel",
                            role="region",
                            **{"aria-label": "Active visual encodings"},
                        ),
                    ],
                    id="colorbar-details",
                    className="osd-details",
                    open=True,
                ),
                html.Details(
                    [
                        html.Summary("ENVIRONMENT MIX", className="summary-title"),
                        html.Div(
                            id="mini-graph",
                            className="mini-graph",
                            title="Live T-web environment composition of the rendered halos",
                        ),
                    ],
                    id="mini-graph-details",
                    className="osd-details",
                    open=False,
                ),
            ],
            className="hud-right-panel",
        ),
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Target view", className="preview-kicker"),
                                html.Strong("No selection", id="selection-preview-title"),
                            ],
                            className="preview-heading-copy",
                        ),
                        html.Span(
                            "STANDBY",
                            id="selection-preview-status",
                            className="preview-status",
                        ),
                    ],
                    className="preview-heading",
                ),
                html.Div(
                    "Select a halo or filament",
                    id="selection-preview-scale",
                    className="preview-scale",
                ),
                        html.Canvas(
                            id="selection-preview-canvas",
                            className="selection-preview-canvas",
                            role="img",
                            tabIndex=0,
                            children="Selected object mini 3D preview",
                            **{
                                "aria-label": (
                                    "Selected object mini 3D preview. Drag to inspect, "
                                    "use arrow keys to rotate, plus or minus to zoom; "
                                    "automatic orbit resumes after five seconds."
                                ),
                                "aria-keyshortcuts": "ArrowLeft ArrowRight ArrowUp ArrowDown + - Home",
                            },
                        ),
            ],
            id="selection-preview-osd",
            className="selection-preview-osd is-hidden",
            role="region",
            **{"aria-label": "Selected object 3D preview"},
        ),
        html.Div(
            [
                html.Div("Z", className="compass-axis compass-z"),
                html.Div("Y", className="compass-axis compass-y"),
                html.Div("X", className="compass-axis compass-x"),
                html.Div(className="compass-origin"),
            ],
            id="orientation-compass",
            className="hud-compass",
            title="Camera-oriented axis compass",
        ),
        html.Div(
            "WebGL context unavailable. Controls remain active; reload after enabling hardware acceleration.",
            id="webgl-status",
            className="webgl-status is-hidden",
            role="alert",
        ),
        dcc.Store(id="custom-data-store", data=None),
    ], className="app-shell")

# ==========================================
# Dash Callbacks
# ==========================================

def get_empty_figure(box_sz=100.0):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title="X [Mpc/h]", gridcolor="#444", zerolinecolor="#444", showbackground=False, range=[0, box_sz], tickfont=dict(color="#8a99ad")),
            yaxis=dict(title="Y [Mpc/h]", gridcolor="#444", zerolinecolor="#444", showbackground=False, range=[0, box_sz], tickfont=dict(color="#8a99ad")),
            zaxis=dict(title="Z [Mpc/h]", gridcolor="#444", zerolinecolor="#444", showbackground=False, range=[0, box_sz], tickfont=dict(color="#8a99ad")),
            aspectmode='cube'
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        )
    )
    return fig


def _active_colorbar_specs(
    *,
    render_layers,
    cmap_var,
    cbar_title,
    filament_catalog,
    filament_method,
    cloud_trace_added,
    cloud_isomin=0.05,
    cloud_isomax=0.8,
):
    """Return every colour scale currently driving an on-screen element.

    The scene can colour several elements at once (mesh by a scalar field,
    filament skeletons by component size, the volumetric cloud by density), so
    the legend lists each active encoding instead of guessing one. The list
    rebuilds on every figure update, so the legend always mirrors exactly what
    is drawn and follows the displayed elements automatically.
    Order: mesh (the primary objects), then filaments, then cloud.
    """
    render_layers = render_layers or []
    specs = []

    if "mesh" in render_layers and cmap_var != "none":
        if cmap_var == "env":
            specs.append({
                "title": "Ellipsoid mesh",
                "subtitle": "Cosmic Web environment (T-web)",
                "unit": "categorical",
                "type": "categorical",
                "items": [
                    ("Void", "grey"),
                    ("Sheet", "limegreen"),
                    ("Filament", "dodgerblue"),
                    ("Knot", "red"),
                ],
            })
        elif cmap_var == "T":
            specs.append({
                "title": "Ellipsoid mesh",
                "subtitle": "Triaxiality T  (oblate → prolate)",
                "unit": "dimensionless",
                "scale_badge": "LINEAR",
                "type": "gradient",
                "gradient": "linear-gradient(90deg, #9e0142, #f46d43, #ffffbf, #66c2a5, #5e4fa2)",
                "ticks": ["0", "0.5", "1"],
            })
        elif cmap_var == "s":
            specs.append({
                "title": "Ellipsoid mesh",
                "subtitle": "Axis ratio s = c/a  (flat → round)",
                "unit": "dimensionless",
                "scale_badge": "LINEAR",
                "type": "gradient",
                "gradient": "linear-gradient(90deg, #ffffcc, #fd8d3c, #800026)",
                "ticks": ["0", "0.5", "1"],
            })

    if (
        filament_method in ["MST", "all"]
        and filament_catalog is not None
        and filament_catalog.components
    ):
        counts = np.array(
            [component.halo_count for component in filament_catalog.components],
            dtype=np.float64,
        )
        scale = filament_count_scale(counts)
        specs.append({
            "title": "Filament size",
            "subtitle": "Member halos per threshold-connected component",
            "unit": "halos / component",
            "scale_badge": scale.mode.upper(),
            "type": "gradient",
            "gradient": "linear-gradient(90deg, rgb(0, 200, 200), rgb(120, 100, 220), rgb(220, 60, 180))",
            "ticks": [f"{value:,} halos" for value in scale.tick_values],
        })

    if "cloud" in render_layers and cloud_trace_added:
        specs.append({
            "title": "Filament cloud",
            "subtitle": "Smoothed CIC weighted cell mass / peak",
            "unit": "dimensionless",
            "scale_badge": "LINEAR",
            "type": "gradient",
            "gradient": "linear-gradient(90deg, #0d0887, #9c179e, #ed7953, #f0f921)",
            "ticks": [
                f"{float(cloud_isomin):.2f}",
                f"{0.5 * (float(cloud_isomin) + float(cloud_isomax)):.2f}",
                f"{float(cloud_isomax):.2f} of peak",
            ],
        })

    return specs


def _colorbar_card(spec):
    """One compact legend card for a single colour encoding."""
    head = [
        html.Div(
            [
                html.Div(spec["title"], className="colorbar-title"),
                html.Span(spec["scale_badge"], className="colorbar-scale-badge")
                if spec.get("scale_badge")
                else None,
            ],
            className="colorbar-heading",
        ),
        html.Div(spec["subtitle"], className="colorbar-subtitle"),
    ]
    if spec["type"] == "categorical":
        body = [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(className="color-swatch", style={"background": color}),
                            html.Span(label),
                        ],
                        className="colorbar-category",
                    )
                    for label, color in spec["items"]
                ],
                className="colorbar-categories",
            )
        ]
    else:  # gradient
        ticks = [html.Span(label) for label in spec["ticks"]]
        body = [
            html.Div(className="colorbar-gradient", style={"background": spec["gradient"]}),
            html.Div(ticks, className="colorbar-range"),
        ]
    unit = spec.get("unit")
    foot = [html.Div(f"Unit: {unit}", className="colorbar-unit")] if unit else []
    return html.Div(head + body + foot, className="colorbar-card")


def _colorbar_panel_children(specs):
    if not specs:
        return [
            html.Div("Solid colour", className="colorbar-title"),
            html.Div("No active scalar map", className="colorbar-subtitle"),
        ]
    return [_colorbar_card(spec) for spec in specs]


def _mini_graph_children(env_values, count):
    """A lightweight environment-composition bar chart (no extra WebGL).

    Shows how the rendered halos split across the four T-web classes so the
    right HUD carries live information instead of a decorative spinner.
    """
    env_values = np.asarray(env_values)
    classes = [
        ("Void", 0, "grey"),
        ("Sheet", 1, "limegreen"),
        ("Filament", 2, "dodgerblue"),
        ("Knot", 3, "red"),
    ]
    counts = [int(np.sum(env_values == value)) for _, value, _ in classes]
    peak = max(counts) if counts else 0
    total = max(int(count), 1)
    bars = []
    for (label, _value, color), n in zip(classes, counts):
        width_pct = (n / peak * 100.0) if peak > 0 else 0.0
        pct = n / total * 100.0
        bars.append(
            html.Div(
                [
                    html.Span(label, className="mini-bar-label"),
                    html.Div(
                        html.Div(
                            className="mini-bar-fill",
                            style={"width": f"{width_pct:.1f}%", "background": color},
                        ),
                        className="mini-bar-track",
                    ),
                    html.Span(f"{pct:.1f}%", className="mini-bar-value"),
                ],
                className="mini-bar-row",
            )
        )
    return [
        html.Div(bars, className="mini-bars"),
    ]


def _summary_chip(label, value):
    return html.Div(
        [html.Span(label, className="summary-label"), html.Strong(value)],
        className="summary-chip",
    )


def _scene_summary_children(
    *,
    count,
    data_prefix,
    file_index,
    data_label=None,
    tensor_val,
    render_layers,
    pbc_enabled,
    cmap_var,
    cbar_title,
    mesh_image_count,
    mesh_drawn_count,
    filament_catalog,
    filament_method,
    mst_th,
    cloud_trace_added,
    pbc_crossing_edges=0,
    filament_warning=None,
):
    source_label = data_label or f"{data_prefix}{file_index}"
    layer_names = []
    if "mesh" in render_layers:
        layer_names.append("mesh")
    if filament_method in ["MST", "all"]:
        layer_names.append("filaments")
    elif filament_method == "T-web":
        layer_names.append("T-web nodes")
    if "cloud" in render_layers:
        layer_names.append("cloud")
    if not layer_names:
        layer_names.append("hover centers")

    filament_count = 0
    filament_members = 0
    if filament_catalog is not None:
        filament_count = len(filament_catalog.components)
        filament_members = sum(
            component.halo_count for component in filament_catalog.components
        )

    color_encodings = []
    if "mesh" in render_layers:
        color_encodings.append(cbar_title if cmap_var != "none" else "solid mesh")
    if filament_method in ["MST", "all"] and filament_count:
        count_scale = filament_count_scale(
            np.array(
                [component.halo_count for component in filament_catalog.components],
                dtype=np.float64,
            )
        )
        color_encodings.append(f"filaments {count_scale.mode} member count")
    if "cloud" in render_layers:
        color_encodings.append("cloud relative weighted cell mass")

    cloud_status = "off"
    if "cloud" in render_layers:
        cloud_status = "shown" if cloud_trace_added else "below iso-min"

    chips = [
        _summary_chip("Data", f"{source_label} · {tensor_val.upper()}"),
        _summary_chip("Halos", f"{count:,}"),
        _summary_chip("Layers", ", ".join(layer_names)),
        _summary_chip("PBC", "ON" if pbc_enabled else "OFF"),
        _summary_chip(
            "Mesh images",
            f"{mesh_drawn_count:,}/{mesh_image_count:,}"
            if "mesh" in render_layers
            else "off",
        ),
        _summary_chip("Filaments", f"{filament_count:,} ({filament_members:,} halos)"),
        _summary_chip("PBC links", f"{int(pbc_crossing_edges):,}" if pbc_enabled else "n/a"),
        _summary_chip("MST link", f"≤{float(mst_th):g} Mpc/h"),
        _summary_chip("Cloud", cloud_status),
    ]
    footnote = (
        "Encodings: " + " · ".join(color_encodings)
        if color_encodings
        else "Encodings: none"
    )
    if filament_warning:
        footnote = f"Warning: {filament_warning} · {footnote}"
    return [
        html.Div(
            chips,
            className="summary-grid",
        ),
        html.Div(footnote, className="summary-footnote"),
    ]

@app.callback(
    Output("fixed-length-collapse", "is_open"),
    [Input("shape-scale-mode", "value")]
)
def toggle_fixed_length_input(mode):
    return mode == "fixed"

@app.callback(
    [Output("sidebar-collapse", "is_open"),
     Output("parameter-title", "children"),
     Output("toggle-sidebar-btn", "aria-expanded")],
    [Input("toggle-sidebar-btn", "n_clicks")],
    [State("sidebar-collapse", "is_open")]
)
def toggle_sidebar(n, is_open):
    if n:
        is_open = not is_open
    return is_open, "Parameters" if is_open else "", "true" if is_open else "false"


# Object selection (filament-line highlight + mini target preview) is handled
# entirely client-side in assets/filament_hover.js. The earlier server-side
# approach added a second `allow_duplicate` callback writing `halo-plot.figure`
# through a Dash Patch, which spammed "Cannot patch undefined" on every initial
# render. The client-side path restyles only the single clicked filament (O(1))
# and deliberately avoids main-scene camera relayouts, so no extra figure
# callback, bridge store, or Patch is needed here.


@app.callback(
    [Output("custom-data-store", "data"),
     Output("custom-data-status", "children")],
    [Input("custom-path-input", "value"),
     Input("upload-data", "contents"),
     Input("clear-custom-btn", "n_clicks")],
    [State("upload-data", "filename"),
     State("file-index", "value"),
     State("custom-data-store", "data")]
)
def handle_custom_data(
    custom_path, upload_contents, clear_clicks, filename, active_file_index,
    previous_custom_data
):
    import dash
    from dash import callback_context as ctx

    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger == "clear-custom-btn":
        _cleanup_custom_catalog_dir(previous_custom_data)
        load_cached_data.cache_clear()
        return None, ""

    if trigger == "upload-data" and upload_contents:
        tmp_dir = None
        try:
            if not _is_bin_catalog_name(filename):
                return dash.no_update, "Catalog upload must be a .bin file."
            _content_type, content_string = upload_contents.split(',', 1)
            decoded_size = (len(content_string) * 3) // 4
            if decoded_size > MAX_CUSTOM_CATALOG_BYTES:
                return (
                    dash.no_update,
                    f"Upload exceeds the {MAX_CUSTOM_CATALOG_BYTES / 1024**2:.0f} MiB "
                    "interactive safety limit.",
                )
            decoded = base64.b64decode(content_string, validate=True)
            tmp_dir, filepath = _prepare_custom_catalog_dir()
            with open(filepath, "wb") as f:
                f.write(decoded)
            row_count = _validate_custom_catalog_path(filepath)
            source_name = filename or "uploaded catalog"
            _cleanup_custom_catalog_dir(previous_custom_data)
            load_cached_data.cache_clear()
            return {
                "data_dir": tmp_dir,
                "file_index": 0,
                "custom_path": filepath,
                "label": f"Upload · {source_name}",
                "temporary_dir": tmp_dir,
            }, f"Loaded uploaded file: {source_name} ({row_count:,} catalog rows)"
        except Exception as e:
            if tmp_dir is not None:
                _cleanup_custom_catalog_dir({"temporary_dir": tmp_dir})
            return dash.no_update, f"Error processing upload: {str(e)}"

    if trigger == "custom-path-input" and custom_path:
        raw_custom_path = custom_path.strip().strip("'\"")
        custom_path, checked_paths = _resolve_existing_custom_input_path(raw_custom_path)
        if os.path.isfile(custom_path) and not _is_bin_catalog_name(custom_path):
            return dash.no_update, "Catalog path must point to a .bin file."
        if os.path.exists(custom_path):
            if os.path.isfile(custom_path):
                tmp_dir = None
                try:
                    if _custom_catalog_too_large(custom_path):
                        size_mib = os.path.getsize(custom_path) / 1024**2
                        return (
                            dash.no_update,
                            f"Catalog is {size_mib:.1f} MiB; the interactive "
                            f"safety limit is {MAX_CUSTOM_CATALOG_BYTES / 1024**2:.0f} MiB.",
                        )
                    row_count = _validate_custom_catalog_path(custom_path)
                    tmp_dir, filepath = _prepare_custom_catalog_dir()
                    os.symlink(custom_path, filepath)
                    source_name = os.path.basename(custom_path)
                    _cleanup_custom_catalog_dir(previous_custom_data)
                    load_cached_data.cache_clear()
                    return {
                        "data_dir": tmp_dir,
                        "file_index": 0,
                        "custom_path": filepath,
                        "label": f"Custom · {source_name}",
                        "temporary_dir": tmp_dir,
                    }, f"Loaded path: {source_name} ({row_count:,} catalog rows)"
                except Exception as e:
                    if tmp_dir is not None:
                        _cleanup_custom_catalog_dir({"temporary_dir": tmp_dir})
                    return dash.no_update, f"Failed to symlink path: {str(e)}"
            elif os.path.isdir(custom_path):
                # The user pasted a directory containing CATALOG, 499/CATALOG,
                # resE499/CATALOG, resR499/CATALOG, etc. Validate the currently
                # selected index instead of blindly probing index 0; otherwise a
                # directory holding only resE499 is incorrectly rejected before
                # update_plot() can use the file-index input.
                try:
                    index = int(active_file_index) if active_file_index is not None else 0
                    selected_catalog = _resolve_custom_catalog_path(custom_path, index)
                    row_count = _validate_custom_catalog_path(selected_catalog)
                    source_name = os.path.basename(os.path.normpath(custom_path))
                    _cleanup_custom_catalog_dir(previous_custom_data)
                    load_cached_data.cache_clear()
                    return {
                        "data_dir": custom_path,
                        "file_index": 0,
                        "custom_path": None,
                        "label": f"Directory · {source_name}",
                    }, (
                        f"Loaded directory: {source_name} "
                        f"(index {index}: {row_count:,} rows)"
                    )
                except Exception as exc:
                    return dash.no_update, f"Invalid catalog directory: {exc}"
        checked = ", ".join(checked_paths[:2])
        return dash.no_update, f"Path does not exist. Checked: {checked}"

    return dash.no_update, dash.no_update


@app.callback(
    [Output("halo-plot", "figure"),
     Output("halo-count-badge", "children"),
     Output("scene-summary", "children"),
     Output("colorbar-panel", "children"),
     Output("mini-graph", "children")],
    [Input("data-prefix", "value"),
     Input("file-index", "value"),
     Input("tensor-type", "value"),
     Input("particle-min", "value"),
     Input("color-map-var", "value"),
     Input("lambda-th-slider", "value"),
     Input("mst-th-slider", "value"),
     Input("k-sig-slider", "value"),
     Input("log-scale-toggle", "value"),
     Input("draw-line-toggle", "value"),
     Input("highlight-mst-axis-toggle", "value"),
     Input("shape-scale-mode", "value"),
     Input("norm-length-input", "value"),
     Input("render-layers", "value"),
     Input("pbc-toggle", "value"),
     Input("filament-method", "value"),
     Input("cloud-sigma-slider", "value"),
     Input("cloud-isomin-slider", "value"),
     Input("custom-data-store", "data")]
)
def update_plot(data_prefix, file_index, tensor_val, p_min, cmap_var, lambda_th, mst_th, k_sig_val, log_scale_on, draw_line_on, highlight_axis_on, shape_scale_mode, norm_len, render_layers, pbc_toggle, filament_method, cloud_sigma, cloud_isomin, custom_data):
    try:
        custom_source = None
        data_label = None
        if custom_data is not None:
            custom_source = custom_data
            if custom_data.get("custom_path"):
                file_index = custom_data.get("file_index", 0)
            data_label = custom_data.get("label") or "Custom input"
        return _update_plot_core(data_prefix, file_index, tensor_val, p_min, cmap_var, lambda_th, mst_th, k_sig_val, log_scale_on, draw_line_on, highlight_axis_on, shape_scale_mode, norm_len, render_layers, pbc_toggle, filament_method, cloud_sigma, cloud_isomin, custom_source, data_label)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return get_empty_figure(), f"Error: {str(e)}", "No scene summary", "", ""


def _scene_uirevision(
    data_prefix, file_index, tensor_val, p_min, cmap_var, lambda_th, mst_th,
    k_sig_val, log_scale_on, draw_line_on, highlight_axis_on, shape_scale_mode,
    norm_len, render_layers, pbc_toggle, filament_method, cloud_sigma,
    cloud_isomin, data_label,
):
    """Reset stale focus ranges whenever any rendered scene input changes."""
    return repr((
        data_label or f"{data_prefix}{int(file_index)}",
        tensor_val,
        p_min,
        cmap_var,
        lambda_th,
        mst_th,
        k_sig_val,
        tuple(log_scale_on or []),
        tuple(draw_line_on or []),
        tuple(highlight_axis_on or []),
        shape_scale_mode,
        norm_len,
        tuple(render_layers or []),
        tuple(pbc_toggle or []),
        filament_method,
        cloud_sigma,
        cloud_isomin,
    ))

def _update_plot_core(data_prefix, file_index, tensor_val, p_min, cmap_var, lambda_th, mst_th, k_sig_val, log_scale_on, draw_line_on, highlight_axis_on, shape_scale_mode, norm_len, render_layers, pbc_toggle, filament_method, cloud_sigma, cloud_isomin, custom_dir=None, data_label=None):
    file_index = int(file_index) if file_index is not None else 0
    # The store payload is either a dict (uploaded/symlinked file carries an
    # explicit custom_path; a pasted directory carries only data_dir) or None.
    # When custom_path is absent, load_cached_data falls back to directory logic.
    custom_path = custom_dir.get("custom_path") if isinstance(custom_dir, dict) else None
    c_dir = custom_dir.get("data_dir") if isinstance(custom_dir, dict) else custom_dir

    g_data = load_cached_data(data_prefix, file_index, custom_path=custom_path, custom_dir=c_dir)
    if g_data is None:
        return get_empty_figure(), "No Data", "No data loaded", "", ""
        
    # Tensor selection
    if tensor_val == "xx":
        val, w, v, s, q, T = g_data.val_x, g_data.w_x, g_data.v_x, g_data.s_x, g_data.q_x, g_data.T_x
        f_pos_all = g_data.pos
    else:
        val, w, v, s, q, T = g_data.val_q, g_data.w_q, g_data.v_q, g_data.s_q, g_data.q_q, g_data.T_q
        f_pos_all = g_data.pos_q
        
    # Filter
    p_min_val = p_min if p_min is not None else 0
    mask, candidate_count = select_scene_halos(
        val & (g_data.particles >= p_min_val),
        g_data.particles,
        max_halos=MAX_SCENE_HALOS,
    )
    count = np.sum(mask)
    if count == 0:
        return get_empty_figure(g_data.box_size), "0 Halos", "No halos pass the filter", "", ""
        
    f_pos = f_pos_all[mask]
    f_mass = g_data.mass[mask]
    f_w = w[mask]
    f_v = v[mask]
    f_s = s[mask]
    f_q = q[mask]
    f_T = T[mask]
    f_particles = g_data.particles[mask]
    f_rows = g_data.row_index[mask]
    
    # Convert the integer slider value back to the physical 1e-12 scale
    real_lambda_th = lambda_th * 1e-12
    real_k_sig = k_sig_val * 1e12
    
    # Compute T-web environment and continuous filament soft weight.
    f_env = get_environment(g_data.lambda_tide[mask], threshold=real_lambda_th)
    P_fil = get_filament_probability(g_data.lambda_tide[mask], threshold=real_lambda_th, k_sig=real_k_sig)

    is_log_scale = bool(log_scale_on)
    boundary_mode = "uniform" if shape_scale_mode == "uniform" else "rms"
    size_model = interactive_size_model(
        f_w,
        f_particles,
        tensor_val,
        g_data.box_size,
        boundary_mode=boundary_mode,
        log_scale=is_log_scale,
    )
    L_major = size_model.rms_axes[:, 2]
    physical_L = size_model.physical_axes[:, 2]
    visual_L = size_model.rendered_axes[:, 2]
    K_scale = size_model.mesh_scale_factors
    pbc_enabled = 1 in (pbc_toggle or [])

    mesh_extents = ellipsoid_axis_aligned_extents(f_w, f_v, K_scale)
    if pbc_enabled:
        mesh_source, mesh_shifts = periodic_image_indices_and_shifts(
            f_pos,
            mesh_extents,
            g_data.box_size,
        )
    else:
        mesh_source = np.arange(len(f_pos), dtype=np.int64)
        mesh_shifts = np.zeros((len(f_pos), 3), dtype=np.float64)

    connected_indices = set()
    halo_component_ids = np.full(count, -1, dtype=np.int32)
    highlight_axis = bool(highlight_axis_on)
    filament_catalog = None
    filament_definition = None
    filament_warning = None

    if filament_method in ["MST", "all"] or highlight_axis:
        if filament_method == "all":
            skel_mask = f_env >= 2
            skel_idx = np.where(skel_mask)[0]
            filament_definition = (
                "T-web env≥2 distance-threshold component with MST skeleton"
            )
        else:
            skel_idx = np.arange(count)
            filament_definition = "Distance-threshold component with MST skeleton"

        if len(skel_idx) > 1:
            try:
                filament_catalog = build_periodic_filament_catalog(
                    f_pos[skel_idx],
                    f_mass[skel_idx],
                    box_size=g_data.box_size,
                    max_distance=float(mst_th),
                    min_members=3,
                    source_indices=skel_idx,
                    periodic=pbc_enabled,
                    max_candidate_edges=MAX_FILAMENT_CANDIDATE_EDGES,
                )
                halo_component_ids[skel_idx] = filament_catalog.halo_component_ids
                connected_indices = {
                    int(index)
                    for component in filament_catalog.components
                    for index in component.source_indices
                }
            except ValueError as exc:
                if "candidate edge budget" not in str(exc):
                    raise
                filament_warning = str(exc)
        
    # Color mapping
    flat_shading = False # Always smooth shading for PBR lighting
    cbar_title = ""
    colorscale = "Viridis"
    cmin, cmax = None, None
    tickvals, ticktext = None, None
    
    if cmap_var == "env":
        intensity_val = f_env
        cbar_title = "Cosmic Web"
        colorscale = [
            [0.0, 'grey'], [0.25, 'grey'],
            [0.25, 'limegreen'], [0.5, 'limegreen'],
            [0.5, 'dodgerblue'], [0.75, 'dodgerblue'],
            [0.75, 'red'], [1.0, 'red']
        ]
        cmin, cmax = -0.5, 3.5
        tickvals = [0, 1, 2, 3]
        ticktext = ["Void", "Sheet", "Filament", "Knot"]
    elif cmap_var == "T":
        intensity_val = f_T
        cbar_title = "Triaxiality (T)"
        colorscale = "Spectral"
    elif cmap_var == "s":
        intensity_val = f_s
        cbar_title = "Axis Ratio (s)"
        colorscale = "YlOrRd"
    else: # cmap_var == "none"
        intensity_val = np.zeros_like(f_T) # Dummy array
        cbar_title = ""
        colorscale = None
        
    fig = go.Figure()
    render_layers = render_layers or []
    
    # Preserve all physical ellipsoids at adaptive tessellation. A high hard
    # cap remains only for pathological periodic-image amplification.
    num_visible_images = len(mesh_source)
    mesh_mask = select_mesh_images(
        f_particles[mesh_source], max_images=MAX_MESH_IMAGES
    )
    mesh_resolution = mesh_resolution_for_count(int(np.sum(mesh_mask)))

    # 1. Mesh Trace (Visuals only, no hover)
    if "mesh" in render_layers:
        if np.any(~mesh_mask):
            # Render fallback scatter points for halos exceeding the mesh quota
            scatter_idx = np.where(~mesh_mask)[0]
            scatter_pos = (f_pos[mesh_source] + mesh_shifts)[scatter_idx]
            scatter_colors = intensity_val[mesh_source][scatter_idx]
            marker_color = '#39d9c5' if cmap_var == 'none' else scatter_colors
            fig.add_trace(go.Scatter3d(
                x=scatter_pos[:, 0], y=scatter_pos[:, 1], z=scatter_pos[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=marker_color,
                    colorscale=colorscale,
                    cmin=cmin, cmax=cmax,
                    opacity=0.8,
                    showscale=False,
                    line=dict(width=0)
                ),
                name=f"Scatter Fallback ({len(scatter_idx):,} halos)",
                hoverinfo="none",
                showlegend=True,
                meta={"role": "mesh_fallback"},
            ))

        # Build Mesh for the permitted halos
        if np.any(mesh_mask):
            mesh_idx = np.where(mesh_mask)[0]
            mx, my, mz, mi, mj, mk, mint = build_ellipsoids_mesh(
                (f_pos[mesh_source] + mesh_shifts)[mesh_idx],
                f_w[mesh_source][mesh_idx],
                f_v[mesh_source][mesh_idx],
                intensity_val[mesh_source][mesh_idx],
                K_scale=K_scale[mesh_source][mesh_idx],
                resolution=mesh_resolution,
            )
            
            mesh_kwargs = dict(
                x=mx, y=my, z=mz, i=mi, j=mj, k=mk,
                opacity=1.0, # solid for better PBR lighting
                flatshading=flat_shading,
                hoverinfo='skip', # CRITICAL: disables massive payload
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    specular=0.4,
                    roughness=0.6,
                    fresnel=0.2
                ),
                lightposition=dict(x=100, y=100, z=100),
                name='Ellipsoids',
                meta={"role": "mesh", "base_opacity": 1.0},
            )
            
            if cmap_var == "none":
                mesh_kwargs['color'] = '#00ffcc' # Cyan
            else:
                mesh_kwargs['intensity'] = mint
                mesh_kwargs['colorscale'] = colorscale
                if cmin is not None and cmax is not None:
                    mesh_kwargs['cmin'] = cmin
                    mesh_kwargs['cmax'] = cmax
                mesh_kwargs['showscale'] = False
                
            fig.add_trace(go.Mesh3d(**mesh_kwargs))
    
    # 2. Transparent Center Markers (For Hovertext only)
    env_names = {0: "Void", 1: "Sheet", 2: "Filament", 3: "Knot"}
    halo_focus_radius = np.maximum(
        visual_L * 3.0,
        np.full(count, 0.015 * float(g_data.box_size), dtype=np.float64),
    )
    physical_preview_axes, preview_bases = halo_preview_geometry(
        size_model.physical_axes,
        f_v,
    )
    rendered_preview_axes, _ = halo_preview_geometry(
        size_model.rendered_axes,
        f_v,
    )
    halo_skeleton_summaries = _halo_skeleton_angle_summaries(
        filament_catalog,
        positions=f_pos,
        major_axes=f_v[:, :, 2],
        row_indices=f_rows,
        box_size=g_data.box_size,
        periodic=pbc_enabled,
    )
    halo_customdata = [
        [
            "halo",
            int(f_rows[idx]),
            int(f_particles[idx]),
            float(f_mass[idx]),
            tensor_val.upper(),
            env_names.get(f_env[idx], "Unknown"),
            float(P_fil[idx]),
            float(L_major[idx]),
            float(physical_L[idx]),
            float(visual_L[idx]),
            float(f_s[idx]),
            float(f_q[idx]),
            float(f_T[idx]),
            float(f_pos[idx, 0]),
            float(f_pos[idx, 1]),
            float(f_pos[idx, 2]),
            float(halo_focus_radius[idx]),
            *[float(value) for value in physical_preview_axes[idx]],
            *[float(value) for value in rendered_preview_axes[idx]],
            *[float(value) for value in preview_bases[idx].ravel()],
            size_model.size_label,
            size_model.display_transform_label,
            int(halo_skeleton_summaries[idx]["component_id"]),
            int(halo_skeleton_summaries[idx]["edge_count"]),
            str(halo_skeleton_summaries[idx]["summary"]),
        ]
        for idx in range(count)
    ]
    
    fig.add_trace(go.Scatter3d(
        x=f_pos[:, 0], y=f_pos[:, 1], z=f_pos[:, 2],
        mode='markers',
        marker=dict(
            size=HALO_HIT_TARGET_SIZE,
            color='#39d9c5',
            opacity=0.008,
            line=dict(width=0),
        ),
        text="",
        customdata=halo_customdata,
        hoverinfo='text',
        showlegend=False,
        name='Halo Info',
        meta={
            "role": "halo",
            "component_ids": halo_component_ids.tolist(),
        },
    ))
    
    # 2.5 True Volume Cloud (Filament Mass Density)
    cloud_trace_added = False
    if "cloud" in render_layers:
        # go.Volume runs marching cubes over the whole grid for every isosurface,
        # so cost scales with grid_res**3 * surface_count. A 64**3 grid with 15
        # surfaces (~3.9M cell-evaluations) hangs the WebGL renderer; 48**3 with
        # 6 surfaces is ~9x lighter and still smooth after Gaussian blur.
        grid_res = 48
        weights = f_mass * P_fil
        grid = get_cic_grid(
            f_pos, weights, g_data.box_size, grid_res, periodic=pbc_enabled
        )
        
        # Apply Gaussian smoothing to remove sharp points
        if cloud_sigma > 0:
            sigma_pix = cloud_sigma / (g_data.box_size / grid_res)
            grid = gaussian_filter(
                grid,
                sigma=sigma_pix,
                mode="wrap" if pbc_enabled else "nearest",
            )
        
        # Normalize grid for better visualization mapping
        max_val = np.max(grid)
        if max_val > 0:
            grid = grid / max_val
            
        grid_coordinates = (
            np.arange(grid_res, dtype=np.float32)
            * (g_data.box_size / grid_res)
        )
        # The box is a cube tessellated at equal resolution on every axis, so
        # the three grid-coordinate vectors are identical.
        X, Y, Z = np.meshgrid(
            grid_coordinates, grid_coordinates, grid_coordinates, indexing="ij"
        )
        
        # Flatten for plotly
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        Z_flat = Z.ravel()
        V_flat = np.asarray(grid, dtype=np.float32).ravel()
        
        if np.max(V_flat) > cloud_isomin:
            fig.add_trace(go.Volume(
                x=X_flat,
                y=Y_flat,
                z=Z_flat,
                value=V_flat,
                isomin=cloud_isomin,
                isomax=0.8,
                opacity=0.3, # overall opacity
                surface_count=6, # Fewer isosurfaces: lighter, still reads as a cloud
                colorscale='Plasma',
                showscale=False,
                name='Volume Cloud',
                meta={"role": "cloud", "encoding": "Mass-weighted filament probability density"},
            ))
            cloud_trace_added = True
            
    # 3. Filament skeleton traces. One retained MST component is one filament.
    pbc_crossing_edges = (
        count_periodic_crossing_edges(filament_catalog, g_data.box_size)
        if pbc_enabled
        else 0
    )
    if (
        filament_catalog is not None
        and filament_catalog.components
        and filament_method in ["MST", "all"]
    ):
        component_colors = filament_component_colors(filament_catalog)
        fil_traces = build_filament_traces(
            filament_catalog,
            component_colors,
            box_size=g_data.box_size,
            method_label=filament_definition,
            periodic_edges=pbc_enabled,
        )
        for idx, trace in enumerate(fil_traces):
            if idx == 0:
                # Show one legend entry for the whole filament group
                trace.showlegend = True
                trace.name = f"Filaments ({len(filament_catalog.components)} MST)"
            fig.add_trace(trace)
        member_trace = build_filament_member_trace(
            filament_catalog,
            component_colors,
            f_mass=f_mass,
            f_particles=f_particles,
            method_label=filament_definition,
            box_size=g_data.box_size,
            periodic_edges=pbc_enabled,
        )
        if member_trace is not None:
            fig.add_trace(member_trace)
        if pbc_enabled:
            continuation_trace = build_pbc_continuation_trace(
                filament_catalog,
                component_colors,
                box_size=g_data.box_size,
                method_label=filament_definition,
            )
            if continuation_trace is not None:
                fig.add_trace(continuation_trace)
                
    # 3b. T-web node highlighting
    if filament_method in ["T-web", "all"]:
        tweb_mask = (f_env >= 2)
        if tweb_mask.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=f_pos[tweb_mask, 0], y=f_pos[tweb_mask, 1], z=f_pos[tweb_mask, 2],
                mode='markers',
                marker=dict(color='red', size=4, symbol='cross', opacity=0.8),
                hoverinfo='skip',
                name='T-web Nodes'
            ))

    # 4. Major Axis Lines
    # The red MST-axis switch must be independently meaningful.  Previously
    # this whole block was gated by draw_line_on, so toggling only
    # "Highlight MST Connected Axis" changed the UI state but drew nothing.
    if bool(draw_line_on) or highlight_axis:
        line_coords_normal = []
        line_coords_highlight = []

        if shape_scale_mode == "fixed" and norm_len:
            line_half_lengths = np.full(count, norm_len / 2.0, dtype=np.float64)
        else:
            line_half_lengths = visual_L * 1.5

        line_extents = np.abs(f_v[:, :, 2]) * line_half_lengths[:, None]
        if pbc_enabled:
            line_source, line_shifts = periodic_image_indices_and_shifts(
                f_pos,
                line_extents,
                g_data.box_size,
            )
        else:
            line_source = np.arange(count, dtype=np.int64)
            line_shifts = np.zeros((count, 3), dtype=np.float64)

        for source_index, shift in zip(line_source, line_shifts):
            xc = f_pos[source_index] + shift
            v_major = f_v[source_index, :, 2]
            half_length = line_half_lengths[source_index]

            p1 = xc - half_length * v_major
            p2 = xc + half_length * v_major

            if highlight_axis and (source_index in connected_indices):
                line_coords_highlight.extend([p1, p2, [np.nan, np.nan, np.nan]])
            elif bool(draw_line_on):
                line_coords_normal.extend([p1, p2, [np.nan, np.nan, np.nan]])
                
        if len(line_coords_normal) > 0:
            arr_normal = np.array(line_coords_normal)
            fig.add_trace(go.Scatter3d(
                x=arr_normal[:, 0], y=arr_normal[:, 1], z=arr_normal[:, 2],
                mode='lines',
                line=dict(color='white', width=5),
                hoverinfo='skip',
                name='Major Axis'
            ))
            
        if len(line_coords_highlight) > 0:
            arr_highlight = np.array(line_coords_highlight)
            fig.add_trace(go.Scatter3d(
                x=arr_highlight[:, 0], y=arr_highlight[:, 1], z=arr_highlight[:, 2],
                mode='lines',
                line=dict(color='red', width=6),
                hoverinfo='skip',
                name='MST Connected Axis'
            ))

    box_sz = g_data.box_size
    axis_title_style = dict(color="#c6d3df", family="Inter, sans-serif")
    scene_revision = _scene_uirevision(
        data_prefix, file_index, tensor_val, p_min, cmap_var, lambda_th,
        mst_th, k_sig_val, log_scale_on, draw_line_on, highlight_axis_on,
        shape_scale_mode, norm_len, render_layers, pbc_toggle,
        filament_method, cloud_sigma, cloud_isomin, data_label,
    )
    fig.update_layout(
        paper_bgcolor='rgba(10, 12, 20, 1)',
        plot_bgcolor='rgba(10, 12, 20, 1)',
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
        uirevision=scene_revision,
        meta={
            "box_size": float(box_sz),
            "pbc_enabled": bool(pbc_enabled),
            "scene_revision": scene_revision,
        },
        scene=dict(
            uirevision=scene_revision,
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
            xaxis=dict(title=dict(text="X [Mpc/h]", font=axis_title_style), gridcolor="#283243", zerolinecolor="#283243", showbackground=False, range=[0, box_sz], tickfont=dict(color="#8a99ad")),
            yaxis=dict(title=dict(text="Y [Mpc/h]", font=axis_title_style), gridcolor="#283243", zerolinecolor="#283243", showbackground=False, range=[0, box_sz], tickfont=dict(color="#8a99ad")),
            zaxis=dict(title=dict(text="Z [Mpc/h]", font=axis_title_style), gridcolor="#283243", zerolinecolor="#283243", showbackground=False, range=[0, box_sz], tickfont=dict(color="#8a99ad")),
            aspectmode='cube'
        ),
        # The fullscreen scene is documented by the right-hand OSB (scene
        # summary + active colour scales), so the in-scene Plotly legend would
        # only duplicate it and sit behind the left control panel. Hide it.
        showlegend=False,
    )

    drawn_mesh_count = int(np.sum(mesh_mask))
    # Concise top-bar status. The full breakdown (layers, mesh images, filament
    # members, MST link, size model, cloud) lives in the right-hand scene
    # summary OSB, so the badge stays to one line and only flags safety caps.
    mesh_badge = (
        f"{drawn_mesh_count:,} ellipsoids" if "mesh" in render_layers else "mesh off"
    )
    badge_text = (
        f"{count:,} halos · {tensor_val.upper()} · {mesh_badge}"
        f" · PBC {'ON' if pbc_enabled else 'OFF'}"
    )
    if filament_catalog is not None:
        badge_text += f" · {len(filament_catalog.components):,} filaments"
    elif filament_method == "T-web":
        badge_text += " · T-web nodes"
    caps = []
    if candidate_count > count:
        caps.append(f"Np-cap {count:,}/{candidate_count:,}")
    if "mesh" in render_layers and np.any(~mesh_mask):
        caps.append(f"mesh-cap {MAX_MESH_IMAGES:,}")
    if filament_warning:
        caps.append("filament graph budget")
    if caps:
        badge_text += " · ⚠ " + " · ".join(caps)

    scene_summary = _scene_summary_children(
        count=count,
        data_prefix=data_prefix,
        file_index=file_index,
        data_label=data_label,
        tensor_val=tensor_val,
        render_layers=render_layers,
        pbc_enabled=pbc_enabled,
        cmap_var=cmap_var,
        cbar_title=cbar_title,
        mesh_image_count=num_visible_images,
        mesh_drawn_count=drawn_mesh_count,
        filament_catalog=filament_catalog,
        filament_method=filament_method,
        mst_th=mst_th,
        cloud_trace_added=cloud_trace_added,
        pbc_crossing_edges=pbc_crossing_edges,
        filament_warning=filament_warning,
    )
    colorbar_panel = _colorbar_panel_children(
        _active_colorbar_specs(
            render_layers=render_layers,
            cmap_var=cmap_var,
            cbar_title=cbar_title,
            filament_catalog=filament_catalog,
            filament_method=filament_method,
            cloud_trace_added=cloud_trace_added,
            cloud_isomin=cloud_isomin,
            cloud_isomax=0.8,
        )
    )
    mini_graph = _mini_graph_children(f_env, count)

    return (
        fig,
        badge_text,
        scene_summary,
        colorbar_panel,
        mini_graph,
    )

def main():
    parser = setup_common_argparse("Interactive 3D Halo Shape Visualizer (Dash)")

    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for the server")
    parser.add_argument("--port", type=int, default=8050, help="Port for the server")
    parser.add_argument("--threads", type=int, default=min(8, (os.cpu_count() or 1) * 2), help="Waitress worker threads")
    args = parser.parse_args()
    
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = get_global_config(args)


    
    # Pre-load initial dataset (data_prefix from config, index 0)
    load_cached_data(GLOBAL_CONFIG.data_prefix, 0)
    
    app.layout = create_layout()
    host, port = args.host, args.port
    print(f"Open your web browser to view the interactive 3D visualization at http://{host}:{port}/")
    try:
        # Serve through a production WSGI server (waitress) so the Flask/Werkzeug
        # "development server" warning never applies. waitress runs in-process
        # (no reloader/fork), so the 4.3 MB catalog is still loaded only once,
        # while multiple worker threads keep hover/click callbacks responsive
        # while a figure rebuild is in flight.
        from waitress import serve
        print(f"Starting production WSGI server (waitress) on http://{host}:{port} with {args.threads} threads")
        serve(app.server, host=host, port=port, threads=args.threads)
    except ImportError:
        # Last-resort fallback if waitress is not installed in the environment.
        print(
            "waitress not found; falling back to the Flask development server. "
            "Install waitress for a production server: pip install waitress"
        )
        print(f"Starting Dash development server on http://{host}:{port}")
        # debug=False prevents Werkzeug from forking and reloading the catalog twice.
        app.run(debug=False, host=host, port=port)

if __name__ == "__main__":
    main()
