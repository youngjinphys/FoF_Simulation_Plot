"""Periodic MST filament component analysis shared by visualizations."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FilamentComponent:
    component_id: int
    member_indices: np.ndarray
    source_indices: np.ndarray
    halo_count: int
    total_mass: float
    median_mass: float
    min_mass: float
    max_mass: float
    total_edge_length: float
    bounding_box_extents: tuple[float, float, float]
    bounding_box_diagonal: float


@dataclass(frozen=True)
class FilamentCatalog:
    wrapped_positions: np.ndarray
    halo_component_ids: np.ndarray
    edges: np.ndarray
    edge_component_ids: np.ndarray
    edge_source_indices: np.ndarray
    edge_lengths: np.ndarray
    components: tuple[FilamentComponent, ...]
    max_distance: float
    excluded_halo_count: int


def _validate_inputs(
    positions: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    max_distance: float,
    min_members: int,
    source_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if masses.shape != (len(positions),):
        raise ValueError("masses must have shape (N,)")
    if np.any(~np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if np.any(~np.isfinite(masses) | (masses <= 0.0)):
        raise ValueError("masses must be finite and positive")
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_distance must be finite and positive")
    if isinstance(min_members, bool) or not isinstance(min_members, (int, np.integer)):
        raise ValueError("min_members must be an integer >= 2")
    if min_members < 2:
        raise ValueError("min_members must be an integer >= 2")

    if source_indices is None:
        sources = np.arange(len(positions), dtype=np.int64)
    else:
        sources = np.asarray(source_indices)
        if sources.shape != (len(positions),):
            raise ValueError("source_indices must have shape (N,)")
        if not np.issubdtype(sources.dtype, np.integer):
            raise ValueError("source_indices must be integers")
        sources = sources.astype(np.int64, copy=False)
    return positions, masses, sources


def _unwrap_component(
    wrapped_positions: np.ndarray,
    member_indices: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    box_size: float,
) -> np.ndarray:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for row, col in zip(edge_rows, edge_cols):
        adjacency[int(row)].append(int(col))
        adjacency[int(col)].append(int(row))

    root = int(member_indices[0])
    unwrapped = {root: wrapped_positions[root].copy()}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in unwrapped:
                continue
            displacement = wrapped_positions[neighbor] - wrapped_positions[current]
            displacement -= np.round(displacement / box_size) * box_size
            unwrapped[neighbor] = unwrapped[current] + displacement
            queue.append(neighbor)
    return np.vstack([unwrapped[int(index)] for index in member_indices])


def build_periodic_filament_catalog(
    positions: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    max_distance: float,
    min_members: int = 3,
    source_indices: np.ndarray | None = None,
    periodic: bool = True,
    max_candidate_edges: int | None = 2_000_000,
) -> FilamentCatalog:
    """Build a thresholded minimum-spanning forest and its components.

    With periodic=True, the KD-tree uses the box size and component extents are
    measured after minimum-image unwrapping. With periodic=False, the wrapped
    coordinates are treated as an open cube, so opposite faces are not linked.
    """
    positions, masses, sources = _validate_inputs(
        positions,
        masses,
        box_size,
        max_distance,
        min_members,
        source_indices,
    )
    wrapped = np.mod(positions, box_size)
    halo_component_ids = np.full(len(wrapped), -1, dtype=np.int64)
    if len(wrapped) < 2:
        return FilamentCatalog(
            wrapped,
            halo_component_ids,
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            (),
            float(max_distance),
            len(wrapped),
        )

    tree = cKDTree(wrapped, boxsize=box_size if periodic else None)
    if max_candidate_edges is not None:
        if (
            isinstance(max_candidate_edges, bool)
            or not isinstance(max_candidate_edges, (int, np.integer))
            or max_candidate_edges < 1
        ):
            raise ValueError("max_candidate_edges must be a positive integer or None")
        neighbour_counts = tree.query_ball_point(
            wrapped,
            r=max_distance,
            return_length=True,
        )
        # Each undirected pair is counted from both endpoints; every point also
        # counts itself. This cheap integer vector prevents a clustered input
        # from allocating an unbounded COO graph without altering the science
        # threshold or silently dropping neighbours.
        candidate_edges = int((np.sum(neighbour_counts, dtype=np.int64) - len(wrapped)) // 2)
        if candidate_edges > max_candidate_edges:
            raise ValueError(
                "filament candidate edge budget exceeded: "
                f"{candidate_edges:,} > {max_candidate_edges:,}; "
                "lower the MST link threshold or raise the explicit budget"
            )
    raw_graph = tree.sparse_distance_matrix(
        tree, max_distance=max_distance, output_type="coo_matrix"
    )
    # Sparse graph algorithms use numerical zero to mean "no edge". Distinct
    # halos can nevertheless share an identical centre, so retain those
    # physical zero-length links as a tiny internal weight while removing the
    # diagonal self-links. They are converted back to exactly zero below.
    off_diagonal = raw_graph.row != raw_graph.col
    graph_rows = raw_graph.row[off_diagonal]
    graph_cols = raw_graph.col[off_diagonal]
    graph_data = raw_graph.data[off_diagonal].astype(np.float64, copy=True)
    duplicate_epsilon = np.finfo(np.float64).eps * max(1.0, float(box_size))
    graph_data[graph_data == 0.0] = duplicate_epsilon
    distance_graph = coo_matrix(
        (graph_data, (graph_rows, graph_cols)),
        shape=(len(wrapped), len(wrapped)),
    ).tocsr()
    forest = minimum_spanning_tree(distance_graph)
    forest_coo = forest.tocoo()
    _, raw_labels = connected_components(forest + forest.T, directed=False)

    labels, counts = np.unique(raw_labels, return_counts=True)
    retained_labels = labels[counts >= min_members]
    retained_labels = sorted(
        retained_labels.tolist(),
        key=lambda label: int(np.flatnonzero(raw_labels == label)[0]),
    )

    components: list[FilamentComponent] = []
    all_edges: list[np.ndarray] = []
    all_edge_lengths: list[np.ndarray] = []
    all_edge_component_ids: list[np.ndarray] = []
    all_edge_source_indices: list[np.ndarray] = []
    for component_id, raw_label in enumerate(retained_labels):
        members = np.flatnonzero(raw_labels == raw_label).astype(np.int64)
        halo_component_ids[members] = component_id
        edge_mask = (
            (raw_labels[forest_coo.row] == raw_label)
            & (raw_labels[forest_coo.col] == raw_label)
        )
        rows = forest_coo.row[edge_mask].astype(np.int64)
        cols = forest_coo.col[edge_mask].astype(np.int64)
        edge_lengths = forest_coo.data[edge_mask].astype(np.float64)
        edge_lengths[edge_lengths <= duplicate_epsilon] = 0.0
        edges = np.stack([wrapped[rows], wrapped[cols]], axis=1)
        edge_source_indices = np.stack([sources[rows], sources[cols]], axis=1)

        if periodic:
            unwrapped = _unwrap_component(wrapped, members, rows, cols, box_size)
        else:
            unwrapped = wrapped[members]
        extents_array = np.ptp(unwrapped, axis=0)
        component_masses = masses[members]
        components.append(
            FilamentComponent(
                component_id=component_id,
                member_indices=members,
                source_indices=sources[members],
                halo_count=len(members),
                total_mass=float(np.sum(component_masses)),
                median_mass=float(np.median(component_masses)),
                min_mass=float(np.min(component_masses)),
                max_mass=float(np.max(component_masses)),
                total_edge_length=float(np.sum(edge_lengths)),
                bounding_box_extents=tuple(float(value) for value in extents_array),
                bounding_box_diagonal=float(np.linalg.norm(extents_array)),
            )
        )
        all_edges.append(edges)
        all_edge_lengths.append(edge_lengths)
        all_edge_source_indices.append(edge_source_indices.astype(np.int64, copy=False))
        all_edge_component_ids.append(
            np.full(len(edges), component_id, dtype=np.int64)
        )

    edges_out = (
        np.concatenate(all_edges, axis=0)
        if all_edges
        else np.empty((0, 2, 3), dtype=np.float64)
    )
    edge_lengths_out = (
        np.concatenate(all_edge_lengths)
        if all_edge_lengths
        else np.empty(0, dtype=np.float64)
    )
    edge_ids_out = (
        np.concatenate(all_edge_component_ids)
        if all_edge_component_ids
        else np.empty(0, dtype=np.int64)
    )
    edge_source_indices_out = (
        np.concatenate(all_edge_source_indices, axis=0)
        if all_edge_source_indices
        else np.empty((0, 2), dtype=np.int64)
    )
    return FilamentCatalog(
        wrapped_positions=wrapped,
        halo_component_ids=halo_component_ids,
        edges=edges_out,
        edge_component_ids=edge_ids_out,
        edge_source_indices=edge_source_indices_out,
        edge_lengths=edge_lengths_out,
        components=tuple(components),
        max_distance=float(max_distance),
        excluded_halo_count=int(np.sum(halo_component_ids < 0)),
    )
