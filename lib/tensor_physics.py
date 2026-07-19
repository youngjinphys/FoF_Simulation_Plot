"""Tensor geometry used by the interactive halo visualizer."""

from __future__ import annotations

from itertools import product

import numpy as np


def principal_rms_scale_factors(
    particle_counts: np.ndarray,
    tensor_type: str,
    box_size: float,
) -> np.ndarray:
    """
    Return the catalog-specific factor that converts sqrt(eigenvalue) to Mpc/h.

    Direct catalog scaling diagnostics strongly support qq as a COM-centered
    particle-averaged second moment in box-normalized coordinates, while xx
    is COM-centered and carries one additional factor of particle count. The
    COM center convention is user-confirmed; the scalar normalization is the
    catalog-consistent conversion established by binary diagnostics:

        qq: L_box * sqrt(lambda)
        xx: L_box * sqrt(lambda / Np)

    This is an RMS scale about the catalog COM, not a density-profile-dependent
    ellipsoid semi-axis.
    """
    particles = np.asarray(particle_counts, dtype=np.float64)
    if box_size <= 0.0:
        raise ValueError("box_size must be positive")
    if tensor_type == "qq":
        return np.full(particles.shape, box_size, dtype=np.float64)
    if tensor_type == "xx":
        if np.any(~np.isfinite(particles) | (particles <= 0.0)):
            raise ValueError("xx RMS conversion requires positive particle counts")
        return box_size / np.sqrt(particles)
    raise ValueError(f"unsupported tensor_type: {tensor_type!r}")


def principal_rms_lengths(
    eigenvalues: np.ndarray,
    particle_counts: np.ndarray,
    tensor_type: str,
    box_size: float,
) -> np.ndarray:
    """Convert ascending tensor eigenvalues to principal RMS lengths in Mpc/h."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("eigenvalues must have shape (N, 3)")
    particles = np.asarray(particle_counts, dtype=np.float64)
    if particles.shape != (len(values),):
        raise ValueError("particle_counts must have shape (N,)")
    scale = principal_rms_scale_factors(
        particles,
        tensor_type,
        box_size,
    )
    lengths = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0.0)
    lengths[valid] = np.sqrt(values[valid])
    lengths *= scale[:, None]
    return lengths


def ellipsoid_axis_aligned_extents(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    scale_factors: np.ndarray,
) -> np.ndarray:
    """Return each rotated ellipsoid's half-extent along the box x/y/z axes."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    scales = np.asarray(scale_factors, dtype=np.float64)
    if values.shape != (len(values), 3):
        raise ValueError("eigenvalues must have shape (N, 3)")
    if vectors.shape != (len(values), 3, 3):
        raise ValueError("eigenvectors must have shape (N, 3, 3)")
    if scales.shape != (len(values),):
        raise ValueError("scale_factors must have shape (N,)")
    axis_variance = np.maximum(values, 0.0) * scales[:, None] ** 2
    return np.sqrt(np.einsum("nij,nj->ni", vectors * vectors, axis_variance))


def periodic_image_indices_and_shifts(
    centers: np.ndarray,
    extents: np.ndarray,
    box_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return source indices and box shifts needed to render periodic images.

    The unshifted image is always included. An image shifted by +L or -L is
    added when an ellipsoid intersects the lower or upper boundary.
    """
    centers = np.asarray(centers, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    if extents.shape != centers.shape:
        raise ValueError("extents must have the same shape as centers")
    if box_size <= 0.0:
        raise ValueError("box_size must be positive")

    source_indices = []
    shifts = []
    for index, (center, extent) in enumerate(zip(centers, extents)):
        per_axis = []
        for coordinate, half_extent in zip(center, extent):
            options = [0.0]
            if coordinate - half_extent < 0.0:
                options.append(box_size)
            if coordinate + half_extent > box_size:
                options.append(-box_size)
            per_axis.append(options)
        for shift in product(*per_axis):
            source_indices.append(index)
            shifts.append(shift)

    return (
        np.asarray(source_indices, dtype=np.int64),
        np.asarray(shifts, dtype=np.float64).reshape(-1, 3),
    )
