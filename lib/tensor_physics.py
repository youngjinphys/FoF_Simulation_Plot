"""Shared tensor geometry and scale-invariant TTT operations."""

from __future__ import annotations

from itertools import product

import numpy as np


def positions_to_physical(
    coordinates: np.ndarray,
    box_size: float,
    grid_size: int,
) -> np.ndarray:
    """Convert catalog positions from grid coordinates to Mpc/h."""
    if box_size <= 0.0 or grid_size <= 0:
        raise ValueError("box_size and grid_size must be positive")
    return np.asarray(coordinates, dtype=np.float64) * (box_size / grid_size)


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


def determinant_normalized_covariance(
    tensors: np.ndarray,
    target_geometric_mean_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preserve tensor shape while setting the geometric-mean Gaussian sigma.

    Multiplying the input tensor by any positive scalar leaves the result
    unchanged. Invalid or non-positive-definite inputs are returned as NaN.
    """
    tensor = np.asarray(tensors, dtype=np.float64)
    targets = np.asarray(target_geometric_mean_sigma, dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
        raise ValueError("tensors must have shape (N, 3, 3)")
    if targets.shape != (len(tensor),):
        raise ValueError("target_geometric_mean_sigma must have shape (N,)")

    symmetric = 0.5 * (tensor + np.swapaxes(tensor, 1, 2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    valid = (
        np.all(np.isfinite(eigenvalues), axis=1)
        & np.all(eigenvalues > 0.0, axis=1)
        & np.isfinite(targets)
        & (targets > 0.0)
    )
    covariance = np.full_like(symmetric, np.nan, dtype=np.float64)
    if np.any(valid):
        determinant_cuberoot = np.prod(eigenvalues[valid], axis=1) ** (1.0 / 3.0)
        scale_squared = targets[valid] ** 2 / determinant_cuberoot
        covariance[valid] = symmetric[valid] * scale_squared[:, None, None]
    return covariance, valid


def scale_invariant_ttt_direction(
    tidal_tensors: np.ndarray,
    second_moment_tensors: np.ndarray,
    relative_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the TTT torque direction after per-halo Frobenius normalization.

    Positive scalar normalizations of either input tensor do not affect the
    torque-like direction from the tidal tensor and catalog second-moment
    tensor. The validity threshold is therefore applied to the dimensionless
    relative commutator amplitude rather than raw catalog units.
    """
    tidal = np.asarray(tidal_tensors, dtype=np.float64)
    second_moment = np.asarray(second_moment_tensors, dtype=np.float64)
    if (
        tidal.shape != second_moment.shape
        or tidal.ndim != 3
        or tidal.shape[1:] != (3, 3)
    ):
        raise ValueError(
            "tidal_tensors and second_moment_tensors must have shape (N, 3, 3)"
        )
    if relative_eps < 0.0:
        raise ValueError("relative_eps must be non-negative")

    tidal = 0.5 * (tidal + np.swapaxes(tidal, 1, 2))
    second_moment = 0.5 * (second_moment + np.swapaxes(second_moment, 1, 2))
    tidal_norm = np.linalg.norm(tidal, axis=(1, 2))
    second_moment_norm = np.linalg.norm(second_moment, axis=(1, 2))
    finite_scale = (
        np.isfinite(tidal_norm)
        & np.isfinite(second_moment_norm)
        & (tidal_norm > 0.0)
        & (second_moment_norm > 0.0)
    )

    tidal_unit = np.zeros_like(tidal)
    second_moment_unit = np.zeros_like(second_moment)
    tidal_unit[finite_scale] = tidal[finite_scale] / tidal_norm[finite_scale, None, None]
    second_moment_unit[finite_scale] = (
        second_moment[finite_scale]
        / second_moment_norm[finite_scale, None, None]
    )

    product_ti = np.einsum("nij,njk->nik", tidal_unit, second_moment_unit)
    torque = np.column_stack([
        product_ti[:, 1, 2] - product_ti[:, 2, 1],
        product_ti[:, 2, 0] - product_ti[:, 0, 2],
        product_ti[:, 0, 1] - product_ti[:, 1, 0],
    ])
    relative_torque = np.linalg.norm(torque, axis=1)
    valid = (
        finite_scale
        & np.isfinite(relative_torque)
        & (relative_torque > relative_eps)
    )
    direction = np.zeros_like(torque)
    direction[valid] = torque[valid] / relative_torque[valid, None]
    return direction, valid, relative_torque
