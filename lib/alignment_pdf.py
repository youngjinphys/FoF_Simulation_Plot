"""Mass-stratified conditional PDF primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AggregatedPDF:
    mean: np.ndarray
    sem: np.ndarray
    n_files: np.ndarray
    total_support: np.ndarray


def log_mass_edges(logm_min: float, logm_max: float, step: float) -> np.ndarray:
    if not all(np.isfinite(value) for value in (logm_min, logm_max, step)):
        raise ValueError("mass limits and step must be finite")
    if logm_max <= logm_min or step <= 0.0:
        raise ValueError("mass range and step must be positive")
    intervals = (logm_max - logm_min) / step
    rounded = int(round(intervals))
    if not np.isclose(intervals, rounded, rtol=0.0, atol=1.0e-10):
        raise ValueError("mass-bin step must exactly divide the requested range")
    return np.linspace(logm_min, logm_max, rounded + 1, dtype=np.float64)


def conditional_pdf_by_mass(
    log_mass: np.ndarray,
    values: np.ndarray,
    mass_edges: np.ndarray,
    value_edges: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    log_mass = np.asarray(log_mass, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mass_edges = np.asarray(mass_edges, dtype=np.float64)
    value_edges = np.asarray(value_edges, dtype=np.float64)
    if log_mass.ndim != 1 or values.shape != log_mass.shape:
        raise ValueError("log_mass and values must have the same one-dimensional shape")
    if np.any(~np.isfinite(mass_edges)) or np.any(np.diff(mass_edges) <= 0.0):
        raise ValueError("mass_edges must be finite and strictly increasing")
    if np.any(~np.isfinite(value_edges)) or np.any(np.diff(value_edges) <= 0.0):
        raise ValueError("value_edges must be finite and strictly increasing")
    if weights is None:
        sample_weights = np.ones(len(log_mass), dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != log_mass.shape:
            raise ValueError("weights must have the same shape as log_mass")
        if np.any(np.isfinite(sample_weights) & (sample_weights < 0.0)):
            raise ValueError("weights must be non-negative")

    valid = (
        np.isfinite(log_mass)
        & np.isfinite(values)
        & np.isfinite(sample_weights)
        & (sample_weights >= 0.0)
    )
    histogram, _, _ = np.histogram2d(
        log_mass[valid],
        values[valid],
        bins=[mass_edges, value_edges],
        weights=sample_weights[valid],
    )
    support = np.sum(histogram, axis=1)
    pdf = np.full(histogram.shape, np.nan, dtype=np.float64)
    populated = support > 0.0
    pdf[populated] = histogram[populated] / (
        support[populated, None] * np.diff(value_edges)[None, :]
    )
    return pdf, support


def aggregate_realization_pdfs(
    pdfs: np.ndarray,
    supports: np.ndarray,
) -> AggregatedPDF:
    pdfs = np.asarray(pdfs, dtype=np.float64)
    supports = np.asarray(supports, dtype=np.float64)
    if pdfs.ndim != 3:
        raise ValueError("pdfs must have shape (realization, mass_bin, value_bin)")
    if supports.shape != pdfs.shape[:2]:
        raise ValueError("supports must have shape (realization, mass_bin)")

    n_mass, n_value = pdfs.shape[1:]
    mean = np.full((n_mass, n_value), np.nan, dtype=np.float64)
    sem = np.full_like(mean, np.nan)
    for mass_index in range(n_mass):
        for value_index in range(n_value):
            column = pdfs[:, mass_index, value_index]
            column = column[np.isfinite(column)]
            if len(column) == 0:
                continue
            mean[mass_index, value_index] = float(np.mean(column))
            if len(column) > 1:
                sem[mass_index, value_index] = float(
                    np.std(column, ddof=1) / np.sqrt(len(column))
                )
    valid_realization = (supports > 0.0) & np.any(np.isfinite(pdfs), axis=2)
    n_files = np.sum(valid_realization, axis=0).astype(np.int64)
    total_support = np.sum(
        np.where(np.isfinite(supports), supports, 0.0), axis=0
    )
    return AggregatedPDF(mean, sem, n_files, total_support)

