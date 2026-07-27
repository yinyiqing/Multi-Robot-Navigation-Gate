import math
from dataclasses import dataclass

import numpy as np


DEFAULT_HORIZONTAL_FOV = (-math.pi / 2.0, math.pi / 2.0)
DEFAULT_VERTICAL_FOV = (math.radians(-15.0), math.radians(15.0))


@dataclass(frozen=True)
class RangeView:
    ranges: np.ndarray
    heights: np.ndarray
    valid: np.ndarray
    horizontal_fov: tuple
    max_range: float

    @property
    def shape(self):
        return self.ranges.shape


def _validate_points(points):
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("points must have shape [N, 3+]")
    return values[:, :3]


def project_range_view(
    points,
    vertical_bins=16,
    horizontal_bins=360,
    horizontal_fov=DEFAULT_HORIZONTAL_FOV,
    vertical_fov=DEFAULT_VERTICAL_FOV,
    max_range=6.0,
):
    """Project unordered XYZ points into the native VLP-16 angular grid."""
    if vertical_bins < 1 or horizontal_bins < 1:
        raise ValueError("range-view dimensions must be positive")
    horizontal_min, horizontal_max = map(float, horizontal_fov)
    vertical_min, vertical_max = map(float, vertical_fov)
    if horizontal_max <= horizontal_min or vertical_max <= vertical_min:
        raise ValueError("field-of-view bounds are invalid")
    if max_range <= 0.0:
        raise ValueError("max_range must be positive")

    values = _validate_points(points)
    ranges_image = np.zeros((vertical_bins, horizontal_bins), dtype=np.float32)
    heights_image = np.zeros_like(ranges_image)
    valid_image = np.zeros_like(ranges_image, dtype=bool)
    if len(values) == 0:
        return RangeView(
            ranges_image,
            heights_image,
            valid_image,
            (horizontal_min, horizontal_max),
            float(max_range),
        )

    finite = np.all(np.isfinite(values), axis=1)
    planar_ranges = np.linalg.norm(values[:, :2], axis=1)
    ranges = np.linalg.norm(values, axis=1)
    azimuths = np.arctan2(values[:, 1], values[:, 0])
    elevations = np.arctan2(values[:, 2], np.maximum(planar_ranges, 1e-6))
    keep = (
        finite
        & (planar_ranges > 1e-6)
        & (ranges <= max_range)
        & (azimuths >= horizontal_min)
        & (azimuths <= horizontal_max)
        & (elevations >= vertical_min)
        & (elevations <= vertical_max)
    )
    if not np.any(keep):
        return RangeView(
            ranges_image,
            heights_image,
            valid_image,
            (horizontal_min, horizontal_max),
            float(max_range),
        )

    values = values[keep]
    ranges = ranges[keep]
    azimuths = azimuths[keep]
    elevations = elevations[keep]
    columns = np.floor(
        (azimuths - horizontal_min)
        / (horizontal_max - horizontal_min)
        * horizontal_bins
    ).astype(np.int64)
    rows = np.floor(
        (elevations - vertical_min)
        / (vertical_max - vertical_min)
        * vertical_bins
    ).astype(np.int64)
    columns = np.clip(columns, 0, horizontal_bins - 1)
    rows = np.clip(rows, 0, vertical_bins - 1)

    flat_indices = rows * horizontal_bins + columns
    nearest_order = np.argsort(ranges, kind="stable")
    ordered_flat = flat_indices[nearest_order]
    _, first_positions = np.unique(ordered_flat, return_index=True)
    selected = nearest_order[first_positions]
    selected_rows = rows[selected]
    selected_columns = columns[selected]
    ranges_image[selected_rows, selected_columns] = ranges[selected]
    heights_image[selected_rows, selected_columns] = values[selected, 2]
    valid_image[selected_rows, selected_columns] = True
    return RangeView(
        ranges_image,
        heights_image,
        valid_image,
        (horizontal_min, horizontal_max),
        float(max_range),
    )


def extract_local_patch(
    view,
    candidate_xy,
    physical_width=1.2,
    output_width=64,
    height_scale=0.75,
):
    """Extract a range-normalized window with constant physical width."""
    if not isinstance(view, RangeView):
        raise TypeError("view must be a RangeView")
    if physical_width <= 0.0 or output_width < 1 or height_scale <= 0.0:
        raise ValueError("patch dimensions and scales must be positive")
    candidate = np.asarray(candidate_xy, dtype=np.float32)
    if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
        raise ValueError("candidate_xy must contain finite [x, y]")
    candidate_range = float(np.linalg.norm(candidate))
    if candidate_range <= 1e-6:
        raise ValueError("candidate must be away from the lidar origin")

    center_angle = math.atan2(float(candidate[1]), float(candidate[0]))
    half_angle = math.atan2(physical_width * 0.5, candidate_range)
    sample_angles = np.linspace(
        center_angle - half_angle,
        center_angle + half_angle,
        output_width,
        dtype=np.float32,
    )
    horizontal_min, horizontal_max = view.horizontal_fov
    source_width = view.ranges.shape[1]
    source_positions = (
        (sample_angles - horizontal_min)
        / (horizontal_max - horizontal_min)
        * source_width
    )
    source_columns = np.floor(source_positions).astype(np.int64)
    in_bounds = (source_columns >= 0) & (source_columns < source_width)
    clipped_columns = np.clip(source_columns, 0, source_width - 1)

    ranges = view.ranges[:, clipped_columns].copy()
    heights = view.heights[:, clipped_columns].copy()
    valid = view.valid[:, clipped_columns].copy()
    valid[:, ~in_bounds] = False
    relative_depth = np.zeros_like(ranges, dtype=np.float32)
    relative_depth[valid] = np.clip(
        (ranges[valid] - candidate_range) / physical_width,
        -1.0,
        1.0,
    )
    normalized_height = np.zeros_like(heights, dtype=np.float32)
    normalized_height[valid] = np.clip(
        heights[valid] / height_scale,
        -1.0,
        1.0,
    )
    return np.stack(
        (relative_depth, normalized_height, valid.astype(np.float32)), axis=0
    ).astype(np.float32)
