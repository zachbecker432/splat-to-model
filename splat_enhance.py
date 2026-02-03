"""
Gaussian Splat Enhancement Tool

Improves Gaussian splat quality by:
1. Detecting and removing floaters/outliers
2. Plane detection and flattening (walls, floors, ceilings)
3. Local depth consistency filtering (removes reflection artifacts)
4. Surface thickness compression (condenses depth-spread splats)
5. Identifying sparse regions (potential holes)
6. Densifying sparse areas by cloning/interpolating Gaussians

The plane-fitting algorithms (--flatten-planes, --depth-filter, --compress-thickness)
are especially useful for indoor scenes where reflective surfaces cause Gaussians
to be placed at incorrect depths (e.g., mirrors, windows, glossy walls).

Usage:
    python splat_enhance.py input.ply output.ply [options]
    
    # For rooms with reflections:
    python splat_enhance.py input.ply output.ply --flatten-planes --depth-filter
"""

import argparse
import numpy as np
import time
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter, binary_dilation
from collections import defaultdict


def log_step(msg, indent=2):
    """Print a log message with consistent formatting."""
    prefix = " " * indent
    print(f"{prefix}[INFO] {msg}")


def sigmoid(x):
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(x):
    """Inverse sigmoid (logit) function."""
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


class GaussianSplat:
    """
    Represents a 3D Gaussian Splat with all its properties.
    Handles different PLY formats from various 3DGS implementations.
    """
    
    def __init__(self, ply_path=None):
        self.positions = None  # Nx3 float32
        self.colors_dc = None  # Nx3 (SH DC component)
        self.colors_rest = None  # Nx45 (remaining SH coefficients, optional)
        self.opacity = None  # Nx1 (logit space)
        self.scales = None  # Nx3 (log space)
        self.rotations = None  # Nx4 (quaternion)
        self.extra_fields = {}  # Any additional fields
        
        self._ply_format = None  # Track the format for re-export
        
        if ply_path:
            self.load(ply_path)
    
    def load(self, ply_path):
        """Load Gaussian splat from PLY file."""
        print(f"  Loading: {ply_path}")
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        names = vertex.data.dtype.names
        
        n = len(vertex)
        print(f"  Found {n:,} Gaussians")
        print(f"  Properties: {list(names)}")
        
        # Extract positions
        self.positions = np.column_stack([
            np.array(vertex['x']),
            np.array(vertex['y']),
            np.array(vertex['z'])
        ]).astype(np.float32)
        
        # Extract opacity (may be in logit or direct form)
        if 'opacity' in names:
            opacity_raw = np.array(vertex['opacity'])
            # Store in logit form internally
            if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
                self.opacity = opacity_raw.astype(np.float32)
                self._opacity_is_logit = True
            else:
                self.opacity = inverse_sigmoid(opacity_raw).astype(np.float32)
                self._opacity_is_logit = False
        
        # Extract scales (may be in log or linear form)
        if all(f'scale_{i}' in names for i in range(3)):
            scales = np.column_stack([
                np.array(vertex['scale_0']),
                np.array(vertex['scale_1']),
                np.array(vertex['scale_2'])
            ])
            # Detect if in log space
            if scales.min() < 0 or scales.max() < 1:
                self.scales = scales.astype(np.float32)
                self._scales_is_log = True
            else:
                self.scales = np.log(np.clip(scales, 1e-7, None)).astype(np.float32)
                self._scales_is_log = False
        elif all(f'scaling_{i}' in names for i in range(3)):
            scales = np.column_stack([
                np.array(vertex['scaling_0']),
                np.array(vertex['scaling_1']),
                np.array(vertex['scaling_2'])
            ])
            if scales.min() < 0 or scales.max() < 1:
                self.scales = scales.astype(np.float32)
                self._scales_is_log = True
            else:
                self.scales = np.log(np.clip(scales, 1e-7, None)).astype(np.float32)
                self._scales_is_log = False
        
        # Extract rotations (quaternion)
        if all(f'rot_{i}' in names for i in range(4)):
            self.rotations = np.column_stack([
                np.array(vertex['rot_0']),
                np.array(vertex['rot_1']),
                np.array(vertex['rot_2']),
                np.array(vertex['rot_3'])
            ]).astype(np.float32)
        
        # Extract SH DC (base color)
        if all(f'f_dc_{i}' in names for i in range(3)):
            self.colors_dc = np.column_stack([
                np.array(vertex['f_dc_0']),
                np.array(vertex['f_dc_1']),
                np.array(vertex['f_dc_2'])
            ]).astype(np.float32)
            self._ply_format = 'standard_3dgs'
        elif all(c in names for c in ['red', 'green', 'blue']):
            # Convert RGB to SH DC
            r = np.array(vertex['red'])
            g = np.array(vertex['green'])
            b = np.array(vertex['blue'])
            if r.dtype == np.uint8:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            elif r.max() > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            # Convert to SH DC: SH_DC = (color - 0.5) / SH_C0
            SH_C0 = 0.28209479177387814
            self.colors_dc = np.column_stack([
                (r - 0.5) / SH_C0,
                (g - 0.5) / SH_C0,
                (b - 0.5) / SH_C0
            ]).astype(np.float32)
            self._ply_format = 'rgb_direct'
        
        # Extract remaining SH coefficients (for view-dependent colors)
        sh_rest_fields = [f'f_rest_{i}' for i in range(45)]
        if all(f in names for f in sh_rest_fields):
            self.colors_rest = np.column_stack([
                np.array(vertex[f]) for f in sh_rest_fields
            ]).astype(np.float32)
        
        print(f"  Loaded successfully. Format: {self._ply_format}")
        return self
    
    def save(self, ply_path):
        """Save Gaussian splat to PLY file."""
        print(f"  Saving to: {ply_path}")
        
        n = len(self.positions)
        
        # Build dtype and data based on what we have
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ]
        data_dict = {
            'x': self.positions[:, 0],
            'y': self.positions[:, 1],
            'z': self.positions[:, 2],
        }
        
        # Add opacity
        if self.opacity is not None:
            dtype_list.append(('opacity', 'f4'))
            data_dict['opacity'] = self.opacity
        
        # Add scales
        if self.scales is not None:
            for i in range(3):
                dtype_list.append((f'scale_{i}', 'f4'))
                data_dict[f'scale_{i}'] = self.scales[:, i]
        
        # Add rotations
        if self.rotations is not None:
            for i in range(4):
                dtype_list.append((f'rot_{i}', 'f4'))
                data_dict[f'rot_{i}'] = self.rotations[:, i]
        
        # Add SH DC
        if self.colors_dc is not None:
            for i in range(3):
                dtype_list.append((f'f_dc_{i}', 'f4'))
                data_dict[f'f_dc_{i}'] = self.colors_dc[:, i]
        
        # Add SH rest
        if self.colors_rest is not None:
            for i in range(45):
                dtype_list.append((f'f_rest_{i}', 'f4'))
                data_dict[f'f_rest_{i}'] = self.colors_rest[:, i]
        
        # Create structured array
        vertices = np.zeros(n, dtype=dtype_list)
        for key, value in data_dict.items():
            vertices[key] = value
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(ply_path)
        print(f"  Saved {n:,} Gaussians")
    
    def get_opacity_linear(self):
        """Get opacity values in 0-1 range."""
        return sigmoid(self.opacity)
    
    def get_scales_linear(self):
        """Get scale values in linear space."""
        return np.exp(self.scales)
    
    def __len__(self):
        return len(self.positions)


def remove_floaters(splat, std_threshold=3.0, min_opacity=0.05, verbose=True):
    """
    Remove floater Gaussians that are outliers in position or have very low opacity.
    
    Args:
        splat: GaussianSplat object
        std_threshold: Remove points beyond this many standard deviations from centroid
        min_opacity: Remove Gaussians with opacity below this threshold
        verbose: Print progress
    
    Returns:
        Modified GaussianSplat object, number of removed Gaussians
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  FLOATER REMOVAL")
        print("  " + "-" * 50)
    
    n_original = len(splat)
    mask = np.ones(n_original, dtype=bool)
    
    # Remove by position outliers
    centroid = np.mean(splat.positions, axis=0)
    distances = np.linalg.norm(splat.positions - centroid, axis=1)
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    
    position_outliers = distances > (dist_mean + std_threshold * dist_std)
    mask &= ~position_outliers
    
    if verbose:
        print(f"    Position outliers (>{std_threshold} std): {np.sum(position_outliers):,}")
    
    # Remove by low opacity
    if splat.opacity is not None:
        opacity_linear = splat.get_opacity_linear()
        low_opacity = opacity_linear < min_opacity
        mask &= ~low_opacity
        
        if verbose:
            print(f"    Low opacity (<{min_opacity}): {np.sum(low_opacity):,}")
    
    # Remove by extreme scale (very large Gaussians are often artifacts)
    if splat.scales is not None:
        scales_linear = splat.get_scales_linear()
        avg_scale = np.mean(scales_linear, axis=1)
        scale_99th = np.percentile(avg_scale, 99)
        large_scale = avg_scale > scale_99th * 3
        mask &= ~large_scale
        
        if verbose:
            print(f"    Extreme scale (3x 99th percentile): {np.sum(large_scale):,}")
    
    # Apply mask
    n_removed = n_original - np.sum(mask)
    
    splat.positions = splat.positions[mask]
    if splat.opacity is not None:
        splat.opacity = splat.opacity[mask]
    if splat.scales is not None:
        splat.scales = splat.scales[mask]
    if splat.rotations is not None:
        splat.rotations = splat.rotations[mask]
    if splat.colors_dc is not None:
        splat.colors_dc = splat.colors_dc[mask]
    if splat.colors_rest is not None:
        splat.colors_rest = splat.colors_rest[mask]
    
    if verbose:
        print(f"    Total removed: {n_removed:,}")
        print(f"    Remaining: {len(splat):,}")
    
    return splat, n_removed


def find_sparse_regions(splat, grid_resolution=50, density_threshold_percentile=10, verbose=True):
    """
    Find regions where Gaussian density is low (potential holes).
    
    Uses a 3D grid to compute local density and identifies cells with
    significantly lower density than average.
    
    Args:
        splat: GaussianSplat object
        grid_resolution: Number of cells per axis
        density_threshold_percentile: Percentile below which regions are "sparse"
        verbose: Print progress
    
    Returns:
        sparse_cell_centers: Nx3 array of sparse region centers
        sparse_cell_indices: Indices into the grid
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  SPARSE REGION DETECTION")
        print("  " + "-" * 50)
    
    positions = splat.positions
    
    # Compute bounding box
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bbox_size = bbox_max - bbox_min
    
    if verbose:
        print(f"    Bounding box: {bbox_size}")
        print(f"    Grid resolution: {grid_resolution}^3")
    
    # Create density grid
    cell_size = bbox_size / grid_resolution
    
    # Compute cell indices for each Gaussian
    cell_indices = ((positions - bbox_min) / cell_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_resolution - 1)
    
    # Count Gaussians per cell
    density_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.int32)
    for idx in cell_indices:
        density_grid[idx[0], idx[1], idx[2]] += 1
    
    # Apply Gaussian smoothing to get continuous density field
    density_smoothed = gaussian_filter(density_grid.astype(float), sigma=1.5)
    
    # Find cells that are occupied but sparse
    occupied_mask = density_grid > 0
    
    # Also include neighbors of occupied cells (edge regions)
    structure = np.ones((3, 3, 3))
    near_occupied = binary_dilation(occupied_mask, structure, iterations=2)
    
    # Cells that are near occupied regions but have low density
    density_values = density_smoothed[near_occupied]
    if len(density_values) > 0:
        threshold = np.percentile(density_values[density_values > 0], density_threshold_percentile)
    else:
        threshold = 0
    
    # Sparse cells: near surface but low density
    sparse_mask = near_occupied & (density_smoothed < threshold) & (density_smoothed > 0)
    
    # Get centers of sparse cells
    sparse_indices = np.argwhere(sparse_mask)
    sparse_cell_centers = bbox_min + (sparse_indices + 0.5) * cell_size
    
    if verbose:
        print(f"    Non-empty cells: {np.sum(occupied_mask):,}")
        print(f"    Density threshold: {threshold:.2f}")
        print(f"    Sparse cells found: {len(sparse_cell_centers):,}")
    
    return sparse_cell_centers, sparse_indices


def densify_sparse_regions(splat, sparse_centers, k_neighbors=5, jitter=0.02, verbose=True):
    """
    Add new Gaussians in sparse regions by interpolating from nearby existing Gaussians.
    
    Args:
        splat: GaussianSplat object
        sparse_centers: Nx3 array of sparse region centers to fill
        k_neighbors: Number of neighbors to interpolate from
        jitter: Random offset factor (relative to local scale)
        verbose: Print progress
    
    Returns:
        Modified GaussianSplat object with additional Gaussians
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  SPARSE REGION DENSIFICATION")
        print("  " + "-" * 50)
    
    if len(sparse_centers) == 0:
        if verbose:
            print("    No sparse regions to fill")
        return splat, 0
    
    n_original = len(splat)
    
    # Build KD-tree for existing Gaussians
    tree = KDTree(splat.positions)
    
    # For each sparse center, find neighbors and interpolate
    new_positions = []
    new_opacities = []
    new_scales = []
    new_rotations = []
    new_colors_dc = []
    new_colors_rest = []
    
    for center in sparse_centers:
        # Find k nearest neighbors
        distances, indices = tree.query(center, k=k_neighbors)
        
        # Weight by inverse distance
        weights = 1.0 / (distances + 1e-6)
        weights /= np.sum(weights)
        
        # Interpolate position (with jitter)
        new_pos = np.average(splat.positions[indices], axis=0, weights=weights)
        if jitter > 0:
            local_scale = np.mean(distances)
            new_pos += np.random.randn(3) * local_scale * jitter
        new_positions.append(new_pos)
        
        # Interpolate other properties
        if splat.opacity is not None:
            new_opacities.append(np.average(splat.opacity[indices], weights=weights))
        
        if splat.scales is not None:
            new_scales.append(np.average(splat.scales[indices], axis=0, weights=weights))
        
        if splat.rotations is not None:
            # Simple weighted average (not proper quaternion interpolation, but okay for similar orientations)
            avg_rot = np.average(splat.rotations[indices], axis=0, weights=weights)
            avg_rot /= np.linalg.norm(avg_rot)  # Normalize
            new_rotations.append(avg_rot)
        
        if splat.colors_dc is not None:
            new_colors_dc.append(np.average(splat.colors_dc[indices], axis=0, weights=weights))
        
        if splat.colors_rest is not None:
            new_colors_rest.append(np.average(splat.colors_rest[indices], axis=0, weights=weights))
    
    # Append new Gaussians to splat
    n_new = len(new_positions)
    
    splat.positions = np.vstack([splat.positions, np.array(new_positions)])
    
    if splat.opacity is not None and new_opacities:
        splat.opacity = np.concatenate([splat.opacity, np.array(new_opacities)])
    
    if splat.scales is not None and new_scales:
        splat.scales = np.vstack([splat.scales, np.array(new_scales)])
    
    if splat.rotations is not None and new_rotations:
        splat.rotations = np.vstack([splat.rotations, np.array(new_rotations)])
    
    if splat.colors_dc is not None and new_colors_dc:
        splat.colors_dc = np.vstack([splat.colors_dc, np.array(new_colors_dc)])
    
    if splat.colors_rest is not None and new_colors_rest:
        splat.colors_rest = np.vstack([splat.colors_rest, np.array(new_colors_rest)])
    
    if verbose:
        print(f"    Original Gaussians: {n_original:,}")
        print(f"    New Gaussians added: {n_new:,}")
        print(f"    Total Gaussians: {len(splat):,}")
    
    return splat, n_new


def clone_and_split(splat, split_threshold_percentile=95, max_new=10000, verbose=True):
    """
    Clone and split large Gaussians into smaller ones for better coverage.
    
    This mimics the densification strategy from the original 3DGS paper,
    but applied post-hoc based on scale.
    
    Args:
        splat: GaussianSplat object
        split_threshold_percentile: Split Gaussians above this percentile in scale
        max_new: Maximum number of new Gaussians to add
        verbose: Print progress
    
    Returns:
        Modified GaussianSplat object
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  CLONE AND SPLIT LARGE GAUSSIANS")
        print("  " + "-" * 50)
    
    if splat.scales is None:
        if verbose:
            print("    No scale data - skipping")
        return splat, 0
    
    n_original = len(splat)
    
    # Find large Gaussians
    scales_linear = splat.get_scales_linear()
    max_scale = np.max(scales_linear, axis=1)
    threshold = np.percentile(max_scale, split_threshold_percentile)
    
    large_mask = max_scale > threshold
    large_indices = np.where(large_mask)[0]
    
    if verbose:
        print(f"    Scale threshold (p{split_threshold_percentile}): {threshold:.6f}")
        print(f"    Gaussians to split: {len(large_indices):,}")
    
    if len(large_indices) == 0:
        return splat, 0
    
    # Limit number of splits
    if len(large_indices) > max_new // 2:
        large_indices = np.random.choice(large_indices, max_new // 2, replace=False)
    
    # For each large Gaussian, create two children offset along the largest scale axis
    new_positions = []
    new_opacities = []
    new_scales = []
    new_rotations = []
    new_colors_dc = []
    new_colors_rest = []
    
    for idx in large_indices:
        pos = splat.positions[idx]
        scale = scales_linear[idx]
        
        # Find largest scale axis
        largest_axis = np.argmax(scale)
        offset_dir = np.zeros(3)
        offset_dir[largest_axis] = 1.0
        
        # If we have rotation, transform the offset
        if splat.rotations is not None:
            # Simplified: assume rotation is close to identity for now
            # Full implementation would apply quaternion rotation
            pass
        
        offset = offset_dir * scale[largest_axis] * 0.5
        
        # Create two children
        for sign in [-1, 1]:
            new_positions.append(pos + sign * offset)
            
            if splat.opacity is not None:
                new_opacities.append(splat.opacity[idx])
            
            if splat.scales is not None:
                # Reduce scale
                new_scale = splat.scales[idx].copy()
                new_scale[largest_axis] -= 0.5  # Halve in log space
                new_scales.append(new_scale)
            
            if splat.rotations is not None:
                new_rotations.append(splat.rotations[idx].copy())
            
            if splat.colors_dc is not None:
                new_colors_dc.append(splat.colors_dc[idx].copy())
            
            if splat.colors_rest is not None:
                new_colors_rest.append(splat.colors_rest[idx].copy())
    
    # Append new Gaussians
    n_new = len(new_positions)
    
    splat.positions = np.vstack([splat.positions, np.array(new_positions)])
    
    if splat.opacity is not None and new_opacities:
        splat.opacity = np.concatenate([splat.opacity, np.array(new_opacities)])
    
    if splat.scales is not None and new_scales:
        splat.scales = np.vstack([splat.scales, np.array(new_scales)])
    
    if splat.rotations is not None and new_rotations:
        splat.rotations = np.vstack([splat.rotations, np.array(new_rotations)])
    
    if splat.colors_dc is not None and new_colors_dc:
        splat.colors_dc = np.vstack([splat.colors_dc, np.array(new_colors_dc)])
    
    if splat.colors_rest is not None and new_colors_rest:
        splat.colors_rest = np.vstack([splat.colors_rest, np.array(new_colors_rest)])
    
    if verbose:
        print(f"    New Gaussians from splitting: {n_new:,}")
        print(f"    Total Gaussians: {len(splat):,}")
    
    return splat, n_new


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def get_gaussian_normals(splat):
    """
    Estimate surface normals from Gaussian orientations.
    
    The normal is the axis corresponding to the smallest scale (flattest direction).
    
    Returns:
        normals: Nx3 array of unit normals
    """
    if splat.rotations is None or splat.scales is None:
        return None
    
    n = len(splat)
    normals = np.zeros((n, 3), dtype=np.float32)
    scales_linear = splat.get_scales_linear()
    
    for i in range(n):
        # Find the smallest scale axis (this is the "flat" direction / normal)
        smallest_axis = np.argmin(scales_linear[i])
        
        # Get the rotation matrix
        R = quaternion_to_rotation_matrix(splat.rotations[i])
        
        # The normal is the column of R corresponding to the smallest scale
        normal = R[:, smallest_axis]
        normals[i] = normal / (np.linalg.norm(normal) + 1e-8)
    
    return normals


def ransac_fit_plane(points, n_iterations=100, distance_threshold=0.05, min_inliers_ratio=0.1):
    """
    Fit a plane to points using RANSAC.
    
    Args:
        points: Nx3 array of points
        n_iterations: Number of RANSAC iterations
        distance_threshold: Max distance from plane to be considered inlier
        min_inliers_ratio: Minimum ratio of inliers for a valid plane
    
    Returns:
        best_plane: (normal, d) where plane equation is normal.dot(p) + d = 0
        inlier_mask: Boolean mask of inlier points
        None, None if no valid plane found
    """
    n_points = len(points)
    if n_points < 3:
        return None, None
    
    best_inliers = None
    best_plane = None
    best_count = 0
    min_inliers = int(n_points * min_inliers_ratio)
    
    for _ in range(n_iterations):
        # Sample 3 random points
        indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[indices]
        
        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-8:
            continue
        
        normal = normal / norm
        d = -np.dot(normal, p1)
        
        # Count inliers
        distances = np.abs(np.dot(points, normal) + d)
        inlier_mask = distances < distance_threshold
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers > best_count and n_inliers >= min_inliers:
            best_count = n_inliers
            best_plane = (normal, d)
            best_inliers = inlier_mask
    
    return best_plane, best_inliers


def detect_planes_ransac(splat, max_planes=10, distance_threshold=0.05, 
                         min_inliers_ratio=0.05, n_iterations=200, verbose=True):
    """
    Detect multiple dominant planes in the scene using iterative RANSAC.
    
    Args:
        splat: GaussianSplat object
        max_planes: Maximum number of planes to detect
        distance_threshold: Distance threshold for RANSAC inliers
        min_inliers_ratio: Minimum ratio of remaining points to form a plane
        n_iterations: RANSAC iterations per plane
        verbose: Print progress
    
    Returns:
        planes: List of (normal, d, inlier_indices) tuples
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  PLANE DETECTION (RANSAC)")
        print("  " + "-" * 50)
    
    positions = splat.positions.copy()
    n_total = len(positions)
    
    planes = []
    remaining_mask = np.ones(n_total, dtype=bool)
    remaining_indices = np.arange(n_total)
    
    for plane_idx in range(max_planes):
        # Get remaining points
        current_points = positions[remaining_mask]
        current_indices = remaining_indices[remaining_mask]
        
        if len(current_points) < n_total * 0.01:  # Stop if less than 1% remain
            break
        
        # Fit plane
        plane, inlier_mask = ransac_fit_plane(
            current_points, 
            n_iterations=n_iterations,
            distance_threshold=distance_threshold,
            min_inliers_ratio=min_inliers_ratio
        )
        
        if plane is None:
            break
        
        # Get global indices of inliers
        inlier_global_indices = current_indices[inlier_mask]
        
        if verbose:
            normal, d = plane
            print(f"    Plane {plane_idx + 1}: {len(inlier_global_indices):,} Gaussians")
            print(f"      Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        
        planes.append((plane[0], plane[1], inlier_global_indices))
        
        # Remove inliers from remaining
        remaining_mask[inlier_global_indices] = False
    
    if verbose:
        total_assigned = sum(len(p[2]) for p in planes)
        print(f"    Total planes detected: {len(planes)}")
        print(f"    Gaussians assigned to planes: {total_assigned:,} ({100*total_assigned/n_total:.1f}%)")
    
    return planes


def flatten_to_planes(splat, planes, max_distance=0.1, flatten_strength=0.8, verbose=True):
    """
    Flatten Gaussians to their nearest detected plane.
    
    Args:
        splat: GaussianSplat object
        planes: List of (normal, d, inlier_indices) from detect_planes_ransac
        max_distance: Maximum distance to plane for flattening
        flatten_strength: How much to flatten (0=none, 1=fully onto plane)
        verbose: Print progress
    
    Returns:
        Modified splat, number of flattened Gaussians
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  PLANE FLATTENING")
        print("  " + "-" * 50)
    
    if len(planes) == 0:
        if verbose:
            print("    No planes detected - skipping flattening")
        return splat, 0
    
    n_flattened = 0
    
    for plane_idx, (normal, d, inlier_indices) in enumerate(planes):
        # For each Gaussian in this plane's inliers
        for idx in inlier_indices:
            pos = splat.positions[idx]
            
            # Compute signed distance to plane
            dist = np.dot(normal, pos) + d
            
            if abs(dist) <= max_distance:
                # Project onto plane (partial based on strength)
                projection = pos - dist * normal * flatten_strength
                splat.positions[idx] = projection
                n_flattened += 1
    
    if verbose:
        print(f"    Gaussians flattened: {n_flattened:,}")
    
    return splat, n_flattened


def local_depth_consistency_filter(splat, k_neighbors=20, depth_std_threshold=2.0, 
                                   flatten_outliers=True, verbose=True):
    """
    Identify and handle Gaussians that are depth outliers in their local neighborhood.
    
    This targets reflection artifacts where splats are placed at incorrect depths
    but surrounded by splats at the correct surface depth.
    
    Args:
        splat: GaussianSplat object
        k_neighbors: Number of neighbors to analyze
        depth_std_threshold: Flag as outlier if depth differs by more than this many local std devs
        flatten_outliers: If True, project outliers to local median depth; if False, remove them
        verbose: Print progress
    
    Returns:
        Modified splat, number of affected Gaussians
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  LOCAL DEPTH CONSISTENCY FILTER")
        print("  " + "-" * 50)
    
    n_original = len(splat)
    
    # First, estimate local surface normals
    normals = get_gaussian_normals(splat)
    
    if normals is None:
        if verbose:
            print("    Cannot compute normals - skipping depth filter")
        return splat, 0
    
    # Build KD-tree
    tree = KDTree(splat.positions)
    
    outlier_mask = np.zeros(n_original, dtype=bool)
    projected_positions = splat.positions.copy()
    
    for i in range(n_original):
        # Find neighbors
        distances, indices = tree.query(splat.positions[i], k=k_neighbors + 1)
        neighbor_indices = indices[1:]  # Exclude self
        
        # Get positions relative to current point
        neighbor_positions = splat.positions[neighbor_indices]
        current_pos = splat.positions[i]
        current_normal = normals[i]
        
        # Compute depth along the local normal direction
        relative_positions = neighbor_positions - current_pos
        depths = np.dot(relative_positions, current_normal)
        
        # Current point's depth is 0 (reference)
        # Check if neighbors form a consistent surface
        depth_median = np.median(depths)
        depth_std = np.std(depths)
        
        if depth_std < 1e-6:
            continue
        
        # The current point should be near the surface (depth ~0 relative to neighbors)
        # If depth_median is significantly non-zero, neighbors are offset from us
        # This means WE might be the outlier
        
        # Check if we're an outlier (far from the local surface)
        # The local surface is approximately at -depth_median from us
        our_depth_from_surface = -depth_median  # How far we are from neighbor median
        
        if abs(our_depth_from_surface) > depth_std_threshold * depth_std:
            outlier_mask[i] = True
            
            if flatten_outliers:
                # Project to local median surface
                projected_positions[i] = current_pos + our_depth_from_surface * current_normal
    
    n_outliers = np.sum(outlier_mask)
    
    if verbose:
        print(f"    Depth outliers found: {n_outliers:,}")
    
    if n_outliers == 0:
        return splat, 0
    
    if flatten_outliers:
        # Update positions to projected ones
        splat.positions = projected_positions
        if verbose:
            print(f"    Outliers projected to local surface")
    else:
        # Remove outliers
        keep_mask = ~outlier_mask
        splat.positions = splat.positions[keep_mask]
        if splat.opacity is not None:
            splat.opacity = splat.opacity[keep_mask]
        if splat.scales is not None:
            splat.scales = splat.scales[keep_mask]
        if splat.rotations is not None:
            splat.rotations = splat.rotations[keep_mask]
        if splat.colors_dc is not None:
            splat.colors_dc = splat.colors_dc[keep_mask]
        if splat.colors_rest is not None:
            splat.colors_rest = splat.colors_rest[keep_mask]
        if verbose:
            print(f"    Outliers removed")
    
    return splat, n_outliers


def compress_surface_thickness(splat, k_neighbors=15, max_thickness_factor=3.0, 
                               compression_strength=0.7, verbose=True):
    """
    Compress regions where Gaussians are spread too thick along the surface normal.
    
    This helps condense reflection artifacts that create false depth.
    
    Args:
        splat: GaussianSplat object
        k_neighbors: Number of neighbors to analyze local thickness
        max_thickness_factor: Compress if local thickness exceeds this factor of median scale
        compression_strength: How much to compress (0=none, 1=fully to median)
        verbose: Print progress
    
    Returns:
        Modified splat, number of compressed regions
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  SURFACE THICKNESS COMPRESSION")
        print("  " + "-" * 50)
    
    normals = get_gaussian_normals(splat)
    
    if normals is None:
        if verbose:
            print("    Cannot compute normals - skipping thickness compression")
        return splat, 0
    
    n = len(splat)
    scales_linear = splat.get_scales_linear()
    median_scale = np.median(scales_linear)
    
    tree = KDTree(splat.positions)
    
    n_compressed = 0
    new_positions = splat.positions.copy()
    
    # Process in chunks for efficiency
    chunk_size = 1000
    
    for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        
        for i in range(start_idx, end_idx):
            distances, indices = tree.query(splat.positions[i], k=k_neighbors + 1)
            neighbor_indices = indices[1:]
            
            # Compute thickness along normal
            current_normal = normals[i]
            neighbor_positions = splat.positions[neighbor_indices]
            
            # Project neighbors onto normal axis
            depths = np.dot(neighbor_positions - splat.positions[i], current_normal)
            
            # Measure thickness (range of depths)
            thickness = np.max(depths) - np.min(depths)
            
            # Check if too thick
            expected_thickness = median_scale * max_thickness_factor
            
            if thickness > expected_thickness:
                # Compress toward median depth
                median_depth = np.median(depths)
                current_depth = 0  # We're at the reference point
                
                # Move toward median
                adjustment = (median_depth - current_depth) * compression_strength
                new_positions[i] = splat.positions[i] + adjustment * current_normal
                n_compressed += 1
    
    splat.positions = new_positions
    
    if verbose:
        print(f"    Gaussians with position adjusted: {n_compressed:,}")
    
    return splat, n_compressed


def enhance_splat(input_path, output_path, 
                  remove_floaters_enabled=True,
                  densify_sparse_enabled=True,
                  split_large_enabled=True,
                  # New plane-fitting options
                  plane_detection_enabled=False,
                  depth_consistency_enabled=False,
                  thickness_compression_enabled=False,
                  grid_resolution=50,
                  density_threshold_percentile=10,
                  # Floater removal parameters
                  floater_std_threshold=3.0,
                  floater_min_opacity=0.05,
                  # Densification parameters
                  densify_k_neighbors=5,
                  densify_jitter=0.02,
                  # Split parameters
                  split_threshold_percentile=95,
                  split_max_new=10000,
                  # Plane detection parameters
                  plane_max_planes=10,
                  plane_distance_threshold=0.05,
                  plane_min_inliers_ratio=0.05,
                  plane_flatten_strength=0.8,
                  # Depth consistency parameters
                  depth_k_neighbors=20,
                  depth_std_threshold=2.0,
                  depth_flatten_outliers=True,
                  # Thickness compression parameters
                  thickness_k_neighbors=15,
                  thickness_max_factor=3.0,
                  thickness_compression_strength=0.7,
                  verbose=True):
    """
    Run full enhancement pipeline on a Gaussian splat.
    
    Args:
        input_path: Path to input PLY file
        output_path: Path to output PLY file
        remove_floaters_enabled: Remove outlier Gaussians
        densify_sparse_enabled: Add Gaussians in sparse regions
        split_large_enabled: Split large Gaussians
        plane_detection_enabled: Detect planes and flatten Gaussians to them
        depth_consistency_enabled: Filter depth outliers in local neighborhoods
        thickness_compression_enabled: Compress overly thick surface regions
        grid_resolution: Resolution for sparse detection grid
        density_threshold_percentile: What percentile is "sparse"
        floater_std_threshold: Remove points beyond this many std devs from centroid
        floater_min_opacity: Remove Gaussians with opacity below this threshold
        densify_k_neighbors: Number of neighbors to interpolate from when densifying
        densify_jitter: Random offset factor for new Gaussians (relative to local scale)
        split_threshold_percentile: Split Gaussians above this percentile in scale
        split_max_new: Maximum number of new Gaussians to add from splitting
        plane_max_planes: Maximum number of planes to detect
        plane_distance_threshold: RANSAC distance threshold for plane inliers
        plane_min_inliers_ratio: Minimum ratio of points to form a valid plane
        plane_flatten_strength: How much to flatten (0=none, 1=fully onto plane)
        depth_k_neighbors: Number of neighbors for depth consistency analysis
        depth_std_threshold: Std devs from local surface to be considered outlier
        depth_flatten_outliers: If True, project outliers; if False, remove them
        thickness_k_neighbors: Number of neighbors for thickness analysis
        thickness_max_factor: Compress if thickness exceeds this factor of median scale
        thickness_compression_strength: How much to compress thick regions
        verbose: Print progress
    
    Returns:
        Statistics dict
    """
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("GAUSSIAN SPLAT ENHANCEMENT")
    print("=" * 60)
    
    # Load splat
    splat = GaussianSplat(input_path)
    n_original = len(splat)
    
    stats = {
        'original_count': n_original,
        'floaters_removed': 0,
        'planes_detected': 0,
        'plane_flattened': 0,
        'depth_outliers': 0,
        'thickness_compressed': 0,
        'sparse_added': 0,
        'split_added': 0,
    }
    
    # Step 1: Remove floaters
    if remove_floaters_enabled:
        splat, n_removed = remove_floaters(
            splat, 
            std_threshold=floater_std_threshold,
            min_opacity=floater_min_opacity,
            verbose=verbose
        )
        stats['floaters_removed'] = n_removed
    
    # Step 2: Plane detection and flattening (for walls, floors, etc.)
    if plane_detection_enabled:
        planes = detect_planes_ransac(
            splat,
            max_planes=plane_max_planes,
            distance_threshold=plane_distance_threshold,
            min_inliers_ratio=plane_min_inliers_ratio,
            verbose=verbose
        )
        stats['planes_detected'] = len(planes)
        
        splat, n_flattened = flatten_to_planes(
            splat,
            planes,
            max_distance=plane_distance_threshold * 2,
            flatten_strength=plane_flatten_strength,
            verbose=verbose
        )
        stats['plane_flattened'] = n_flattened
    
    # Step 3: Local depth consistency filter (removes/flattens reflection artifacts)
    if depth_consistency_enabled:
        splat, n_depth = local_depth_consistency_filter(
            splat,
            k_neighbors=depth_k_neighbors,
            depth_std_threshold=depth_std_threshold,
            flatten_outliers=depth_flatten_outliers,
            verbose=verbose
        )
        stats['depth_outliers'] = n_depth
    
    # Step 4: Surface thickness compression
    if thickness_compression_enabled:
        splat, n_thickness = compress_surface_thickness(
            splat,
            k_neighbors=thickness_k_neighbors,
            max_thickness_factor=thickness_max_factor,
            compression_strength=thickness_compression_strength,
            verbose=verbose
        )
        stats['thickness_compressed'] = n_thickness
    
    # Step 5: Find and fill sparse regions
    if densify_sparse_enabled:
        sparse_centers, _ = find_sparse_regions(
            splat, 
            grid_resolution=grid_resolution,
            density_threshold_percentile=density_threshold_percentile,
            verbose=verbose
        )
        splat, n_added = densify_sparse_regions(
            splat, 
            sparse_centers, 
            k_neighbors=densify_k_neighbors,
            jitter=densify_jitter,
            verbose=verbose
        )
        stats['sparse_added'] = n_added
    
    # Step 6: Split large Gaussians
    if split_large_enabled:
        splat, n_split = clone_and_split(
            splat, 
            split_threshold_percentile=split_threshold_percentile,
            max_new=split_max_new,
            verbose=verbose
        )
        stats['split_added'] = n_split
    
    # Save result
    print("\n  " + "-" * 50)
    print("  SAVING RESULT")
    print("  " + "-" * 50)
    splat.save(output_path)
    
    stats['final_count'] = len(splat)
    stats['elapsed_time'] = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("ENHANCEMENT SUMMARY")
    print("=" * 60)
    print(f"  Original Gaussians:    {stats['original_count']:,}")
    print(f"  Floaters removed:      {stats['floaters_removed']:,}")
    if plane_detection_enabled:
        print(f"  Planes detected:       {stats['planes_detected']}")
        print(f"  Flattened to planes:   {stats['plane_flattened']:,}")
    if depth_consistency_enabled:
        print(f"  Depth outliers fixed:  {stats['depth_outliers']:,}")
    if thickness_compression_enabled:
        print(f"  Thickness compressed:  {stats['thickness_compressed']:,}")
    print(f"  Sparse region fills:   {stats['sparse_added']:,}")
    print(f"  Large Gaussian splits: {stats['split_added']:,}")
    print(f"  Final Gaussians:       {stats['final_count']:,}")
    print(f"  Elapsed time:          {stats['elapsed_time']:.2f}s")
    print("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Enhance Gaussian Splat by removing artifacts and filling holes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic enhancement
    python splat_enhance.py input.ply output.ply
    
    # Only remove floaters
    python splat_enhance.py input.ply output.ply --no-densify --no-split
    
    # Aggressive hole filling
    python splat_enhance.py input.ply output.ply --grid 30 --density-threshold 20
    
    # Fine grid for detailed models
    python splat_enhance.py input.ply output.ply --grid 100
    
    # Room/indoor scene with reflections - flatten to walls
    python splat_enhance.py input.ply output.ply --flatten-planes --depth-filter
    
    # Aggressive reflection artifact removal
    python splat_enhance.py input.ply output.ply --flatten-planes --depth-filter --compress-thickness
    
    # Fine-tune plane detection for large rooms
    python splat_enhance.py input.ply output.ply --flatten-planes --plane-max 20 --plane-distance 0.1
        """
    )
    
    parser.add_argument("input", help="Input .ply Gaussian Splat file")
    parser.add_argument("output", help="Output .ply file")
    
    parser.add_argument("--no-floaters", action="store_true",
                        help="Skip floater removal")
    parser.add_argument("--no-densify", action="store_true",
                        help="Skip sparse region densification")
    parser.add_argument("--no-split", action="store_true",
                        help="Skip large Gaussian splitting")
    
    # Plane fitting options (for flattening walls/surfaces)
    parser.add_argument("--flatten-planes", action="store_true",
                        help="Enable plane detection and flattening (good for rooms/walls)")
    parser.add_argument("--depth-filter", action="store_true",
                        help="Enable local depth consistency filtering (removes reflection artifacts)")
    parser.add_argument("--compress-thickness", action="store_true",
                        help="Compress overly thick surface regions")
    
    parser.add_argument("--grid", type=int, default=50,
                        help="Grid resolution for sparse detection (default: 50)")
    parser.add_argument("--density-threshold", type=int, default=10,
                        help="Percentile for sparse threshold (default: 10)")
    
    # Floater removal parameters
    parser.add_argument("--floater-std", type=float, default=3.0,
                        help="Std dev threshold for position outliers (default: 3.0)")
    parser.add_argument("--floater-min-opacity", type=float, default=0.05,
                        help="Minimum opacity threshold (default: 0.05)")
    
    # Densification parameters
    parser.add_argument("--densify-neighbors", type=int, default=5,
                        help="Number of neighbors to interpolate from (default: 5)")
    parser.add_argument("--densify-jitter", type=float, default=0.02,
                        help="Random offset factor for new Gaussians (default: 0.02)")
    
    # Split parameters
    parser.add_argument("--split-percentile", type=int, default=95,
                        help="Split Gaussians above this percentile in scale (default: 95)")
    parser.add_argument("--split-max-new", type=int, default=10000,
                        help="Maximum new Gaussians from splitting (default: 10000)")
    
    # Plane detection parameters
    parser.add_argument("--plane-max", type=int, default=10,
                        help="Maximum number of planes to detect (default: 10)")
    parser.add_argument("--plane-distance", type=float, default=0.05,
                        help="RANSAC distance threshold for plane inliers (default: 0.05)")
    parser.add_argument("--plane-min-ratio", type=float, default=0.05,
                        help="Minimum ratio of points to form a plane (default: 0.05)")
    parser.add_argument("--plane-flatten-strength", type=float, default=0.8,
                        help="How strongly to flatten to planes, 0-1 (default: 0.8)")
    
    # Depth consistency parameters  
    parser.add_argument("--depth-neighbors", type=int, default=20,
                        help="Number of neighbors for depth analysis (default: 20)")
    parser.add_argument("--depth-std", type=float, default=2.0,
                        help="Std devs threshold for depth outliers (default: 2.0)")
    parser.add_argument("--depth-remove", action="store_true",
                        help="Remove depth outliers instead of projecting them")
    
    # Thickness compression parameters
    parser.add_argument("--thickness-neighbors", type=int, default=15,
                        help="Number of neighbors for thickness analysis (default: 15)")
    parser.add_argument("--thickness-factor", type=float, default=3.0,
                        help="Compress if thickness exceeds this factor of median scale (default: 3.0)")
    parser.add_argument("--thickness-strength", type=float, default=0.7,
                        help="Compression strength, 0-1 (default: 0.7)")
    
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    enhance_splat(
        args.input,
        args.output,
        remove_floaters_enabled=not args.no_floaters,
        densify_sparse_enabled=not args.no_densify,
        split_large_enabled=not args.no_split,
        plane_detection_enabled=args.flatten_planes,
        depth_consistency_enabled=args.depth_filter,
        thickness_compression_enabled=args.compress_thickness,
        grid_resolution=args.grid,
        density_threshold_percentile=args.density_threshold,
        floater_std_threshold=args.floater_std,
        floater_min_opacity=args.floater_min_opacity,
        densify_k_neighbors=args.densify_neighbors,
        densify_jitter=args.densify_jitter,
        split_threshold_percentile=args.split_percentile,
        split_max_new=args.split_max_new,
        plane_max_planes=args.plane_max,
        plane_distance_threshold=args.plane_distance,
        plane_min_inliers_ratio=args.plane_min_ratio,
        plane_flatten_strength=args.plane_flatten_strength,
        depth_k_neighbors=args.depth_neighbors,
        depth_std_threshold=args.depth_std,
        depth_flatten_outliers=not args.depth_remove,
        thickness_k_neighbors=args.thickness_neighbors,
        thickness_max_factor=args.thickness_factor,
        thickness_compression_strength=args.thickness_strength,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
