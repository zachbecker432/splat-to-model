"""
Point Cloud to Mesh Converter (using Open3D)

Converts a point cloud PLY file to a mesh using Poisson surface reconstruction.
Supports both standard Poisson reconstruction and segmented reconstruction
that processes planar regions (walls, floors) separately from organic geometry.

Usage:
    python pointcloud_to_mesh.py input.ply output.obj [options]
    
    # Use segmented reconstruction for architectural scenes:
    python pointcloud_to_mesh.py input.ply output.obj --segmented
"""

import argparse
import numpy as np
import open3d as o3d
import time
from pathlib import Path
from scipy.spatial import Delaunay


def log_step(msg, indent=2):
    """Print a log message with consistent formatting."""
    prefix = " " * indent
    print(f"{prefix}[INFO] {msg}")


def load_point_cloud(input_path, verbose=True):
    """Load point cloud from PLY file."""
    if verbose:
        log_step(f"Loading point cloud: {input_path}")
    
    load_start = time.time()
    pcd = o3d.io.read_point_cloud(str(input_path))
    load_time = time.time() - load_start
    
    if verbose:
        print()
        print("  " + "-" * 50)
        print("  POINT CLOUD LOADED")
        print("  " + "-" * 50)
        print(f"    Points:      {len(pcd.points):,}")
        print(f"    Has colors:  {pcd.has_colors()}")
        print(f"    Has normals: {pcd.has_normals()}")
        print(f"    Load time:   {load_time:.2f}s")
        
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            print(f"    Bounding box:")
            print(f"      X: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
            print(f"      Y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
            print(f"      Z: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")
        print()
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size=None, remove_outliers=True, outlier_std_ratio=2.0, 
                           outlier_neighbors=30, verbose=True):
    """
    Preprocess point cloud: downsample and remove outliers.
    
    Args:
        pcd: Open3D point cloud
        voxel_size: Voxel size for downsampling (None = auto-calculate, 0 = disabled)
        remove_outliers: Whether to remove statistical outliers
        outlier_std_ratio: Standard deviation ratio for outlier removal (lower = more aggressive)
        outlier_neighbors: Number of neighbors for outlier detection
        verbose: Print progress
    
    Returns:
        Preprocessed point cloud
    """
    initial_points = len(pcd.points)
    
    if verbose:
        print("  " + "-" * 50)
        print("  PREPROCESSING")
        print("  " + "-" * 50)
        
    # Get bounding box info for auto-calculations
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    diagonal = np.linalg.norm(extent)
    
    # Compute point density for smarter decisions
    distances = pcd.compute_nearest_neighbor_distance()
    avg_spacing = np.mean(distances)
    
    if verbose:
        print(f"    Initial points: {initial_points:,}")
        print(f"    Bounding box extent: [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]")
        print(f"    Diagonal: {diagonal:.4f}")
        print(f"    Average point spacing: {avg_spacing:.6f}")
    
    # Auto-calculate voxel size if not specified (but not if explicitly set to 0)
    if voxel_size is None:
        # Use ~0.2% of diagonal for finer detail preservation (was 0.5%)
        # Also ensure voxel size is at least 2x average spacing to avoid minimal effect
        auto_voxel = diagonal * 0.002
        min_effective_voxel = avg_spacing * 2.0
        voxel_size = max(auto_voxel, min_effective_voxel)
        if verbose:
            print(f"    Auto-calculated voxel size: {voxel_size:.6f} (0.2% diagonal, min 2x spacing)")
    
    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        if verbose:
            print(f"    Downsampling with voxel size: {voxel_size:.6f}...")
        downsample_start = time.time()
        points_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size)
        downsample_time = time.time() - downsample_start
        reduction_pct = (1 - len(pcd.points) / points_before) * 100 if points_before > 0 else 0
        if verbose:
            print(f"    Points after downsampling: {len(pcd.points):,} ({reduction_pct:.1f}% reduction, {downsample_time:.2f}s)")
    else:
        if verbose:
            print(f"    Downsampling: disabled")
    
    # Remove statistical outliers
    if remove_outliers and len(pcd.points) > outlier_neighbors:
        if verbose:
            print(f"    Removing outliers (neighbors={outlier_neighbors}, std_ratio={outlier_std_ratio})...")
        outlier_start = time.time()
        points_before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=outlier_neighbors, std_ratio=outlier_std_ratio)
        outlier_time = time.time() - outlier_start
        removed = points_before - len(pcd.points)
        removal_pct = (removed / points_before) * 100 if points_before > 0 else 0
        if verbose:
            print(f"    Points after outlier removal: {len(pcd.points):,} (removed {removed:,} = {removal_pct:.1f}%, {outlier_time:.2f}s)")
    elif remove_outliers:
        if verbose:
            print(f"    Skipping outlier removal (too few points)")
    
    total_reduction = (1 - len(pcd.points) / initial_points) * 100 if initial_points > 0 else 0
    if verbose:
        print(f"    Total preprocessing reduction: {initial_points:,} -> {len(pcd.points):,} ({total_reduction:.1f}%)")
        print()
    
    return pcd


def estimate_normals(pcd, search_radius=None, max_nn=100, orient_k=30, verbose=True):
    """
    Estimate normals for point cloud.
    
    Args:
        pcd: Open3D point cloud
        search_radius: Search radius for normal estimation (None = auto)
        max_nn: Maximum number of neighbors to consider
        orient_k: Number of neighbors for normal orientation consistency
        verbose: Print progress
    
    Returns:
        Point cloud with normals
    """
    if verbose:
        print("  " + "-" * 50)
        print("  NORMAL ESTIMATION")
        print("  " + "-" * 50)
    
    # Auto-calculate search radius based on point cloud density
    if search_radius is None:
        # Use nearest neighbor distance for more accurate radius estimation
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        # Use 5x average distance for good neighborhood coverage
        search_radius = avg_dist * 5.0
        if verbose:
            print(f"    Average point spacing: {avg_dist:.6f}")
            print(f"    Auto-calculated search radius: {search_radius:.6f} (5x avg spacing)")
    
    if verbose:
        print(f"    Max neighbors: {max_nn}")
        print(f"    Estimating normals...")
    
    normal_start = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius,
            max_nn=max_nn
        )
    )
    normal_time = time.time() - normal_start
    
    if verbose:
        print(f"    Normal estimation complete ({normal_time:.2f}s)")
    
    # Orient normals consistently (important for Poisson reconstruction)
    # Use higher k value for more robust orientation
    if verbose:
        print(f"    Orienting normals consistently (k={orient_k})...")
    
    orient_start = time.time()
    pcd.orient_normals_consistent_tangent_plane(k=orient_k)
    orient_time = time.time() - orient_start
    
    if verbose:
        print(f"    Normal orientation complete ({orient_time:.2f}s)")
        print()
    
    return pcd


def poisson_reconstruction(pcd, depth=9, width=0, scale=1.1, linear_fit=False, verbose=True):
    """
    Perform Poisson surface reconstruction.
    
    Args:
        pcd: Open3D point cloud with normals
        depth: Octree depth (higher = more detail, more polygons)
        width: Target width of finest octree cells (0 = use depth instead)
        scale: Ratio between reconstruction cube and bounding cube
        linear_fit: Use linear interpolation for iso-surface extraction
        verbose: Print progress
    
    Returns:
        Tuple of (mesh, densities)
    """
    if verbose:
        print("  " + "-" * 50)
        print("  POISSON SURFACE RECONSTRUCTION")
        print("  " + "-" * 50)
        print(f"    Octree depth: {depth}")
        print(f"    Scale: {scale}")
        print(f"    Linear fit: {linear_fit}")
        print(f"    Input points: {len(pcd.points):,}")
        print(f"    Running reconstruction (this may take a while)...")
    
    recon_start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit
    )
    recon_time = time.time() - recon_start
    
    if verbose:
        print(f"    Reconstruction complete ({recon_time:.2f}s)")
        print(f"    Generated vertices:  {len(mesh.vertices):,}")
        print(f"    Generated triangles: {len(mesh.triangles):,}")
        
        # Report density statistics
        dens = np.asarray(densities)
        print(f"    Density range: [{dens.min():.4f}, {dens.max():.4f}]")
        print()
    
    return mesh, np.asarray(densities)


def ball_pivoting_reconstruction(pcd, radii=None, fill_holes=True, thin_geometry=True, verbose=True):
    """
    Perform Ball Pivoting Algorithm (BPA) surface reconstruction.
    Better for meshes with clear surface structure and uniform point density.
    
    Args:
        pcd: Open3D point cloud with normals
        radii: List of ball radii to use (None = auto-calculate)
        fill_holes: Attempt to fill holes in the resulting mesh
        thin_geometry: Add extra small radii for thin features (wing tips, tails)
        verbose: Print progress
    
    Returns:
        Triangle mesh
    """
    if verbose:
        print("  " + "-" * 50)
        print("  BALL PIVOTING RECONSTRUCTION")
        print("  " + "-" * 50)
        print(f"    Input points: {len(pcd.points):,}")
        print(f"    Thin geometry mode: {thin_geometry}")
    
    # Auto-calculate radii based on point spacing - use more radii for better coverage
    if radii is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.percentile(distances, 5)   # 5th percentile for thin areas
        max_dist = np.percentile(distances, 95)  # 95th percentile to avoid outliers
        
        # Base radii for normal geometry
        radii = [
            avg_dist * 0.5,
            avg_dist,
            avg_dist * 1.5,
            avg_dist * 2,
            avg_dist * 3,
            avg_dist * 4,
            max_dist * 2,
            max_dist * 4,
        ]
        
        # Add smaller radii for thin geometry (wing tips, tails, thin features)
        if thin_geometry:
            thin_radii = [
                min_dist * 0.5,
                min_dist,
                avg_dist * 0.25,
                avg_dist * 0.33,
            ]
            radii = thin_radii + radii
        
        # Remove duplicates, filter out very small values, and sort
        radii = sorted(set(r for r in radii if r > 0.0001))
        
        if verbose:
            print(f"    Min point spacing (5th pct): {min_dist:.6f}")
            print(f"    Average point spacing: {avg_dist:.6f}")
            print(f"    Std dev spacing: {std_dist:.6f}")
            print(f"    Max point spacing (95th pct): {max_dist:.6f}")
            print(f"    Using {len(radii)} radii: {[f'{r:.4f}' for r in radii]}")
    
    if verbose:
        print(f"    Running BPA reconstruction...")
    
    recon_start = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    recon_time = time.time() - recon_start
    
    if verbose:
        print(f"    Reconstruction complete ({recon_time:.2f}s)")
        print(f"    Generated vertices:  {len(mesh.vertices):,}")
        print(f"    Generated triangles: {len(mesh.triangles):,}")
    
    # Attempt to fill holes
    if fill_holes and len(mesh.triangles) > 0:
        mesh = fill_mesh_holes(mesh, verbose=verbose)
    
    if verbose:
        print()
    
    return mesh


def fill_mesh_holes(mesh, hole_size=100, verbose=True):
    """
    Fill holes in a mesh by identifying boundary edges and filling them.
    
    Args:
        mesh: Open3D TriangleMesh
        hole_size: Maximum number of edges in a hole to fill
        verbose: Print progress
    
    Returns:
        Mesh with holes filled
    """
    if verbose:
        print(f"    Attempting to fill holes (max size: {hole_size} edges)...")
    
    initial_triangles = len(mesh.triangles)
    
    # Get boundary edges (edges that belong to only one triangle)
    mesh.compute_adjacency_list()
    
    # Use Open3D's built-in method if available, otherwise skip
    try:
        # Try to fill holes using mesh repair
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Compute vertex normals for filled regions
        mesh.compute_vertex_normals()
        
        new_triangles = len(mesh.triangles)
        if verbose:
            if new_triangles != initial_triangles:
                print(f"    Triangles after cleanup: {new_triangles:,} (was {initial_triangles:,})")
            else:
                print(f"    Mesh cleaned, {new_triangles:,} triangles")
    except Exception as e:
        if verbose:
            print(f"    Hole filling skipped: {e}")
    
    return mesh


def remove_spurious_triangles(mesh, edge_length_percentile=95, aspect_ratio_threshold=10.0, verbose=True):
    """
    Remove spurious triangles that have edges much longer than typical,
    or that are very thin/elongated (high aspect ratio).
    These are usually caused by BPA connecting points across gaps.
    
    Args:
        mesh: Open3D TriangleMesh
        edge_length_percentile: Keep edges below this percentile (e.g., 95 = remove top 5% longest)
        aspect_ratio_threshold: Max ratio of longest to shortest edge
        verbose: Print progress
    
    Returns:
        Cleaned mesh
    """
    if verbose:
        print("  " + "-" * 50)
        print("  REMOVING SPURIOUS TRIANGLES")
        print("  " + "-" * 50)
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(triangles) == 0:
        if verbose:
            print("    No triangles to process")
        return mesh
    
    initial_count = len(triangles)
    
    # Calculate edge lengths for all triangles
    edge_lengths = []
    triangle_max_edges = []
    triangle_min_edges = []
    triangle_aspect_ratios = []
    
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        e0 = np.linalg.norm(v1 - v0)
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v0 - v2)
        
        edges = [e0, e1, e2]
        edge_lengths.extend(edges)
        triangle_max_edges.append(max(edges))
        triangle_min_edges.append(min(edges))
        
        # Aspect ratio (avoid division by zero)
        min_edge = min(edges)
        if min_edge > 0:
            triangle_aspect_ratios.append(max(edges) / min_edge)
        else:
            triangle_aspect_ratios.append(float('inf'))
    
    edge_lengths = np.array(edge_lengths)
    triangle_max_edges = np.array(triangle_max_edges)
    triangle_aspect_ratios = np.array(triangle_aspect_ratios)
    
    # Use percentile-based threshold for edge length
    edge_length_threshold = np.percentile(triangle_max_edges, edge_length_percentile)
    
    if verbose:
        print(f"    Edge length stats:")
        print(f"      Min: {triangle_max_edges.min():.6f}")
        print(f"      Median: {np.median(triangle_max_edges):.6f}")
        print(f"      Mean: {np.mean(triangle_max_edges):.6f}")
        print(f"      95th percentile: {np.percentile(triangle_max_edges, 95):.6f}")
        print(f"      Max: {triangle_max_edges.max():.6f}")
        print(f"    Edge length threshold: {edge_length_threshold:.6f} ({edge_length_percentile}th percentile)")
        print(f"    Aspect ratio threshold: {aspect_ratio_threshold}")
    
    # Find triangles to keep
    keep_mask = (triangle_max_edges <= edge_length_threshold) & \
                (triangle_aspect_ratios <= aspect_ratio_threshold)
    
    removed_by_edge = np.sum(triangle_max_edges > edge_length_threshold)
    removed_by_aspect = np.sum((triangle_aspect_ratios > aspect_ratio_threshold) & (triangle_max_edges <= edge_length_threshold))
    
    # Keep only valid triangles
    valid_triangles = triangles[keep_mask]
    
    if verbose:
        print(f"    Triangles removed by edge length: {removed_by_edge:,}")
        print(f"    Triangles removed by aspect ratio: {removed_by_aspect:,}")
        print(f"    Triangles remaining: {len(valid_triangles):,} / {initial_count:,}")
    
    # Create new mesh with filtered triangles
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
    
    if mesh.has_vertex_colors():
        new_mesh.vertex_colors = mesh.vertex_colors
    
    if mesh.has_vertex_normals():
        new_mesh.vertex_normals = mesh.vertex_normals
    
    # Clean up unused vertices
    new_mesh.remove_unreferenced_vertices()
    
    # Recompute normals
    new_mesh.compute_vertex_normals()
    
    if verbose:
        print(f"    Final vertices: {len(new_mesh.vertices):,}")
        print(f"    Final triangles: {len(new_mesh.triangles):,}")
        print()
    
    return new_mesh


def alpha_shape_reconstruction(pcd, alpha=None, verbose=True):
    """
    Perform Alpha Shape surface reconstruction.
    Good for concave shapes while preserving sharp features.
    
    Args:
        pcd: Open3D point cloud
        alpha: Alpha value (None = auto-calculate based on point spacing)
        verbose: Print progress
    
    Returns:
        Triangle mesh
    """
    if verbose:
        print("  " + "-" * 50)
        print("  ALPHA SHAPE RECONSTRUCTION")
        print("  " + "-" * 50)
        print(f"    Input points: {len(pcd.points):,}")
    
    # Auto-calculate alpha based on point spacing
    if alpha is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        # Use larger alpha (3x) for better hole coverage while preserving shape
        alpha = avg_dist * 3.0
        if verbose:
            print(f"    Average point spacing: {avg_dist:.6f}")
            print(f"    Auto-calculated alpha: {alpha:.6f} (3x avg spacing)")
    
    if verbose:
        print(f"    Running alpha shape reconstruction...")
    
    recon_start = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    recon_time = time.time() - recon_start
    
    # Compute normals for the mesh
    mesh.compute_vertex_normals()
    
    if verbose:
        print(f"    Reconstruction complete ({recon_time:.2f}s)")
        print(f"    Generated vertices:  {len(mesh.vertices):,}")
        print(f"    Generated triangles: {len(mesh.triangles):,}")
        print()
    
    return mesh


def hybrid_reconstruction(pcd, bpa_radii=None, poisson_depth=9, blend_method='union', verbose=True):
    """
    Hybrid reconstruction: BPA for sharp features, Poisson to fill holes.
    
    Args:
        pcd: Open3D point cloud with normals
        bpa_radii: Custom radii for BPA (None = auto)
        poisson_depth: Depth for Poisson reconstruction
        blend_method: How to combine meshes ('union' or 'bpa_priority')
        verbose: Print progress
    
    Returns:
        Combined triangle mesh
    """
    if verbose:
        print("  " + "-" * 50)
        print("  HYBRID RECONSTRUCTION (BPA + Poisson)")
        print("  " + "-" * 50)
        print(f"    Input points: {len(pcd.points):,}")
        print(f"    Blend method: {blend_method}")
        print()
    
    # Step 1: BPA reconstruction for sharp features
    if verbose:
        print("    Step 1: Ball Pivoting for sharp features...")
    bpa_mesh = ball_pivoting_reconstruction(pcd, radii=bpa_radii, fill_holes=False, verbose=False)
    
    if verbose:
        print(f"      BPA result: {len(bpa_mesh.triangles):,} triangles")
    
    # If BPA failed or produced very little, fall back to Poisson
    if len(bpa_mesh.triangles) < 100:
        if verbose:
            print("      BPA produced too few triangles, using Poisson only")
        mesh, densities = poisson_reconstruction(pcd, depth=poisson_depth, verbose=verbose)
        return mesh, np.asarray(densities)
    
    # Step 2: Check for holes by analyzing boundary edges
    bpa_mesh.compute_vertex_normals()
    
    # Step 3: Poisson reconstruction for gap filling
    if verbose:
        print("    Step 2: Poisson reconstruction for hole filling...")
    poisson_mesh, densities = poisson_reconstruction(pcd, depth=poisson_depth, linear_fit=True, verbose=False)
    
    if verbose:
        print(f"      Poisson result: {len(poisson_mesh.triangles):,} triangles")
    
    # Step 4: Combine meshes
    if verbose:
        print("    Step 3: Combining meshes...")
    
    if blend_method == 'bpa_priority':
        # Use BPA mesh as primary, only add Poisson where BPA has gaps
        # This is a simplified approach - just use BPA with cleaned edges
        combined = bpa_mesh
        combined.remove_degenerate_triangles()
        combined.remove_duplicated_triangles()
        combined.remove_non_manifold_edges()
    else:
        # Union: merge both meshes
        combined = bpa_mesh + poisson_mesh
        combined.remove_duplicated_vertices()
        combined.remove_duplicated_triangles()
        combined.remove_degenerate_triangles()
    
    combined.compute_vertex_normals()
    
    if verbose:
        print(f"      Combined result: {len(combined.triangles):,} triangles")
        print()
    
    return combined, densities


def clean_mesh(mesh, densities, density_threshold=0.01, verbose=True):
    """
    Clean mesh by removing low-density vertices.
    
    Args:
        mesh: Open3D triangle mesh
        densities: Density values from Poisson reconstruction
        density_threshold: Percentile threshold for density filtering (0-1)
        verbose: Print progress
    
    Returns:
        Cleaned mesh
    """
    initial_vertices = len(mesh.vertices)
    initial_triangles = len(mesh.triangles)
    
    if verbose:
        print("  " + "-" * 50)
        print("  MESH CLEANING")
        print("  " + "-" * 50)
        print(f"    Initial vertices:  {initial_vertices:,}")
        print(f"    Initial triangles: {initial_triangles:,}")
    
    # Remove low-density vertices (artifacts from Poisson)
    if density_threshold > 0 and len(densities) > 0:
        density_threshold_value = np.quantile(densities, density_threshold)
        vertices_to_remove = densities < density_threshold_value
        removed_count = np.sum(vertices_to_remove)
        
        if verbose:
            print(f"    Density threshold percentile: {density_threshold} ({density_threshold*100:.0f}%)")
            print(f"    Density threshold value: {density_threshold_value:.6f}")
            print(f"    Removing {removed_count:,} low-density vertices...")
        
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        if verbose:
            print(f"    Vertices after density filter: {len(mesh.vertices):,}")
            print(f"    Triangles after density filter: {len(mesh.triangles):,}")
    
    # Remove degenerate/duplicate geometry
    if verbose:
        print(f"    Removing degenerate triangles...")
    mesh.remove_degenerate_triangles()
    
    if verbose:
        print(f"    Removing duplicated triangles...")
    mesh.remove_duplicated_triangles()
    
    if verbose:
        print(f"    Removing duplicated vertices...")
    mesh.remove_duplicated_vertices()
    
    if verbose:
        print(f"    Removing non-manifold edges...")
    mesh.remove_non_manifold_edges()
    
    vertices_removed = initial_vertices - len(mesh.vertices)
    triangles_removed = initial_triangles - len(mesh.triangles)
    
    if verbose:
        print()
        print(f"    Final vertices:  {len(mesh.vertices):,} (removed {vertices_removed:,})")
        print(f"    Final triangles: {len(mesh.triangles):,} (removed {triangles_removed:,})")
        print()
    
    return mesh


def transfer_colors(mesh, pcd, verbose=True):
    """
    Transfer colors from point cloud to mesh vertices using nearest neighbor.
    
    Args:
        mesh: Open3D triangle mesh
        pcd: Original point cloud with colors
        verbose: Print progress
    
    Returns:
        Mesh with vertex colors
    """
    if verbose:
        print("  " + "-" * 50)
        print("  COLOR TRANSFER")
        print("  " + "-" * 50)
    
    if not pcd.has_colors():
        if verbose:
            print("    [WARNING] Point cloud has no colors, skipping color transfer")
            print("    Mesh will have default coloring")
            print()
        return mesh
    
    if verbose:
        print(f"    Source points: {len(pcd.points):,}")
        print(f"    Target vertices: {len(mesh.vertices):,}")
        print(f"    Building KD-tree...")
    
    transfer_start = time.time()
    
    # Build KD-tree from point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Get mesh vertices
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    
    if verbose:
        print(f"    Transferring colors to {len(mesh_vertices):,} vertices...")
    
    # Find nearest point cloud point for each mesh vertex
    mesh_colors = np.zeros((len(mesh_vertices), 3))
    for i, vertex in enumerate(mesh_vertices):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        mesh_colors[i] = pcd_colors[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    transfer_time = time.time() - transfer_start
    
    if verbose:
        # Analyze transferred colors
        colors_uint8 = (mesh_colors * 255).astype(np.uint8)
        print(f"    Color transfer complete ({transfer_time:.2f}s)")
        print(f"    Color ranges (RGB 0-255):")
        print(f"      R: [{colors_uint8[:,0].min()}, {colors_uint8[:,0].max()}]")
        print(f"      G: [{colors_uint8[:,1].min()}, {colors_uint8[:,1].max()}]")
        print(f"      B: [{colors_uint8[:,2].min()}, {colors_uint8[:,2].max()}]")
        print()
    
    return mesh


def simplify_mesh(mesh, target_triangles=None, target_ratio=None, verbose=True):
    """
    Simplify mesh using quadric decimation.
    
    Args:
        mesh: Open3D triangle mesh
        target_triangles: Target number of triangles (absolute)
        target_ratio: Target ratio of triangles to keep (0-1)
        verbose: Print progress
    
    Returns:
        Simplified mesh
    """
    current_triangles = len(mesh.triangles)
    
    if target_triangles is not None:
        target = target_triangles
    elif target_ratio is not None:
        target = int(current_triangles * target_ratio)
    else:
        return mesh
    
    if verbose:
        print("  " + "-" * 50)
        print("  MESH SIMPLIFICATION")
        print("  " + "-" * 50)
    
    if target >= current_triangles:
        if verbose:
            print(f"    Current triangles: {current_triangles:,}")
            print(f"    Target triangles:  {target:,}")
            print(f"    [SKIPPED] Mesh already at or below target")
            print()
        return mesh
    
    reduction_pct = (1 - target / current_triangles) * 100
    
    if verbose:
        print(f"    Current triangles: {current_triangles:,}")
        print(f"    Target triangles:  {target:,}")
        print(f"    Reduction: {reduction_pct:.1f}%")
        print(f"    Running quadric decimation...")
    
    simplify_start = time.time()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    simplify_time = time.time() - simplify_start
    
    actual_reduction = (1 - len(mesh.triangles) / current_triangles) * 100
    
    if verbose:
        print(f"    Simplification complete ({simplify_time:.2f}s)")
        print(f"    Final triangles: {len(mesh.triangles):,} ({actual_reduction:.1f}% reduction)")
        print()
    
    return mesh


# =============================================================================
# SEGMENTED RECONSTRUCTION FUNCTIONS
# =============================================================================

def segment_planes(pcd, distance_threshold=0.02, min_points_ratio=0.02, 
                   max_planes=10, verbose=True):
    """
    Segment point cloud into planar and non-planar regions using RANSAC.
    
    Args:
        pcd: Open3D point cloud
        distance_threshold: Maximum distance from plane to be considered inlier
        min_points_ratio: Minimum ratio of points for a valid plane (relative to total)
        max_planes: Maximum number of planes to detect
        verbose: Print progress
    
    Returns:
        planes: List of dicts with 'model', 'normal', 'points', 'orientation'
        remaining_pcd: Points that don't belong to any plane
    """
    if verbose:
        print("  " + "-" * 50)
        print("  PLANE SEGMENTATION (RANSAC)")
        print("  " + "-" * 50)
        print(f"    Distance threshold: {distance_threshold}")
        print(f"    Min points ratio: {min_points_ratio}")
        print(f"    Max planes: {max_planes}")
    
    segment_start = time.time()
    
    total_points = len(pcd.points)
    min_points = int(total_points * min_points_ratio)
    
    if verbose:
        print(f"    Total points: {total_points:,}")
        print(f"    Min points per plane: {min_points:,}")
        print()
    
    planes = []
    remaining_pcd = pcd
    
    for i in range(max_planes):
        if len(remaining_pcd.points) < min_points:
            if verbose:
                print(f"    [Plane {i+1}] Stopping: remaining points ({len(remaining_pcd.points):,}) < minimum ({min_points:,})")
            break
        
        # RANSAC plane segmentation
        try:
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
        except RuntimeError:
            if verbose:
                print(f"    [Plane {i+1}] RANSAC failed, stopping")
            break
        
        if len(inliers) < min_points:
            if verbose:
                print(f"    [Plane {i+1}] Found plane with {len(inliers):,} points (< {min_points:,}), stopping")
            break
        
        # Extract plane points
        plane_pcd = remaining_pcd.select_by_index(inliers)
        
        # Calculate plane properties
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        
        # Determine plane orientation
        orientation = classify_plane_orientation(normal)
        
        planes.append({
            'model': plane_model,
            'normal': normal,
            'points': plane_pcd,
            'orientation': orientation
        })
        
        if verbose:
            pct = len(inliers) / total_points * 100
            print(f"    [Plane {i+1}] {orientation:12s} | {len(inliers):,} points ({pct:.1f}%) | normal: [{a:.3f}, {b:.3f}, {c:.3f}]")
        
        # Remove plane points from remaining
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    
    segment_time = time.time() - segment_start
    
    if verbose:
        planar_points = sum(len(p['points'].points) for p in planes)
        remaining_points = len(remaining_pcd.points)
        print()
        print(f"    Segmentation complete ({segment_time:.2f}s)")
        print(f"    Planes found: {len(planes)}")
        print(f"    Planar points: {planar_points:,} ({planar_points/total_points*100:.1f}%)")
        print(f"    Organic points: {remaining_points:,} ({remaining_points/total_points*100:.1f}%)")
        print()
    
    return planes, remaining_pcd


def classify_plane_orientation(normal):
    """Classify plane orientation based on its normal vector."""
    normal = normal / np.linalg.norm(normal)
    
    # Check alignment with principal axes
    abs_normal = np.abs(normal)
    
    if abs_normal[1] > 0.8:  # Y-dominant (up/down)
        return "horizontal" if normal[1] > 0 else "floor/ceil"
    elif abs_normal[0] > 0.7 or abs_normal[2] > 0.7:  # X or Z dominant
        return "wall"
    else:
        return "angled"


def create_plane_mesh(plane_pcd, plane_model, verbose=True):
    """
    Create a clean mesh from planar points by projecting to the fitted plane
    and triangulating the 2D boundary.
    
    Args:
        plane_pcd: Open3D point cloud of planar points
        plane_model: Plane equation [a, b, c, d] where ax + by + cz + d = 0
        verbose: Print progress
    
    Returns:
        Open3D TriangleMesh
    """
    points_3d = np.asarray(plane_pcd.points)
    colors = np.asarray(plane_pcd.colors) if plane_pcd.has_colors() else None
    
    if len(points_3d) < 3:
        return None
    
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # Project points onto the plane
    projected_points = project_points_to_plane(points_3d, plane_model)
    
    # Create local 2D coordinate system on the plane
    points_2d, _, _, _ = create_local_2d_coords(projected_points, normal)
    
    # Triangulate the 2D points
    try:
        if len(points_2d) < 3:
            return None
            
        # Use Delaunay triangulation
        tri = Delaunay(points_2d)
        triangles = tri.simplices
        
        # Filter triangles by edge length to remove long thin triangles at boundaries
        triangles = filter_triangles_by_edge_length(points_2d, triangles, max_ratio=5.0)
        
        if len(triangles) == 0:
            return None
        
    except Exception as e:
        if verbose:
            print(f"      Triangulation failed: {e}")
        return None
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(projected_points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Transfer colors
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Set consistent normals pointing in the plane's normal direction
    vertex_normals = np.tile(normal, (len(projected_points), 1))
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    
    # Clean up the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    
    return mesh


def project_points_to_plane(points, plane_model):
    """Project 3D points onto a plane."""
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # Distance from each point to plane
    distances = (points @ normal + d)
    
    # Project points
    projected = points - np.outer(distances, normal)
    
    return projected


def create_local_2d_coords(points_3d, normal):
    """
    Create a local 2D coordinate system on a plane.
    
    Returns:
        points_2d: Points in local 2D coordinates
        basis_u, basis_v: Basis vectors
        origin: Origin point
    """
    # Find two orthogonal vectors on the plane
    if abs(normal[0]) < 0.9:
        basis_u = np.cross(normal, np.array([1, 0, 0]))
    else:
        basis_u = np.cross(normal, np.array([0, 1, 0]))
    basis_u = basis_u / np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)
    
    # Use centroid as origin
    origin = np.mean(points_3d, axis=0)
    
    # Project to 2D
    centered = points_3d - origin
    points_2d = np.column_stack([
        centered @ basis_u,
        centered @ basis_v
    ])
    
    return points_2d, basis_u, basis_v, origin


def filter_triangles_by_edge_length(points_2d, triangles, max_ratio=5.0):
    """
    Filter out triangles with edges that are too long relative to the median edge length.
    This removes spurious triangles at the boundary of the point set.
    """
    if len(triangles) == 0:
        return triangles
    
    # Calculate all edge lengths
    edge_lengths = []
    for tri in triangles:
        for i in range(3):
            p1 = points_2d[tri[i]]
            p2 = points_2d[tri[(i+1) % 3]]
            edge_lengths.append(np.linalg.norm(p1 - p2))
    
    median_length = np.median(edge_lengths)
    max_length = median_length * max_ratio
    
    # Filter triangles
    valid_triangles = []
    for tri in triangles:
        valid = True
        for i in range(3):
            p1 = points_2d[tri[i]]
            p2 = points_2d[tri[(i+1) % 3]]
            if np.linalg.norm(p1 - p2) > max_length:
                valid = False
                break
        if valid:
            valid_triangles.append(tri)
    
    return np.array(valid_triangles) if valid_triangles else np.array([]).reshape(0, 3)


def create_plane_mesh_alpha(plane_pcd, plane_model, alpha_ratio=2.0, verbose=True):
    """
    Alternative: Create plane mesh using alpha shapes for better boundary handling.
    
    Args:
        plane_pcd: Open3D point cloud of planar points
        plane_model: Plane equation [a, b, c, d]
        alpha_ratio: Multiplier for auto-computed alpha value
        verbose: Print progress
    
    Returns:
        Open3D TriangleMesh
    """
    points_3d = np.asarray(plane_pcd.points)
    
    if len(points_3d) < 4:
        return None
    
    # Project points to plane
    projected_points = project_points_to_plane(points_3d, plane_model)
    
    # Create point cloud from projected points
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    
    if plane_pcd.has_colors():
        projected_pcd.colors = plane_pcd.colors
    
    # Compute alpha value based on point spacing
    distances = projected_pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    alpha = avg_dist * alpha_ratio
    
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            projected_pcd, alpha
        )
    except Exception as e:
        if verbose:
            print(f"      Alpha shape failed: {e}")
        return None
    
    if len(mesh.triangles) == 0:
        return None
    
    # Set normals
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    vertex_normals = np.tile(normal, (len(mesh.vertices), 1))
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    
    return mesh


def reconstruct_organic_region(pcd, poisson_depth=10, density_threshold=0.01, verbose=True):
    """
    Reconstruct non-planar (organic) regions using Poisson reconstruction.
    
    Args:
        pcd: Open3D point cloud
        poisson_depth: Octree depth for Poisson
        density_threshold: Density threshold for cleanup
        verbose: Print progress
    
    Returns:
        Open3D TriangleMesh
    """
    if len(pcd.points) < 100:
        if verbose:
            print(f"    Too few organic points ({len(pcd.points)}), skipping")
        return None
    
    if verbose:
        print("  " + "-" * 50)
        print("  ORGANIC REGION RECONSTRUCTION (Poisson)")
        print("  " + "-" * 50)
        print(f"    Points: {len(pcd.points):,}")
    
    # Estimate normals if not present
    if not pcd.has_normals():
        if verbose:
            print(f"    Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
        )
        pcd.orient_normals_consistent_tangent_plane(k=20)
    
    # Poisson reconstruction
    if verbose:
        print(f"    Running Poisson reconstruction (depth={poisson_depth})...")
    
    recon_start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth,
        scale=1.1,
        linear_fit=True
    )
    recon_time = time.time() - recon_start
    
    if verbose:
        print(f"    Reconstruction complete ({recon_time:.2f}s)")
        print(f"    Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")
    
    # Clean mesh
    if density_threshold > 0 and len(densities) > 0:
        densities = np.asarray(densities)
        threshold_value = np.quantile(densities, density_threshold)
        vertices_to_remove = densities < threshold_value
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        if verbose:
            print(f"    After density cleanup: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    
    if verbose:
        print()
    
    return mesh


def merge_meshes(meshes, verbose=True):
    """
    Merge multiple meshes into one.
    
    Args:
        meshes: List of Open3D TriangleMesh objects
        verbose: Print progress
    
    Returns:
        Combined Open3D TriangleMesh
    """
    if verbose:
        print("  " + "-" * 50)
        print("  MERGING MESHES")
        print("  " + "-" * 50)
    
    valid_meshes = [m for m in meshes if m is not None and len(m.triangles) > 0]
    
    if len(valid_meshes) == 0:
        if verbose:
            print("    [ERROR] No valid meshes to merge")
        return None
    
    if len(valid_meshes) == 1:
        if verbose:
            print(f"    Only one mesh, no merging needed")
            print()
        return valid_meshes[0]
    
    if verbose:
        print(f"    Merging {len(valid_meshes)} meshes...")
        for i, m in enumerate(valid_meshes):
            print(f"      Mesh {i+1}: {len(m.vertices):,} vertices, {len(m.triangles):,} triangles")
    
    merge_start = time.time()
    
    # Combine all meshes
    combined = valid_meshes[0]
    for mesh in valid_meshes[1:]:
        combined += mesh
    
    # Clean up merged mesh
    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    combined.remove_degenerate_triangles()
    
    merge_time = time.time() - merge_start
    
    if verbose:
        print(f"    Merge complete ({merge_time:.2f}s)")
        print(f"    Combined: {len(combined.vertices):,} vertices, {len(combined.triangles):,} triangles")
        print()
    
    return combined


def smooth_mesh_taubin(mesh, iterations=10, lambda_val=0.5, mu=-0.53, verbose=True):
    """
    Apply Taubin smoothing (volume-preserving) to mesh.
    
    Args:
        mesh: Open3D TriangleMesh
        iterations: Number of smoothing iterations
        lambda_val: Lambda parameter
        mu: Mu parameter (should be slightly larger magnitude than lambda)
        verbose: Print progress
    
    Returns:
        Smoothed mesh
    """
    if verbose:
        print("  " + "-" * 50)
        print("  TAUBIN SMOOTHING")
        print("  " + "-" * 50)
        print(f"    Iterations: {iterations}")
        print(f"    Lambda: {lambda_val}, Mu: {mu}")
    
    smooth_start = time.time()
    
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=iterations,
        lambda_filter=lambda_val,
        mu=mu
    )
    
    smooth_time = time.time() - smooth_start
    
    if verbose:
        print(f"    Smoothing complete ({smooth_time:.2f}s)")
        print()
    
    return mesh


def segmented_reconstruction(
    pcd,
    pcd_original,
    plane_distance_threshold=0.02,
    plane_min_ratio=0.02,
    max_planes=10,
    organic_depth=10,
    density_threshold=0.01,
    smooth_organic=True,
    smooth_iterations=5,
    use_alpha_shapes=False,
    verbose=True
):
    """
    Perform segmented reconstruction: planes reconstructed separately from organic geometry.
    
    Args:
        pcd: Preprocessed point cloud with normals
        pcd_original: Original point cloud (for color transfer)
        plane_distance_threshold: RANSAC distance threshold for plane detection
        plane_min_ratio: Minimum ratio of points for valid plane
        max_planes: Maximum number of planes to detect
        organic_depth: Poisson depth for organic regions
        density_threshold: Density threshold for Poisson cleanup
        smooth_organic: Whether to apply Taubin smoothing to organic mesh
        smooth_iterations: Number of smoothing iterations
        use_alpha_shapes: Use alpha shapes instead of Delaunay for planes
        verbose: Print progress
    
    Returns:
        Combined mesh, or None on failure
    """
    if verbose:
        print()
        print("  " + "=" * 50)
        print("  SEGMENTED RECONSTRUCTION MODE")
        print("  " + "=" * 50)
        print()
    
    total_start = time.time()
    meshes = []
    
    # Step 1: Segment planes
    planes, organic_pcd = segment_planes(
        pcd,
        distance_threshold=plane_distance_threshold,
        min_points_ratio=plane_min_ratio,
        max_planes=max_planes,
        verbose=verbose
    )
    
    # Step 2: Create meshes for each plane
    if len(planes) > 0:
        if verbose:
            print("  " + "-" * 50)
            print("  PLANE MESH GENERATION")
            print("  " + "-" * 50)
        
        for i, plane in enumerate(planes):
            if verbose:
                print(f"    Processing plane {i+1}/{len(planes)} ({plane['orientation']})...")
            
            if use_alpha_shapes:
                plane_mesh = create_plane_mesh_alpha(
                    plane['points'],
                    plane['model'],
                    verbose=verbose
                )
            else:
                plane_mesh = create_plane_mesh(
                    plane['points'],
                    plane['model'],
                    verbose=verbose
                )
            
            if plane_mesh is not None and len(plane_mesh.triangles) > 0:
                meshes.append(plane_mesh)
                if verbose:
                    print(f"      Created mesh: {len(plane_mesh.vertices):,} vertices, {len(plane_mesh.triangles):,} triangles")
            else:
                if verbose:
                    print(f"      Failed to create mesh for plane {i+1}")
        
        if verbose:
            print()
    
    # Step 3: Reconstruct organic region
    if len(organic_pcd.points) > 100:
        # Estimate normals for organic region
        organic_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
        )
        organic_pcd.orient_normals_consistent_tangent_plane(k=20)
        
        organic_mesh = reconstruct_organic_region(
            organic_pcd,
            poisson_depth=organic_depth,
            density_threshold=density_threshold,
            verbose=verbose
        )
        
        if organic_mesh is not None:
            # Optionally smooth organic mesh
            if smooth_organic and len(organic_mesh.triangles) > 0:
                organic_mesh = smooth_mesh_taubin(
                    organic_mesh,
                    iterations=smooth_iterations,
                    verbose=verbose
                )
            
            meshes.append(organic_mesh)
    
    # Step 4: Merge all meshes
    if len(meshes) == 0:
        if verbose:
            print("  [ERROR] No meshes were generated")
        return None
    
    combined_mesh = merge_meshes(meshes, verbose=verbose)
    
    if combined_mesh is None:
        return None
    
    # Step 5: Transfer colors from original point cloud
    combined_mesh = transfer_colors(combined_mesh, pcd_original, verbose=verbose)
    
    # Compute final normals
    combined_mesh.compute_vertex_normals()
    
    total_time = time.time() - total_start
    
    if verbose:
        print("  " + "-" * 50)
        print("  SEGMENTED RECONSTRUCTION SUMMARY")
        print("  " + "-" * 50)
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Planes processed: {len(planes)}")
        print(f"    Final mesh:")
        print(f"      Vertices:  {len(combined_mesh.vertices):,}")
        print(f"      Triangles: {len(combined_mesh.triangles):,}")
        print()
    
    return combined_mesh


def orient_mesh_normals(mesh, verbose=True):
    """
    Orient mesh normals consistently (outward-facing).
    
    Args:
        mesh: Open3D TriangleMesh
        verbose: Print progress
    
    Returns:
        Mesh with consistently oriented normals
    """
    if verbose:
        print("  " + "-" * 50)
        print("  ORIENTING NORMALS")
        print("  " + "-" * 50)
    
    # First, ensure the mesh is manifold for orientation to work
    if not mesh.is_orientable():
        if verbose:
            print("    [WARNING] Mesh is not orientable, attempting to fix...")
        mesh.remove_non_manifold_edges()
    
    # Try to orient normals consistently
    try:
        # orient_triangles tries to make all triangles face consistently
        mesh.orient_triangles()
        if verbose:
            print("    Triangles oriented consistently")
    except Exception as e:
        if verbose:
            print(f"    [WARNING] Could not orient triangles: {e}")
    
    # Recompute vertex normals after orientation
    mesh.compute_vertex_normals()
    
    if verbose:
        print("    Vertex normals recomputed")
        print()
    
    return mesh


def flip_mesh_normals(mesh, verbose=True):
    """
    Flip all mesh normals (useful if mesh appears inside-out).
    
    Args:
        mesh: Open3D TriangleMesh
        verbose: Print progress
    
    Returns:
        Mesh with flipped normals
    """
    if verbose:
        print("  " + "-" * 50)
        print("  FLIPPING NORMALS")
        print("  " + "-" * 50)
    
    # Flip triangle winding order (reverses normals)
    triangles = np.asarray(mesh.triangles)
    # Swap second and third vertices of each triangle to flip winding
    triangles[:, [1, 2]] = triangles[:, [2, 1]]
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Recompute vertex normals
    mesh.compute_vertex_normals()
    
    if verbose:
        print("    Triangle winding reversed")
        print("    Vertex normals recomputed")
        print()
    
    return mesh


def make_mesh_double_sided(mesh, verbose=True):
    """
    Make mesh double-sided by duplicating all triangles with flipped normals.
    This ensures the mesh is visible from both sides in Unity/game engines.
    
    Args:
        mesh: Open3D TriangleMesh
        verbose: Print progress
    
    Returns:
        Double-sided mesh
    """
    if verbose:
        print("  " + "-" * 50)
        print("  MAKING MESH DOUBLE-SIDED")
        print("  " + "-" * 50)
    
    initial_triangles = len(mesh.triangles)
    
    # Get current triangles and vertices
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    # Create flipped triangles (reverse winding order)
    flipped_triangles = triangles[:, [0, 2, 1]]  # Swap second and third vertex
    
    # Combine original and flipped triangles
    all_triangles = np.vstack([triangles, flipped_triangles])
    
    # Update mesh
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    
    # Handle vertex colors if present
    if mesh.has_vertex_colors():
        # Colors stay the same since we're using the same vertices
        pass
    
    # Recompute normals
    mesh.compute_vertex_normals()
    
    if verbose:
        print(f"    Original triangles: {initial_triangles:,}")
        print(f"    Final triangles: {len(mesh.triangles):,} (doubled)")
        print()
    
    return mesh


def fix_mesh_normals_consistency(mesh, pcd=None, verbose=True):
    """
    Attempt to fix inconsistent normals by orienting them consistently.
    If a point cloud is provided, orients normals to face away from centroid.
    
    Args:
        mesh: Open3D TriangleMesh
        pcd: Optional point cloud for reference orientation
        verbose: Print progress
    
    Returns:
        Mesh with fixed normals
    """
    if verbose:
        print("  " + "-" * 50)
        print("  FIXING NORMAL CONSISTENCY")
        print("  " + "-" * 50)
    
    # First try Open3D's built-in orientation
    try:
        if mesh.is_orientable():
            mesh.orient_triangles()
            if verbose:
                print("    Mesh was orientable, triangles oriented consistently")
        else:
            if verbose:
                print("    Mesh is not orientable (non-manifold), attempting repair...")
            mesh.remove_non_manifold_edges()
            try:
                mesh.orient_triangles()
                if verbose:
                    print("    Triangles oriented after removing non-manifold edges")
            except:
                if verbose:
                    print("    Could not orient triangles, normals may be inconsistent")
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not orient triangles: {e}")
    
    # Recompute vertex normals
    mesh.compute_vertex_normals()
    
    # If we have a point cloud, use its centroid to help orient normals outward
    if pcd is not None and len(pcd.points) > 0:
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        
        # Check if normals generally point away from centroid
        to_centroid = centroid - vertices
        dots = np.sum(normals * to_centroid, axis=1)
        
        # If more normals point toward centroid than away, flip them all
        if np.sum(dots > 0) > np.sum(dots < 0):
            if verbose:
                print("    Detected inward-facing normals, flipping...")
            mesh = flip_mesh_normals(mesh, verbose=False)
    
    if verbose:
        print("    Normal consistency check complete")
        print()
    
    return mesh


def save_mesh(mesh, output_path, verbose=True):
    """Save mesh to file."""
    if verbose:
        print("  " + "-" * 50)
        print("  SAVING MESH")
        print("  " + "-" * 50)
        print(f"    Output path: {output_path}")
        print(f"    Vertices: {len(mesh.vertices):,}")
        print(f"    Triangles: {len(mesh.triangles):,}")
        print(f"    Has vertex colors: {mesh.has_vertex_colors()}")
        print(f"    Has vertex normals: {mesh.has_vertex_normals()}")
    
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    save_start = time.time()
    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_vertex_colors=True,
        write_vertex_normals=True
    )
    save_time = time.time() - save_start
    
    if success:
        file_size = Path(output_path).stat().st_size
        size_str = f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f} KB"
        if verbose:
            print(f"    Save complete ({save_time:.2f}s)")
            print(f"    File size: {size_str}")
            print()
    else:
        print(f"    [ERROR] Failed to save mesh")
    
    return success


def pointcloud_to_mesh(
    input_path,
    output_path,
    poisson_depth=9,
    density_threshold=0.01,
    voxel_size=None,
    target_triangles=None,
    outlier_std_ratio=2.0,
    reconstruction_method='poisson',
    linear_fit=True,
    bpa_radii=None,
    alpha_value=None,
    segmented=False,
    plane_distance=0.02,
    plane_min_ratio=0.02,
    max_planes=10,
    smooth_organic=True,
    smooth_iterations=5,
    use_alpha_shapes=False,
    flip_normals=False,
    double_sided=False,
    smooth_final=False,
    smooth_final_iterations=5,
    verbose=True
):
    """
    Full pipeline: point cloud to mesh.
    
    Args:
        input_path: Path to input point cloud PLY
        output_path: Path to output mesh (OBJ, PLY, etc.)
        poisson_depth: Octree depth for Poisson reconstruction (6-11)
        density_threshold: Percentile of low-density vertices to remove (0-1)
        voxel_size: Voxel size for downsampling (None = auto, 0 = disabled)
        target_triangles: Target triangle count for simplification (None = no simplification)
        outlier_std_ratio: Standard deviation ratio for outlier removal (lower = more aggressive)
        reconstruction_method: 'poisson', 'bpa' (ball pivoting), or 'alpha' (alpha shapes)
        linear_fit: Use linear fit for Poisson (better adherence to point cloud)
        bpa_radii: Custom radii for ball pivoting (None = auto)
        alpha_value: Custom alpha for alpha shapes (None = auto)
        segmented: Use segmented reconstruction (planes vs organic)
        plane_distance: RANSAC distance threshold for plane detection
        plane_min_ratio: Minimum ratio of points for valid plane
        max_planes: Maximum number of planes to detect
        smooth_organic: Apply Taubin smoothing to organic regions
        smooth_iterations: Number of smoothing iterations
        use_alpha_shapes: Use alpha shapes for plane meshing
        flip_normals: Flip all normals (use if mesh appears inside-out in Unity)
        smooth_final: Apply Taubin smoothing to final mesh
        smooth_final_iterations: Number of final smoothing iterations
        verbose: Print progress
    
    Returns:
        Output mesh path if successful, None otherwise
    """
    total_start = time.time()
    
    # Load point cloud
    pcd = load_point_cloud(input_path, verbose)
    
    if len(pcd.points) == 0:
        print("  [ERROR] Point cloud is empty")
        return None
    
    # Store original for color transfer
    pcd_original = pcd
    
    # Preprocess
    pcd = preprocess_point_cloud(pcd, voxel_size=voxel_size, outlier_std_ratio=outlier_std_ratio, verbose=verbose)
    
    if len(pcd.points) == 0:
        print("  [ERROR] No points remaining after preprocessing")
        print("  [HINT] Try reducing outlier_std_ratio or disabling outlier removal")
        return None
    
    # Choose reconstruction method
    if segmented:
        # Segmented reconstruction: planes vs organic
        mesh = segmented_reconstruction(
            pcd,
            pcd_original,
            plane_distance_threshold=plane_distance,
            plane_min_ratio=plane_min_ratio,
            max_planes=max_planes,
            organic_depth=poisson_depth,
            density_threshold=density_threshold,
            smooth_organic=smooth_organic,
            smooth_iterations=smooth_iterations,
            use_alpha_shapes=use_alpha_shapes,
            verbose=verbose
        )
        
        if mesh is None:
            print("  [ERROR] Segmented reconstruction failed")
            return None
    else:
        # Estimate normals (needed for all methods)
        pcd = estimate_normals(pcd, verbose=verbose)
        
        # Choose reconstruction algorithm
        if reconstruction_method == 'bpa':
            # Ball Pivoting Algorithm with thin geometry support
            mesh = ball_pivoting_reconstruction(pcd, radii=bpa_radii, fill_holes=True, thin_geometry=True, verbose=verbose)
            
            if len(mesh.triangles) == 0:
                print("  [WARNING] BPA produced no triangles, falling back to Poisson")
                mesh, densities = poisson_reconstruction(
                    pcd, depth=poisson_depth, linear_fit=linear_fit, verbose=verbose
                )
                mesh = clean_mesh(mesh, densities, density_threshold=density_threshold, verbose=verbose)
            else:
                # Fix normal consistency for BPA results (common issue)
                mesh = fix_mesh_normals_consistency(mesh, pcd_original, verbose=verbose)
            
        elif reconstruction_method == 'alpha':
            # Alpha Shapes
            mesh = alpha_shape_reconstruction(pcd, alpha=alpha_value, verbose=verbose)
            
            if len(mesh.triangles) == 0:
                print("  [WARNING] Alpha shapes produced no triangles, falling back to Poisson")
                mesh, densities = poisson_reconstruction(
                    pcd, depth=poisson_depth, linear_fit=linear_fit, verbose=verbose
                )
                mesh = clean_mesh(mesh, densities, density_threshold=density_threshold, verbose=verbose)
        
        elif reconstruction_method == 'hybrid':
            # Hybrid: BPA for shape + Poisson for holes
            mesh, _ = hybrid_reconstruction(
                pcd, bpa_radii=bpa_radii, poisson_depth=poisson_depth, verbose=verbose
            )
            
            if len(mesh.triangles) == 0:
                print("  [ERROR] Hybrid reconstruction produced no triangles")
                return None
            
            # Clean mesh without density threshold (densities don't apply to combined mesh)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
                
        else:
            # Standard Poisson reconstruction
            mesh, densities = poisson_reconstruction(
                pcd, depth=poisson_depth, linear_fit=linear_fit, verbose=verbose
            )
            
            if len(mesh.triangles) == 0:
                print("  [ERROR] Poisson reconstruction produced no triangles")
                return None
            
            # Clean mesh
            mesh = clean_mesh(mesh, densities, density_threshold=density_threshold, verbose=verbose)
            
            if len(mesh.triangles) == 0:
                print("  [ERROR] No triangles remaining after cleaning")
                print("  [HINT] Try reducing density_threshold")
                return None
        
        # Transfer colors from original point cloud
        mesh = transfer_colors(mesh, pcd_original, verbose=verbose)
        
        # Compute vertex normals for smooth shading
        if verbose:
            log_step("Computing vertex normals for smooth shading...")
        mesh.compute_vertex_normals()
    
    # Simplify if requested
    if target_triangles is not None:
        mesh = simplify_mesh(mesh, target_triangles=target_triangles, verbose=verbose)
    
    # Optional final smoothing
    if smooth_final and len(mesh.triangles) > 0:
        mesh = smooth_mesh_taubin(mesh, iterations=smooth_final_iterations, verbose=verbose)
    
    # Orient normals consistently
    mesh = orient_mesh_normals(mesh, verbose=verbose)
    
    # Flip normals if requested (useful if mesh appears inside-out)
    if flip_normals:
        mesh = flip_mesh_normals(mesh, verbose=verbose)
    
    # Make double-sided if requested (fixes see-through faces in Unity)
    if double_sided:
        mesh = make_mesh_double_sided(mesh, verbose=verbose)
    
    # Save
    success = save_mesh(mesh, output_path, verbose)
    
    total_time = time.time() - total_start
    
    if verbose and success:
        print("  " + "-" * 50)
        print("  MESH GENERATION SUMMARY")
        print("  " + "-" * 50)
        print(f"    Total processing time: {total_time:.2f}s")
        print(f"    Reconstruction method: {reconstruction_method}")
        print(f"    Final mesh:")
        print(f"      Vertices:  {len(mesh.vertices):,}")
        print(f"      Triangles: {len(mesh.triangles):,}")
        print()
    
    return output_path if success else None


def main():
    parser = argparse.ArgumentParser(
        description="Convert point cloud to mesh using various reconstruction algorithms",
        epilog="""
Examples:
  # Standard Poisson reconstruction (best for organic shapes)
  python pointcloud_to_mesh.py input.ply output.obj
  
  # High quality Poisson with no downsampling
  python pointcloud_to_mesh.py input.ply output.obj --depth 11 --voxel-size 0
  
  # Ball Pivoting Algorithm (better for sharp edges)
  python pointcloud_to_mesh.py input.ply output.obj --method bpa
  
  # Alpha shapes (good for concave objects)
  python pointcloud_to_mesh.py input.ply output.obj --method alpha
  
  # Segmented reconstruction (better for architectural scenes)
  python pointcloud_to_mesh.py input.ply output.obj --segmented
  
  # With final smoothing
  python pointcloud_to_mesh.py input.ply output.obj --smooth
        """
    )
    parser.add_argument("input", help="Input point cloud PLY file")
    parser.add_argument("output", help="Output mesh file (OBJ, PLY, etc.)")
    
    # Basic options
    basic = parser.add_argument_group("Basic Options")
    basic.add_argument(
        "--method", "-m",
        type=str,
        choices=['poisson', 'bpa', 'alpha', 'hybrid'],
        default='poisson',
        help="Reconstruction method: poisson (watertight), bpa (sharp edges), alpha (concave), hybrid (bpa+poisson) (default: poisson)"
    )
    basic.add_argument(
        "--depth", "-d",
        type=int,
        default=9,
        help="Poisson octree depth (6-11, higher=more detail, default: 9)"
    )
    basic.add_argument(
        "--density-threshold", "-t",
        type=float,
        default=0.01,
        help="Percentile of low-density vertices to remove (0-1, default: 0.01)"
    )
    basic.add_argument(
        "--voxel-size", "-v",
        type=float,
        default=None,
        help="Voxel size for downsampling (default: auto-calculate, 0=disabled)"
    )
    basic.add_argument(
        "--simplify", "-s",
        type=int,
        default=None,
        help="Target number of triangles for simplification"
    )
    basic.add_argument(
        "--outlier-std", "-r",
        type=float,
        default=2.0,
        help="Outlier removal std ratio (lower=more aggressive, default: 2.0)"
    )
    basic.add_argument(
        "--linear-fit",
        action="store_true",
        default=True,
        help="Use linear fit for Poisson (better adherence to points, default: True)"
    )
    basic.add_argument(
        "--no-linear-fit",
        action="store_true",
        help="Disable linear fit for Poisson"
    )
    
    # BPA/Alpha options
    alt = parser.add_argument_group("Alternative Reconstruction Options")
    alt.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha value for alpha shapes (default: auto based on point spacing)"
    )
    
    # Segmented reconstruction options
    seg = parser.add_argument_group("Segmented Reconstruction (for architectural scenes)")
    seg.add_argument(
        "--segmented", "-S",
        action="store_true",
        help="Enable segmented reconstruction (planes processed separately from organic geometry)"
    )
    seg.add_argument(
        "--plane-distance",
        type=float,
        default=0.02,
        help="RANSAC distance threshold for plane detection (default: 0.02)"
    )
    seg.add_argument(
        "--plane-min-ratio",
        type=float,
        default=0.02,
        help="Minimum ratio of points for valid plane (default: 0.02 = 2%%)"
    )
    seg.add_argument(
        "--max-planes",
        type=int,
        default=10,
        help="Maximum number of planes to detect (default: 10)"
    )
    seg.add_argument(
        "--no-smooth-organic",
        action="store_true",
        help="Disable Taubin smoothing on organic regions"
    )
    seg.add_argument(
        "--smooth-iterations",
        type=int,
        default=5,
        help="Number of Taubin smoothing iterations (default: 5)"
    )
    seg.add_argument(
        "--use-alpha-shapes",
        action="store_true",
        help="Use alpha shapes instead of Delaunay for plane meshing"
    )
    
    # Output options
    out = parser.add_argument_group("Output Options")
    out.add_argument(
        "--flip-normals",
        action="store_true",
        help="Flip all mesh normals (use if mesh appears inside-out in Unity)"
    )
    out.add_argument(
        "--double-sided",
        action="store_true",
        help="Make mesh double-sided (fixes see-through faces in Unity, doubles triangle count)"
    )
    out.add_argument(
        "--smooth",
        action="store_true",
        help="Apply Taubin smoothing to final mesh"
    )
    out.add_argument(
        "--smooth-final-iterations",
        type=int,
        default=5,
        help="Number of final smoothing iterations (default: 5)"
    )
    out.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    pointcloud_to_mesh(
        args.input,
        args.output,
        poisson_depth=args.depth,
        density_threshold=args.density_threshold,
        voxel_size=args.voxel_size,
        target_triangles=args.simplify,
        outlier_std_ratio=args.outlier_std,
        reconstruction_method=args.method,
        linear_fit=not args.no_linear_fit,
        alpha_value=args.alpha,
        segmented=args.segmented,
        plane_distance=args.plane_distance,
        plane_min_ratio=args.plane_min_ratio,
        max_planes=args.max_planes,
        smooth_organic=not args.no_smooth_organic,
        smooth_iterations=args.smooth_iterations,
        use_alpha_shapes=args.use_alpha_shapes,
        flip_normals=args.flip_normals,
        double_sided=args.double_sided,
        smooth_final=args.smooth,
        smooth_final_iterations=args.smooth_final_iterations,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
