"""
Point Cloud to Mesh Converter (using Open3D)

Converts a point cloud PLY file to a mesh using Poisson surface reconstruction.
Replaces the manual MeshLab workflow with a scriptable Python solution.

Usage:
    python pointcloud_to_mesh.py input.ply output.obj [options]
"""

import argparse
import numpy as np
import open3d as o3d
import time
from pathlib import Path


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


def preprocess_point_cloud(pcd, voxel_size=None, remove_outliers=True, outlier_std_ratio=2.0, verbose=True):
    """
    Preprocess point cloud: downsample and remove outliers.
    
    Args:
        pcd: Open3D point cloud
        voxel_size: Voxel size for downsampling (None = auto-calculate, 0 = disabled)
        remove_outliers: Whether to remove statistical outliers
        outlier_std_ratio: Standard deviation ratio for outlier removal (lower = more aggressive)
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
    
    if verbose:
        print(f"    Initial points: {initial_points:,}")
        print(f"    Bounding box extent: [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]")
        print(f"    Diagonal: {diagonal:.4f}")
    
    # Auto-calculate voxel size if not specified (but not if explicitly set to 0)
    if voxel_size is None:
        # Use ~0.5% of diagonal as voxel size for reasonable downsampling
        voxel_size = diagonal * 0.005
        if verbose:
            print(f"    Auto-calculated voxel size: {voxel_size:.6f}")
    
    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        if verbose:
            print(f"    Downsampling with voxel size: {voxel_size:.6f}...")
        downsample_start = time.time()
        points_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size)
        downsample_time = time.time() - downsample_start
        reduction_pct = (1 - len(pcd.points) / points_before) * 100
        if verbose:
            print(f"    Points after downsampling: {len(pcd.points):,} ({reduction_pct:.1f}% reduction, {downsample_time:.2f}s)")
    else:
        if verbose:
            print(f"    Downsampling: disabled")
    
    # Remove statistical outliers
    if remove_outliers:
        if verbose:
            print(f"    Removing outliers (neighbors=30, std_ratio={outlier_std_ratio})...")
        outlier_start = time.time()
        points_before = len(pcd.points)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=outlier_std_ratio)
        outlier_time = time.time() - outlier_start
        removed = points_before - len(pcd.points)
        removal_pct = (removed / points_before) * 100
        if verbose:
            print(f"    Points after outlier removal: {len(pcd.points):,} (removed {removed:,} = {removal_pct:.1f}%, {outlier_time:.2f}s)")
    
    total_reduction = (1 - len(pcd.points) / initial_points) * 100
    if verbose:
        print(f"    Total preprocessing reduction: {initial_points:,} -> {len(pcd.points):,} ({total_reduction:.1f}%)")
        print()
    
    return pcd


def estimate_normals(pcd, search_radius=None, max_nn=30, verbose=True):
    """
    Estimate normals for point cloud.
    
    Args:
        pcd: Open3D point cloud
        search_radius: Search radius for normal estimation (None = auto)
        max_nn: Maximum number of neighbors to consider
        verbose: Print progress
    
    Returns:
        Point cloud with normals
    """
    if verbose:
        print("  " + "-" * 50)
        print("  NORMAL ESTIMATION")
        print("  " + "-" * 50)
    
    # Auto-calculate search radius based on point cloud extent if not provided
    if search_radius is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        # Use ~1% of the smallest dimension as search radius
        search_radius = min(extent) * 0.01
        if verbose:
            print(f"    Auto-calculated search radius: {search_radius:.6f}")
    
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
    if verbose:
        print(f"    Orienting normals consistently (k=15)...")
    
    orient_start = time.time()
    pcd.orient_normals_consistent_tangent_plane(k=15)
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
    
    # Estimate normals
    pcd = estimate_normals(pcd, verbose=verbose)
    
    # Poisson reconstruction
    mesh, densities = poisson_reconstruction(pcd, depth=poisson_depth, verbose=verbose)
    
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
    
    # Simplify if requested
    if target_triangles is not None:
        mesh = simplify_mesh(mesh, target_triangles=target_triangles, verbose=verbose)
    
    # Compute vertex normals for smooth shading
    if verbose:
        log_step("Computing vertex normals for smooth shading...")
    mesh.compute_vertex_normals()
    
    # Save
    success = save_mesh(mesh, output_path, verbose)
    
    total_time = time.time() - total_start
    
    if verbose and success:
        print("  " + "-" * 50)
        print("  MESH GENERATION SUMMARY")
        print("  " + "-" * 50)
        print(f"    Total processing time: {total_time:.2f}s")
        print(f"    Final mesh:")
        print(f"      Vertices:  {len(mesh.vertices):,}")
        print(f"      Triangles: {len(mesh.triangles):,}")
        print()
    
    return output_path if success else None


def main():
    parser = argparse.ArgumentParser(
        description="Convert point cloud to mesh using Poisson reconstruction"
    )
    parser.add_argument("input", help="Input point cloud PLY file")
    parser.add_argument("output", help="Output mesh file (OBJ, PLY, etc.)")
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=9,
        help="Poisson octree depth (6-11, higher=more detail, default: 9)"
    )
    parser.add_argument(
        "--density-threshold", "-t",
        type=float,
        default=0.01,
        help="Percentile of low-density vertices to remove (0-1, default: 0.01)"
    )
    parser.add_argument(
        "--voxel-size", "-v",
        type=float,
        default=None,
        help="Voxel size for downsampling (default: auto-calculate, 0=disabled)"
    )
    parser.add_argument(
        "--simplify", "-s",
        type=int,
        default=None,
        help="Target number of triangles for simplification"
    )
    parser.add_argument(
        "--outlier-std", "-r",
        type=float,
        default=2.0,
        help="Outlier removal std ratio (lower=more aggressive, default: 2.0)"
    )
    parser.add_argument(
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
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
