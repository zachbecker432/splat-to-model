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
from pathlib import Path


def load_point_cloud(input_path, verbose=True):
    """Load point cloud from PLY file."""
    if verbose:
        print(f"Loading point cloud from: {input_path}")
    
    pcd = o3d.io.read_point_cloud(str(input_path))
    
    if verbose:
        print(f"  Points: {len(pcd.points)}")
        print(f"  Has colors: {pcd.has_colors()}")
        print(f"  Has normals: {pcd.has_normals()}")
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size=None, remove_outliers=True, verbose=True):
    """
    Preprocess point cloud: downsample and remove outliers.
    
    Args:
        pcd: Open3D point cloud
        voxel_size: Voxel size for downsampling (None = no downsampling)
        remove_outliers: Whether to remove statistical outliers
        verbose: Print progress
    
    Returns:
        Preprocessed point cloud
    """
    if verbose:
        print("Preprocessing point cloud...")
    
    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        if verbose:
            print(f"  Downsampling with voxel size: {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
        if verbose:
            print(f"  Points after downsampling: {len(pcd.points)}")
    
    # Remove statistical outliers
    if remove_outliers:
        if verbose:
            print("  Removing statistical outliers...")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if verbose:
            print(f"  Points after outlier removal: {len(pcd.points)}")
    
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
        print("Estimating normals...")
    
    # Auto-calculate search radius based on point cloud extent if not provided
    if search_radius is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        # Use ~1% of the smallest dimension as search radius
        search_radius = min(extent) * 0.01
        if verbose:
            print(f"  Auto search radius: {search_radius:.4f}")
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius,
            max_nn=max_nn
        )
    )
    
    # Orient normals consistently (important for Poisson reconstruction)
    if verbose:
        print("  Orienting normals consistently...")
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
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
        print(f"Running Poisson reconstruction (depth={depth})...")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit
    )
    
    if verbose:
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Triangles: {len(mesh.triangles)}")
    
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
    if verbose:
        print("Cleaning mesh...")
    
    # Remove low-density vertices (artifacts from Poisson)
    if density_threshold > 0 and len(densities) > 0:
        density_threshold_value = np.quantile(densities, density_threshold)
        vertices_to_remove = densities < density_threshold_value
        mesh.remove_vertices_by_mask(vertices_to_remove)
        if verbose:
            print(f"  Removed low-density vertices (threshold: {density_threshold_value:.4f})")
            print(f"  Vertices after: {len(mesh.vertices)}")
            print(f"  Triangles after: {len(mesh.triangles)}")
    
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    
    # Remove duplicated triangles
    mesh.remove_duplicated_triangles()
    
    # Remove duplicated vertices
    mesh.remove_duplicated_vertices()
    
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    
    if verbose:
        print(f"  Final vertices: {len(mesh.vertices)}")
        print(f"  Final triangles: {len(mesh.triangles)}")
    
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
    if not pcd.has_colors():
        if verbose:
            print("Point cloud has no colors, skipping color transfer")
        return mesh
    
    if verbose:
        print("Transferring colors from point cloud to mesh...")
    
    # Build KD-tree from point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Get mesh vertices
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    
    # Find nearest point cloud point for each mesh vertex
    mesh_colors = np.zeros((len(mesh_vertices), 3))
    for i, vertex in enumerate(mesh_vertices):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        mesh_colors[i] = pcd_colors[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    if verbose:
        print("  Color transfer complete")
    
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
    
    if target >= current_triangles:
        if verbose:
            print(f"Mesh already has {current_triangles} triangles, no simplification needed")
        return mesh
    
    if verbose:
        print(f"Simplifying mesh from {current_triangles} to {target} triangles...")
    
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    
    if verbose:
        print(f"  Final triangles: {len(mesh.triangles)}")
    
    return mesh


def save_mesh(mesh, output_path, verbose=True):
    """Save mesh to file."""
    if verbose:
        print(f"Saving mesh to: {output_path}")
    
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_vertex_colors=True,
        write_vertex_normals=True
    )
    
    if success and verbose:
        print("  Mesh saved successfully")
    elif not success:
        print("  ERROR: Failed to save mesh")
    
    return success


def pointcloud_to_mesh(
    input_path,
    output_path,
    poisson_depth=9,
    density_threshold=0.01,
    voxel_size=None,
    target_triangles=None,
    verbose=True
):
    """
    Full pipeline: point cloud to mesh.
    
    Args:
        input_path: Path to input point cloud PLY
        output_path: Path to output mesh (OBJ, PLY, etc.)
        poisson_depth: Octree depth for Poisson reconstruction (6-11)
        density_threshold: Percentile of low-density vertices to remove (0-1)
        voxel_size: Voxel size for downsampling (None = no downsampling)
        target_triangles: Target triangle count for simplification (None = no simplification)
        verbose: Print progress
    
    Returns:
        Output mesh path if successful, None otherwise
    """
    # Load point cloud
    pcd = load_point_cloud(input_path, verbose)
    
    if len(pcd.points) == 0:
        print("ERROR: Point cloud is empty")
        return None
    
    # Store original for color transfer
    pcd_original = pcd
    
    # Preprocess
    pcd = preprocess_point_cloud(pcd, voxel_size=voxel_size, verbose=verbose)
    
    # Estimate normals
    pcd = estimate_normals(pcd, verbose=verbose)
    
    # Poisson reconstruction
    mesh, densities = poisson_reconstruction(pcd, depth=poisson_depth, verbose=verbose)
    
    # Clean mesh
    mesh = clean_mesh(mesh, densities, density_threshold=density_threshold, verbose=verbose)
    
    # Transfer colors from original point cloud
    mesh = transfer_colors(mesh, pcd_original, verbose=verbose)
    
    # Simplify if requested
    if target_triangles is not None:
        mesh = simplify_mesh(mesh, target_triangles=target_triangles, verbose=verbose)
    
    # Compute vertex normals for smooth shading
    mesh.compute_vertex_normals()
    
    # Save
    success = save_mesh(mesh, output_path, verbose)
    
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
        help="Voxel size for downsampling (default: no downsampling)"
    )
    parser.add_argument(
        "--simplify", "-s",
        type=int,
        default=None,
        help="Target number of triangles for simplification"
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
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
