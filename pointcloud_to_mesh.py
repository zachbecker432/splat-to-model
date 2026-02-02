"""
Point Cloud to Mesh Converter (using Open3D)

Converts a point cloud PLY file to a mesh using Ball Pivoting Algorithm (BPA).
Designed to be called as a module from run_pipeline.py in Docker.
"""

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


def preprocess_point_cloud(pcd, voxel_size=None, remove_outliers=True, outlier_std_ratio=2.0, 
                           outlier_neighbors=30, verbose=True):
    """
    Preprocess point cloud: downsample and remove outliers.
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
        auto_voxel = diagonal * 0.002
        min_effective_voxel = avg_spacing * 2.0
        voxel_size = max(auto_voxel, min_effective_voxel)
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
    
    total_reduction = (1 - len(pcd.points) / initial_points) * 100 if initial_points > 0 else 0
    if verbose:
        print(f"    Total preprocessing reduction: {initial_points:,} -> {len(pcd.points):,} ({total_reduction:.1f}%)")
        print()
    
    return pcd


def estimate_normals(pcd, search_radius=None, max_nn=100, orient_k=30, verbose=True):
    """Estimate normals for point cloud."""
    if verbose:
        print("  " + "-" * 50)
        print("  NORMAL ESTIMATION")
        print("  " + "-" * 50)
    
    if search_radius is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        search_radius = avg_dist * 5.0
        if verbose:
            print(f"    Average point spacing: {avg_dist:.6f}")
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
        print(f"    Orienting normals consistently (k={orient_k})...")
    
    orient_start = time.time()
    pcd.orient_normals_consistent_tangent_plane(k=orient_k)
    orient_time = time.time() - orient_start
    
    if verbose:
        print(f"    Normal orientation complete ({orient_time:.2f}s)")
        print()
    
    return pcd


def ball_pivoting_reconstruction(pcd, radii=None, fill_holes=True, thin_geometry=True, verbose=True):
    """Perform Ball Pivoting Algorithm (BPA) surface reconstruction."""
    if verbose:
        print("  " + "-" * 50)
        print("  BALL PIVOTING RECONSTRUCTION")
        print("  " + "-" * 50)
        print(f"    Input points: {len(pcd.points):,}")
        print(f"    Thin geometry mode: {thin_geometry}")
    
    if radii is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        min_dist = np.percentile(distances, 5)
        max_dist = np.percentile(distances, 95)
        
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
        
        if thin_geometry:
            thin_radii = [
                min_dist * 0.5,
                min_dist,
                avg_dist * 0.25,
                avg_dist * 0.33,
            ]
            radii = thin_radii + radii
        
        radii = sorted(set(r for r in radii if r > 0.0001))
        
        if verbose:
            print(f"    Min point spacing (5th pct): {min_dist:.6f}")
            print(f"    Average point spacing: {avg_dist:.6f}")
            print(f"    Max point spacing (95th pct): {max_dist:.6f}")
            print(f"    Using {len(radii)} radii")
    
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
    
    if fill_holes and len(mesh.triangles) > 0:
        mesh = fill_mesh_holes(mesh, verbose=verbose)
    
    if verbose:
        print()
    
    return mesh


def fill_mesh_holes(mesh, hole_size=100, verbose=True):
    """Fill holes in a mesh."""
    if verbose:
        print(f"    Attempting to fill holes...")
    
    initial_triangles = len(mesh.triangles)
    mesh.compute_adjacency_list()
    
    try:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
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


def transfer_colors(mesh, pcd, verbose=True):
    """Transfer colors from point cloud to mesh vertices."""
    if verbose:
        print("  " + "-" * 50)
        print("  COLOR TRANSFER")
        print("  " + "-" * 50)
    
    if not pcd.has_colors():
        if verbose:
            print("    [WARNING] Point cloud has no colors, skipping")
            print()
        return mesh
    
    if verbose:
        print(f"    Source points: {len(pcd.points):,}")
        print(f"    Target vertices: {len(mesh.vertices):,}")
    
    transfer_start = time.time()
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    
    mesh_colors = np.zeros((len(mesh_vertices), 3))
    for i, vertex in enumerate(mesh_vertices):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        mesh_colors[i] = pcd_colors[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    transfer_time = time.time() - transfer_start
    
    if verbose:
        print(f"    Color transfer complete ({transfer_time:.2f}s)")
        print()
    
    return mesh


def simplify_mesh(mesh, target_triangles, verbose=True):
    """Simplify mesh using quadric decimation."""
    current_triangles = len(mesh.triangles)
    
    if verbose:
        print("  " + "-" * 50)
        print("  MESH SIMPLIFICATION")
        print("  " + "-" * 50)
    
    if target_triangles >= current_triangles:
        if verbose:
            print(f"    [SKIPPED] Mesh already at or below target")
            print()
        return mesh
    
    reduction_pct = (1 - target_triangles / current_triangles) * 100
    
    if verbose:
        print(f"    Current triangles: {current_triangles:,}")
        print(f"    Target triangles:  {target_triangles:,}")
        print(f"    Reduction: {reduction_pct:.1f}%")
    
    simplify_start = time.time()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    simplify_time = time.time() - simplify_start
    
    if verbose:
        print(f"    Simplification complete ({simplify_time:.2f}s)")
        print(f"    Final triangles: {len(mesh.triangles):,}")
        print()
    
    return mesh


def smooth_mesh_taubin(mesh, iterations=10, verbose=True):
    """Apply Taubin smoothing to mesh."""
    if verbose:
        print("  " + "-" * 50)
        print("  TAUBIN SMOOTHING")
        print("  " + "-" * 50)
        print(f"    Iterations: {iterations}")
    
    smooth_start = time.time()
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=iterations,
        lambda_filter=0.5,
        mu=-0.53
    )
    smooth_time = time.time() - smooth_start
    
    if verbose:
        print(f"    Smoothing complete ({smooth_time:.2f}s)")
        print()
    
    return mesh


def orient_mesh_normals(mesh, verbose=True):
    """Orient mesh normals consistently."""
    if verbose:
        print("  " + "-" * 50)
        print("  ORIENTING NORMALS")
        print("  " + "-" * 50)
    
    if not mesh.is_orientable():
        if verbose:
            print("    [WARNING] Mesh is not orientable, attempting to fix...")
        mesh.remove_non_manifold_edges()
    
    try:
        mesh.orient_triangles()
        if verbose:
            print("    Triangles oriented consistently")
    except Exception as e:
        if verbose:
            print(f"    [WARNING] Could not orient triangles: {e}")
    
    mesh.compute_vertex_normals()
    
    if verbose:
        print("    Vertex normals recomputed")
        print()
    
    return mesh


def flip_mesh_normals(mesh, verbose=True):
    """Flip all mesh normals."""
    if verbose:
        print("  " + "-" * 50)
        print("  FLIPPING NORMALS")
        print("  " + "-" * 50)
    
    triangles = np.asarray(mesh.triangles)
    triangles[:, [1, 2]] = triangles[:, [2, 1]]
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    if verbose:
        print("    Triangle winding reversed")
        print()
    
    return mesh


def make_mesh_double_sided(mesh, verbose=True):
    """Make mesh double-sided by duplicating triangles with flipped normals."""
    if verbose:
        print("  " + "-" * 50)
        print("  MAKING MESH DOUBLE-SIDED")
        print("  " + "-" * 50)
    
    initial_triangles = len(mesh.triangles)
    triangles = np.asarray(mesh.triangles)
    flipped_triangles = triangles[:, [0, 2, 1]]
    all_triangles = np.vstack([triangles, flipped_triangles])
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    mesh.compute_vertex_normals()
    
    if verbose:
        print(f"    Original triangles: {initial_triangles:,}")
        print(f"    Final triangles: {len(mesh.triangles):,} (doubled)")
        print()
    
    return mesh


def fix_mesh_normals_consistency(mesh, pcd=None, verbose=True):
    """Fix inconsistent normals."""
    if verbose:
        print("  " + "-" * 50)
        print("  FIXING NORMAL CONSISTENCY")
        print("  " + "-" * 50)
    
    try:
        if mesh.is_orientable():
            mesh.orient_triangles()
            if verbose:
                print("    Mesh oriented consistently")
        else:
            if verbose:
                print("    Mesh is not orientable, attempting repair...")
            mesh.remove_non_manifold_edges()
            try:
                mesh.orient_triangles()
            except:
                pass
    except Exception as e:
        if verbose:
            print(f"    Warning: {e}")
    
    mesh.compute_vertex_normals()
    
    if pcd is not None and len(pcd.points) > 0:
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        to_centroid = centroid - vertices
        dots = np.sum(normals * to_centroid, axis=1)
        
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
    voxel_size=None,
    target_triangles=None,
    outlier_std_ratio=2.0,
    thin_geometry=True,
    flip_normals=False,
    double_sided=False,
    smooth_final=False,
    smooth_iterations=5,
    verbose=True
):
    """
    Full pipeline: point cloud to mesh using Ball Pivoting Algorithm.
    """
    total_start = time.time()
    
    # Load point cloud
    pcd = load_point_cloud(input_path, verbose)
    
    if len(pcd.points) == 0:
        print("  [ERROR] Point cloud is empty")
        return None
    
    pcd_original = pcd
    
    # Preprocess
    pcd = preprocess_point_cloud(pcd, voxel_size=voxel_size, outlier_std_ratio=outlier_std_ratio, verbose=verbose)
    
    if len(pcd.points) == 0:
        print("  [ERROR] No points remaining after preprocessing")
        return None
    
    # Estimate normals
    pcd = estimate_normals(pcd, verbose=verbose)
    
    # BPA reconstruction
    mesh = ball_pivoting_reconstruction(pcd, fill_holes=True, thin_geometry=thin_geometry, verbose=verbose)
    
    if len(mesh.triangles) == 0:
        print("  [ERROR] BPA produced no triangles")
        return None
    
    # Fix normals
    mesh = fix_mesh_normals_consistency(mesh, pcd_original, verbose=verbose)
    
    # Transfer colors
    mesh = transfer_colors(mesh, pcd_original, verbose=verbose)
    mesh.compute_vertex_normals()
    
    # Simplify if requested
    if target_triangles is not None:
        mesh = simplify_mesh(mesh, target_triangles=target_triangles, verbose=verbose)
    
    # Smooth if requested
    if smooth_final and len(mesh.triangles) > 0:
        mesh = smooth_mesh_taubin(mesh, iterations=smooth_iterations, verbose=verbose)
    
    # Orient normals
    mesh = orient_mesh_normals(mesh, verbose=verbose)
    
    # Flip normals if requested
    if flip_normals:
        mesh = flip_mesh_normals(mesh, verbose=verbose)
    
    # Double-sided if requested
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
        print(f"    Final mesh:")
        print(f"      Vertices:  {len(mesh.vertices):,}")
        print(f"      Triangles: {len(mesh.triangles):,}")
        print()
    
    return output_path if success else None
