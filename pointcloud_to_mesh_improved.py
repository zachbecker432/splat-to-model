"""
IMPROVED Point Cloud to Mesh Converter

Enhanced version with better quality output:
- Multi-stage outlier removal
- RANSAC plane detection for walls/floors (sharp edges)
- Ball Pivoting Algorithm option (preserves edges better than Poisson)
- Hybrid meshing (planes + Poisson for complex areas)
- Better normal estimation

Usage:
    python pointcloud_to_mesh_improved.py input.ply output.obj [options]
"""

import argparse
import numpy as np
import open3d as o3d
import time
from pathlib import Path
from collections import defaultdict


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


def aggressive_outlier_removal(pcd, verbose=True):
    """
    Multi-stage outlier removal for cleaner point clouds.
    
    Stage 1: Statistical outlier removal (global)
    Stage 2: Radius outlier removal (local density)
    Stage 3: Remove isolated clusters
    """
    if verbose:
        print("  " + "-" * 50)
        print("  AGGRESSIVE OUTLIER REMOVAL")
        print("  " + "-" * 50)
        print(f"    Initial points: {len(pcd.points):,}")
    
    # Get bounding box for scale-aware parameters
    bbox = pcd.get_axis_aligned_bounding_box()
    diagonal = np.linalg.norm(bbox.get_extent())
    
    # Stage 1: Statistical outlier removal (more aggressive)
    if verbose:
        print(f"    Stage 1: Statistical outlier removal...")
    start = time.time()
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
    if verbose:
        print(f"      Points after: {len(pcd.points):,} ({time.time()-start:.2f}s)")
    
    # Stage 2: Radius outlier removal
    # Remove points that don't have enough neighbors within a radius
    radius = diagonal * 0.01  # 1% of scene diagonal
    min_neighbors = 10
    if verbose:
        print(f"    Stage 2: Radius outlier removal (r={radius:.4f}, min_neighbors={min_neighbors})...")
    start = time.time()
    pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    if verbose:
        print(f"      Points after: {len(pcd.points):,} ({time.time()-start:.2f}s)")
    
    # Stage 3: Remove small isolated clusters
    if verbose:
        print(f"    Stage 3: Removing isolated clusters...")
    start = time.time()
    
    # DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=diagonal * 0.02, min_points=20))
    
    if len(labels) > 0 and labels.max() >= 0:
        # Count points in each cluster
        cluster_counts = np.bincount(labels[labels >= 0])
        
        if len(cluster_counts) > 0:
            # Keep only clusters with at least 5% of points in the largest cluster
            max_cluster_size = cluster_counts.max()
            min_cluster_size = max(100, int(max_cluster_size * 0.05))
            
            # Create mask for points in large enough clusters
            valid_clusters = set(np.where(cluster_counts >= min_cluster_size)[0])
            mask = np.array([l in valid_clusters for l in labels])
            
            # Filter point cloud
            pcd = pcd.select_by_index(np.where(mask)[0])
            
            if verbose:
                print(f"      Found {len(cluster_counts)} clusters, kept {len(valid_clusters)} (min size: {min_cluster_size})")
    
    if verbose:
        print(f"      Points after: {len(pcd.points):,} ({time.time()-start:.2f}s)")
        print()
    
    return pcd


def detect_planes_ransac(pcd, distance_threshold=None, min_points_ratio=0.05, 
                         max_planes=10, verbose=True):
    """
    Detect planar regions using RANSAC.
    
    Returns list of (plane_model, inlier_indices) for each detected plane.
    These can be used to create sharp, flat surfaces for walls/floors.
    """
    if verbose:
        print("  " + "-" * 50)
        print("  RANSAC PLANE DETECTION")
        print("  " + "-" * 50)
    
    # Auto-calculate distance threshold based on point cloud scale
    if distance_threshold is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        diagonal = np.linalg.norm(bbox.get_extent())
        distance_threshold = diagonal * 0.005  # 0.5% of diagonal
    
    if verbose:
        print(f"    Distance threshold: {distance_threshold:.6f}")
        print(f"    Max planes to detect: {max_planes}")
    
    planes = []
    remaining_pcd = pcd
    total_points = len(pcd.points)
    min_points = int(total_points * min_points_ratio)
    
    for i in range(max_planes):
        if len(remaining_pcd.points) < min_points:
            break
        
        # RANSAC plane segmentation
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < min_points:
            break
        
        [a, b, c, d] = plane_model
        
        if verbose:
            pct = len(inliers) / total_points * 100
            print(f"    Plane {i+1}: {len(inliers):,} points ({pct:.1f}%)")
            print(f"      Normal: [{a:.4f}, {b:.4f}, {c:.4f}], d={d:.4f}")
        
        # Store plane info
        planes.append({
            'model': plane_model,
            'inliers': inliers,
            'points': remaining_pcd.select_by_index(inliers)
        })
        
        # Remove plane points from remaining cloud
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    
    if verbose:
        print(f"    Detected {len(planes)} planes")
        print(f"    Remaining non-planar points: {len(remaining_pcd.points):,}")
        print()
    
    return planes, remaining_pcd


def create_plane_mesh(plane_pcd, plane_model, verbose=True):
    """
    Create a flat mesh from a planar point cloud region.
    Uses alpha shapes to get the boundary, then creates a clean flat mesh.
    """
    if len(plane_pcd.points) < 10:
        return None
    
    # Project points onto the plane for cleaner 2D boundary
    points = np.asarray(plane_pcd.points)
    colors = np.asarray(plane_pcd.colors) if plane_pcd.has_colors() else None
    
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # Project points onto plane
    # distance = dot(point, normal) + d
    distances = np.dot(points, normal) + d
    projected_points = points - np.outer(distances, normal)
    
    # Create point cloud from projected points
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    if colors is not None:
        projected_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals (all should point same direction for plane)
    projected_pcd.normals = o3d.utility.Vector3dVector(
        np.tile(normal, (len(projected_points), 1))
    )
    
    # Use alpha shapes for boundary-respecting mesh
    try:
        # Calculate appropriate alpha value
        bbox = projected_pcd.get_axis_aligned_bounding_box()
        diagonal = np.linalg.norm(bbox.get_extent())
        alpha = diagonal * 0.05  # Adjust for tightness
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            projected_pcd, alpha=alpha
        )
        
        if len(mesh.triangles) > 0:
            # Transfer colors
            if colors is not None:
                mesh = transfer_colors_to_mesh(mesh, projected_pcd, verbose=False)
            mesh.compute_vertex_normals()
            return mesh
    except Exception as e:
        if verbose:
            print(f"    [WARNING] Alpha shape failed: {e}")
    
    return None


def estimate_normals_improved(pcd, verbose=True):
    """
    Improved normal estimation with consistent orientation.
    Uses hybrid search and multiple orientation passes.
    """
    if verbose:
        print("  " + "-" * 50)
        print("  IMPROVED NORMAL ESTIMATION")
        print("  " + "-" * 50)
    
    # Auto-calculate search radius
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    diagonal = np.linalg.norm(extent)
    
    # Use smaller radius for better detail preservation
    search_radius = diagonal * 0.005
    max_nn = 50
    
    if verbose:
        print(f"    Search radius: {search_radius:.6f}")
        print(f"    Max neighbors: {max_nn}")
        print(f"    Estimating normals...")
    
    start = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius,
            max_nn=max_nn
        )
    )
    
    if verbose:
        print(f"    Normal estimation: {time.time()-start:.2f}s")
    
    # Orient normals consistently using multiple methods
    if verbose:
        print(f"    Orienting normals (tangent plane method, k=30)...")
    
    start = time.time()
    try:
        pcd.orient_normals_consistent_tangent_plane(k=30)
    except Exception:
        # Fallback to simpler orientation
        if verbose:
            print(f"    Tangent plane failed, trying camera orientation...")
        # Orient towards camera at origin (useful for scanned objects)
        pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
    
    if verbose:
        print(f"    Normal orientation: {time.time()-start:.2f}s")
        print()
    
    return pcd


def ball_pivoting_reconstruction(pcd, verbose=True):
    """
    Ball Pivoting Algorithm - better for preserving sharp edges.
    
    Unlike Poisson, BPA doesn't smooth or fill gaps, making it better
    for architectural scenes where you want crisp edges.
    """
    if verbose:
        print("  " + "-" * 50)
        print("  BALL PIVOTING RECONSTRUCTION")
        print("  " + "-" * 50)
    
    # Estimate ball radii based on point cloud density
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
    # Use multiple ball sizes for better coverage
    radii = [avg_dist * 1.5, avg_dist * 2.0, avg_dist * 3.0, avg_dist * 4.0]
    
    if verbose:
        print(f"    Average point distance: {avg_dist:.6f}")
        print(f"    Ball radii: {[f'{r:.6f}' for r in radii]}")
        print(f"    Running reconstruction...")
    
    start = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    if verbose:
        print(f"    Reconstruction time: {time.time()-start:.2f}s")
        print(f"    Generated vertices:  {len(mesh.vertices):,}")
        print(f"    Generated triangles: {len(mesh.triangles):,}")
        print()
    
    return mesh


def poisson_reconstruction_improved(pcd, depth=10, verbose=True):
    """
    Improved Poisson reconstruction with better parameters.
    
    Higher depth = more detail (but more polygons)
    """
    if verbose:
        print("  " + "-" * 50)
        print("  SCREENED POISSON RECONSTRUCTION")
        print("  " + "-" * 50)
        print(f"    Octree depth: {depth}")
        print(f"    Input points: {len(pcd.points):,}")
        print(f"    Running reconstruction...")
    
    start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=True  # Better for preserving features
    )
    
    if verbose:
        print(f"    Reconstruction time: {time.time()-start:.2f}s")
        print(f"    Generated vertices:  {len(mesh.vertices):,}")
        print(f"    Generated triangles: {len(mesh.triangles):,}")
        print()
    
    return mesh, np.asarray(densities)


def clean_mesh_standard(mesh, densities=None, density_percentile=0.01, verbose=True):
    """
    Standard mesh cleaning - conservative, preserves more geometry.
    """
    if verbose:
        print("  " + "-" * 50)
        print("  MESH CLEANING (Standard)")
        print("  " + "-" * 50)
        print(f"    Initial vertices:  {len(mesh.vertices):,}")
        print(f"    Initial triangles: {len(mesh.triangles):,}")
    
    # Remove only very low-density vertices (bottom 1%)
    if densities is not None and density_percentile > 0:
        threshold = np.quantile(densities, density_percentile)
        mask = densities < threshold
        
        if verbose:
            print(f"    Removing {np.sum(mask):,} low-density vertices (bottom {density_percentile*100:.0f}%)...")
        
        mesh.remove_vertices_by_mask(mask)
    
    # Standard cleanup only - no component removal
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    if verbose:
        print(f"    Final vertices:  {len(mesh.vertices):,}")
        print(f"    Final triangles: {len(mesh.triangles):,}")
        print()
    
    return mesh


def clean_mesh_aggressive(mesh, densities=None, density_percentile=0.05, verbose=True):
    """
    Aggressive mesh cleaning to remove Poisson artifacts.
    WARNING: Can remove valid geometry in large interior spaces!
    """
    if verbose:
        print("  " + "-" * 50)
        print("  MESH CLEANING (Aggressive)")
        print("  " + "-" * 50)
        print(f"    Initial vertices:  {len(mesh.vertices):,}")
        print(f"    Initial triangles: {len(mesh.triangles):,}")
    
    # Remove low-density vertices (Poisson artifacts)
    if densities is not None and density_percentile > 0:
        threshold = np.quantile(densities, density_percentile)
        mask = densities < threshold
        
        if verbose:
            print(f"    Removing {np.sum(mask):,} low-density vertices (bottom {density_percentile*100:.0f}%)...")
        
        mesh.remove_vertices_by_mask(mask)
    
    # Remove small disconnected components
    if verbose:
        print(f"    Removing small disconnected components...")
    
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        # Keep only large components (>5% of largest)
        max_cluster = cluster_n_triangles.max()
        min_size = max(100, int(max_cluster * 0.05))
        
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_size
        mesh.remove_triangles_by_mask(triangles_to_remove)
        
        if verbose:
            kept = np.sum(cluster_n_triangles >= min_size)
            print(f"      Kept {kept} components (min size: {min_size} triangles)")
    
    # Standard cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    if verbose:
        print(f"    Final vertices:  {len(mesh.vertices):,}")
        print(f"    Final triangles: {len(mesh.triangles):,}")
        print()
    
    return mesh


def transfer_colors_to_mesh(mesh, pcd, verbose=True):
    """Transfer colors from point cloud to mesh."""
    if not pcd.has_colors():
        return mesh
    
    if verbose:
        print("  " + "-" * 50)
        print("  COLOR TRANSFER")
        print("  " + "-" * 50)
    
    start = time.time()
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    
    # Use k-nearest neighbor averaging for smoother colors
    mesh_colors = np.zeros((len(mesh_vertices), 3))
    k = 3  # Average from 3 nearest neighbors
    
    for i, vertex in enumerate(mesh_vertices):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, k)
        mesh_colors[i] = np.mean(pcd_colors[idx], axis=0)
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    if verbose:
        print(f"    Color transfer complete ({time.time()-start:.2f}s)")
        print()
    
    return mesh


def hybrid_reconstruction(pcd, use_planes=True, plane_min_ratio=0.03, 
                         poisson_depth=10, verbose=True, aggressive_clean=False):
    """
    Hybrid reconstruction: detect planes for sharp surfaces, use Poisson for the rest.
    
    This gives you the best of both worlds:
    - Sharp, flat walls/floors/ceilings from plane detection
    - Smooth organic surfaces from Poisson
    """
    if verbose:
        print()
        print("  " + "=" * 50)
        print("  HYBRID RECONSTRUCTION")
        print("  " + "=" * 50)
        print()
    
    meshes = []
    pcd_original = pcd
    
    # Step 1: Detect and mesh planar regions
    if use_planes:
        planes, remaining_pcd = detect_planes_ransac(
            pcd, 
            min_points_ratio=plane_min_ratio,
            verbose=verbose
        )
        
        for i, plane in enumerate(planes):
            plane_mesh = create_plane_mesh(
                plane['points'], 
                plane['model'],
                verbose=False
            )
            if plane_mesh is not None and len(plane_mesh.triangles) > 0:
                meshes.append(plane_mesh)
                if verbose:
                    log_step(f"Created plane mesh {i+1}: {len(plane_mesh.triangles):,} triangles")
    else:
        remaining_pcd = pcd
    
    # Step 2: Use Poisson for remaining points
    if len(remaining_pcd.points) > 100:
        # Estimate normals for remaining points
        remaining_pcd = estimate_normals_improved(remaining_pcd, verbose=verbose)
        
        # Poisson reconstruction
        poisson_mesh, densities = poisson_reconstruction_improved(
            remaining_pcd, 
            depth=poisson_depth, 
            verbose=verbose
        )
        
        # Clean the Poisson mesh (use standard or aggressive based on setting)
        if aggressive_clean:
            poisson_mesh = clean_mesh_aggressive(
                poisson_mesh, 
                densities, 
                density_percentile=0.05,
                verbose=verbose
            )
        else:
            poisson_mesh = clean_mesh_standard(
                poisson_mesh, 
                densities, 
                density_percentile=0.01,
                verbose=verbose
            )
        
        if len(poisson_mesh.triangles) > 0:
            # Transfer colors
            poisson_mesh = transfer_colors_to_mesh(poisson_mesh, pcd_original, verbose=verbose)
            meshes.append(poisson_mesh)
    
    # Step 3: Combine all meshes
    if len(meshes) == 0:
        print("  [ERROR] No meshes generated!")
        return None
    
    if verbose:
        print("  " + "-" * 50)
        print("  COMBINING MESHES")
        print("  " + "-" * 50)
    
    combined = meshes[0]
    for mesh in meshes[1:]:
        combined += mesh
    
    # Final cleanup
    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    combined.compute_vertex_normals()
    
    if verbose:
        print(f"    Combined {len(meshes)} meshes")
        print(f"    Final vertices:  {len(combined.vertices):,}")
        print(f"    Final triangles: {len(combined.triangles):,}")
        print()
    
    return combined


def pointcloud_to_mesh_improved(
    input_path,
    output_path,
    method='poisson',  # 'hybrid', 'poisson', 'ball_pivoting'
    poisson_depth=11,
    use_planes=False,
    aggressive_outlier=False,
    verbose=True
):
    """
    Improved pipeline for better mesh quality.
    
    Args:
        input_path: Path to input point cloud PLY
        output_path: Path to output mesh
        method: 'poisson' (default), 'hybrid', or 'ball_pivoting'
        poisson_depth: Octree depth for Poisson (8-11, default: 11)
        use_planes: Enable RANSAC plane detection for sharp surfaces
        aggressive_outlier: Use multi-stage outlier removal (can remove valid geometry!)
        verbose: Print progress
    """
    total_start = time.time()
    
    print()
    print("  " + "=" * 50)
    print("  IMPROVED POINT CLOUD TO MESH PIPELINE")
    print("  " + "=" * 50)
    print(f"    Method: {method}")
    print(f"    Poisson depth: {poisson_depth}")
    print(f"    Plane detection: {use_planes}")
    print(f"    Aggressive outlier removal: {aggressive_outlier}")
    print()
    
    # Load point cloud
    pcd = load_point_cloud(input_path, verbose)
    
    if len(pcd.points) == 0:
        print("  [ERROR] Point cloud is empty")
        return None
    
    pcd_original = pcd
    
    # Outlier removal (only if aggressive mode enabled)
    if aggressive_outlier:
        pcd = aggressive_outlier_removal(pcd, verbose)
    
    if len(pcd.points) < 100:
        print("  [ERROR] Not enough points after filtering")
        return None
    
    # Choose reconstruction method
    if method == 'hybrid':
        mesh = hybrid_reconstruction(
            pcd, 
            use_planes=use_planes,
            poisson_depth=poisson_depth,
            verbose=verbose,
            aggressive_clean=aggressive_outlier
        )
    elif method == 'ball_pivoting':
        pcd = estimate_normals_improved(pcd, verbose)
        mesh = ball_pivoting_reconstruction(pcd, verbose)
        if mesh is not None and len(mesh.triangles) > 0:
            mesh = transfer_colors_to_mesh(mesh, pcd_original, verbose)
            mesh.compute_vertex_normals()
    else:  # poisson (default)
        pcd = estimate_normals_improved(pcd, verbose)
        mesh, densities = poisson_reconstruction_improved(pcd, poisson_depth, verbose)
        # Use standard or aggressive cleaning based on setting
        if aggressive_outlier:
            mesh = clean_mesh_aggressive(mesh, densities, verbose=verbose)
        else:
            mesh = clean_mesh_standard(mesh, densities, verbose=verbose)
        if len(mesh.triangles) > 0:
            mesh = transfer_colors_to_mesh(mesh, pcd_original, verbose)
    
    if mesh is None or len(mesh.triangles) == 0:
        print("  [ERROR] No mesh generated")
        return None
    
    # Save mesh
    print("  " + "-" * 50)
    print("  SAVING MESH")
    print("  " + "-" * 50)
    print(f"    Output: {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_vertex_colors=True,
        write_vertex_normals=True
    )
    
    total_time = time.time() - total_start
    
    if success:
        file_size = Path(output_path).stat().st_size
        size_str = f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f} KB"
        print(f"    File size: {size_str}")
        print()
        print("  " + "=" * 50)
        print("  COMPLETE")
        print("  " + "=" * 50)
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Final mesh:")
        print(f"      Vertices:  {len(mesh.vertices):,}")
        print(f"      Triangles: {len(mesh.triangles):,}")
        print()
    
    return output_path if success else None


def main():
    parser = argparse.ArgumentParser(
        description="Improved point cloud to mesh conversion"
    )
    parser.add_argument("input", help="Input point cloud PLY file")
    parser.add_argument("output", help="Output mesh file (OBJ, PLY, etc.)")
    parser.add_argument(
        "--method", "-m",
        choices=['hybrid', 'poisson', 'ball_pivoting'],
        default='hybrid',
        help="Reconstruction method (default: hybrid)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=10,
        help="Poisson octree depth (8-11, default: 10)"
    )
    parser.add_argument(
        "--no-planes",
        action="store_true",
        help="Disable RANSAC plane detection"
    )
    parser.add_argument(
        "--no-aggressive-outlier",
        action="store_true",
        help="Disable aggressive outlier removal"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    pointcloud_to_mesh_improved(
        args.input,
        args.output,
        method=args.method,
        poisson_depth=args.depth,
        use_planes=not args.no_planes,
        aggressive_outlier=not args.no_aggressive_outlier,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
