"""
Splat to Mesh Pipeline Runner

Runs the complete pipeline from Gaussian Splat PLY to Unity-ready mesh.
Combines splat_to_pointcloud.py and pointcloud_to_mesh.py into a single command.

Usage:
    python run_pipeline.py input_splat.ply output_mesh.obj [options]
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Import the pipeline modules
from splat_to_pointcloud import splat_to_pointcloud
from pointcloud_to_mesh import pointcloud_to_mesh


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_size(path):
    """Format file size in human-readable format."""
    size = Path(path).stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def log_header(title):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_pipeline(
    input_splat,
    output_mesh,
    opacity_threshold=0.3,
    max_scale=None,
    poisson_depth=9,
    density_threshold=0.01,
    voxel_size=None,
    target_triangles=None,
    outlier_std_ratio=2.0,
    reconstruction_method='poisson',
    linear_fit=True,
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
    keep_intermediate=False,
    intermediate_dir=None,
    verbose=True
):
    """
    Run the complete splat-to-mesh pipeline.
    
    Args:
        input_splat: Path to input Gaussian Splat PLY file
        output_mesh: Path to output mesh file (OBJ recommended)
        opacity_threshold: Minimum opacity for point extraction (0-1)
        poisson_depth: Octree depth for Poisson reconstruction (6-11)
        density_threshold: Percentile of low-density vertices to remove
        voxel_size: Voxel size for point cloud downsampling
        target_triangles: Target triangle count for mesh simplification
        outlier_std_ratio: Std ratio for outlier removal (lower = more aggressive)
        reconstruction_method: 'poisson', 'bpa' (ball pivoting), or 'alpha' (alpha shapes)
        linear_fit: Use linear fit for Poisson (better adherence to points)
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
        keep_intermediate: Keep intermediate point cloud file
        intermediate_dir: Directory for intermediate files (default: same as output)
        verbose: Print progress information
    
    Returns:
        Path to output mesh if successful, None otherwise
    """
    pipeline_start = time.time()
    input_path = Path(input_splat)
    output_path = Path(output_mesh)
    
    # Validate input
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_splat}")
        return None
    
    # Determine intermediate file location
    if intermediate_dir:
        intermediate_path = Path(intermediate_dir)
    else:
        intermediate_path = output_path.parent
    
    intermediate_path.mkdir(parents=True, exist_ok=True)
    
    # Generate intermediate point cloud filename
    pointcloud_path = intermediate_path / f"{input_path.stem}_pointcloud.ply"
    
    if verbose:
        log_header("SPLAT TO MESH PIPELINE")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("  INPUT/OUTPUT:")
        print(f"    Input file:  {input_splat}")
        print(f"    Input size:  {format_size(input_path)}")
        print(f"    Output file: {output_mesh}")
        print()
        print("  PARAMETERS:")
        print(f"    Opacity threshold:    {opacity_threshold}")
        print(f"    Reconstruction:       {reconstruction_method}")
        if reconstruction_method == 'poisson':
            print(f"    Poisson depth:        {poisson_depth}")
            print(f"    Linear fit:           {linear_fit}")
        elif reconstruction_method == 'alpha':
            print(f"    Alpha value:          {alpha_value if alpha_value else 'auto'}")
        print(f"    Density threshold:    {density_threshold}")
        print(f"    Voxel size:           {voxel_size if voxel_size else 'auto'}")
        print(f"    Target triangles:     {target_triangles if target_triangles else 'none'}")
        print(f"    Outlier std ratio:    {outlier_std_ratio}")
        if smooth_final:
            print(f"    Final smoothing:      {smooth_final_iterations} iterations")
        if segmented:
            print()
            print("  SEGMENTED RECONSTRUCTION:")
            print(f"    Enabled:              True")
            print(f"    Plane distance:       {plane_distance}")
            print(f"    Plane min ratio:      {plane_min_ratio}")
            print(f"    Max planes:           {max_planes}")
            print(f"    Smooth organic:       {smooth_organic}")
            print(f"    Smooth iterations:    {smooth_iterations}")
            print(f"    Use alpha shapes:     {use_alpha_shapes}")
        if flip_normals:
            print()
            print(f"    Flip normals:         {flip_normals}")
        print()
        print(f"    Keep intermediate:    {keep_intermediate}")
        print("=" * 70)
        print()
    
    # Stage 1: Extract point cloud from splat
    if verbose:
        log_header("STAGE 1: Point Cloud Extraction")
        print()
    
    stage1_start = time.time()
    num_points = splat_to_pointcloud(
        str(input_path),
        str(pointcloud_path),
        opacity_threshold=opacity_threshold,
        max_scale=max_scale,
        verbose=verbose
    )
    stage1_time = time.time() - stage1_start
    
    if num_points == 0:
        print("[ERROR] No points extracted from splat file")
        print("[HINT] Try lowering --opacity threshold (e.g., --opacity 0.1)")
        return None
    
    if verbose:
        print()
        print(f"  [STAGE 1 COMPLETE] Extracted {num_points:,} points in {format_time(stage1_time)}")
        if pointcloud_path.exists():
            print(f"  [STAGE 1 OUTPUT] Point cloud size: {format_size(pointcloud_path)}")
    
    # Stage 2: Convert point cloud to mesh
    if verbose:
        log_header("STAGE 2: Mesh Generation")
        print()
    
    stage2_start = time.time()
    result = pointcloud_to_mesh(
        str(pointcloud_path),
        str(output_path),
        poisson_depth=poisson_depth,
        density_threshold=density_threshold,
        voxel_size=voxel_size,
        target_triangles=target_triangles,
        outlier_std_ratio=outlier_std_ratio,
        reconstruction_method=reconstruction_method,
        linear_fit=linear_fit,
        alpha_value=alpha_value,
        segmented=segmented,
        plane_distance=plane_distance,
        plane_min_ratio=plane_min_ratio,
        max_planes=max_planes,
        smooth_organic=smooth_organic,
        smooth_iterations=smooth_iterations,
        use_alpha_shapes=use_alpha_shapes,
        flip_normals=flip_normals,
        double_sided=double_sided,
        smooth_final=smooth_final,
        smooth_final_iterations=smooth_final_iterations,
        verbose=verbose
    )
    stage2_time = time.time() - stage2_start
    
    if verbose and result:
        print()
        print(f"  [STAGE 2 COMPLETE] Generated mesh in {format_time(stage2_time)}")
        if output_path.exists():
            print(f"  [STAGE 2 OUTPUT] Mesh size: {format_size(output_path)}")
    
    # Clean up intermediate file unless requested to keep
    if not keep_intermediate and pointcloud_path.exists():
        pointcloud_path.unlink()
        if verbose:
            print(f"\n  [CLEANUP] Removed intermediate file: {pointcloud_path}")
    elif keep_intermediate and verbose:
        print(f"\n  [KEPT] Intermediate file: {pointcloud_path}")
    
    pipeline_time = time.time() - pipeline_start
    
    if verbose:
        log_header("PIPELINE SUMMARY")
        if result:
            print()
            print(f"  STATUS:     SUCCESS")
            print(f"  OUTPUT:     {output_mesh}")
            if output_path.exists():
                print(f"  FILE SIZE:  {format_size(output_path)}")
            print()
            print(f"  TIMING:")
            print(f"    Stage 1 (extraction): {format_time(stage1_time)}")
            print(f"    Stage 2 (meshing):    {format_time(stage2_time)}")
            print(f"    Total:                {format_time(pipeline_time)}")
            print()
            print("  NEXT STEPS:")
            print("    1. Import the .obj file into Unity")
            print("    2. Create a material with vertex color shader")
            print("    3. Apply material to mesh")
        else:
            print()
            print(f"  STATUS: FAILED")
            print()
            print("  TROUBLESHOOTING:")
            print("    - Check the error messages above")
            print("    - Try --opacity 0.1 to extract more points")
            print("    - Use --keep-intermediate to inspect point cloud")
        print()
        print("=" * 70)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gaussian Splat PLY to Unity-ready mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (defaults work well for most cases)
  python run_pipeline.py model.ply mesh.obj

  # Higher quality mesh
  python run_pipeline.py model.ply mesh.obj --depth 10 --opacity 0.2

  # Ball Pivoting Algorithm (better for sharp edges, distinct surfaces)
  python run_pipeline.py model.ply mesh.obj --method bpa --voxel-size 0

  # Alpha shapes (good for concave objects)
  python run_pipeline.py model.ply mesh.obj --method alpha

  # Mobile-optimized (fewer triangles)
  python run_pipeline.py model.ply mesh.obj --simplify 50000

  # Segmented reconstruction (better for architectural scenes with walls/floors)
  python run_pipeline.py model.ply mesh.obj --segmented

  # Segmented with custom plane detection
  python run_pipeline.py model.ply mesh.obj --segmented --plane-distance 0.03 --max-planes 15

  # Keep intermediate point cloud for inspection
  python run_pipeline.py model.ply mesh.obj --keep-intermediate

  # With final mesh smoothing
  python run_pipeline.py model.ply mesh.obj --smooth

Quality Presets:
  Low (fast):    --depth 7 --opacity 0.5 --simplify 20000
  Medium:        --depth 8 --opacity 0.3 --simplify 50000
  High:          --depth 9 --opacity 0.2 --voxel-size 0
  Ultra:         --depth 11 --opacity 0.1 --voxel-size 0
  Sharp edges:   --method bpa --opacity 0.2 --voxel-size 0
  Architectural: --depth 10 --opacity 0.2 --segmented --max-planes 15
        """
    )
    
    parser.add_argument("input", help="Input Gaussian Splat PLY file (from Postshot)")
    parser.add_argument("output", help="Output mesh file (OBJ recommended for Unity)")
    
    # Point extraction options
    extraction_group = parser.add_argument_group("Point Extraction Options")
    extraction_group.add_argument(
        "--opacity", "-o",
        type=float,
        default=0.3,
        help="Minimum opacity threshold for points (0-1, default: 0.3)"
    )
    extraction_group.add_argument(
        "--max-scale",
        type=float,
        default=None,
        help="Maximum Gaussian scale to include (filters large/blobby Gaussians, e.g., 0.05)"
    )
    
    # Mesh generation options
    mesh_group = parser.add_argument_group("Mesh Generation Options")
    mesh_group.add_argument(
        "--method", "-m",
        type=str,
        choices=['poisson', 'bpa', 'alpha', 'hybrid'],
        default='poisson',
        help="Reconstruction method: poisson (watertight), bpa (sharp edges), alpha (concave), hybrid (bpa+poisson fills holes) (default: poisson)"
    )
    mesh_group.add_argument(
        "--depth", "-d",
        type=int,
        default=9,
        help="Poisson octree depth (6-11, higher=more detail, default: 9)"
    )
    mesh_group.add_argument(
        "--density-threshold", "-t",
        type=float,
        default=0.01,
        help="Percentile of low-density vertices to remove (0-1, default: 0.01)"
    )
    mesh_group.add_argument(
        "--voxel-size", "-v",
        type=float,
        default=None,
        help="Voxel size for point cloud downsampling (default: auto, 0=disabled)"
    )
    mesh_group.add_argument(
        "--simplify", "-s",
        type=int,
        default=None,
        help="Target number of triangles (default: no simplification)"
    )
    mesh_group.add_argument(
        "--outlier-std", "-r",
        type=float,
        default=2.0,
        help="Outlier removal std ratio - lower=more aggressive (default: 2.0)"
    )
    mesh_group.add_argument(
        "--no-linear-fit",
        action="store_true",
        help="Disable linear fit for Poisson (default: enabled)"
    )
    mesh_group.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha value for alpha shapes method (default: auto)"
    )
    
    # Segmented reconstruction options
    seg_group = parser.add_argument_group("Segmented Reconstruction (for architectural scenes)")
    seg_group.add_argument(
        "--segmented", "-S",
        action="store_true",
        help="Enable segmented reconstruction (planes processed separately)"
    )
    seg_group.add_argument(
        "--plane-distance",
        type=float,
        default=0.02,
        help="RANSAC distance threshold for plane detection (default: 0.02)"
    )
    seg_group.add_argument(
        "--plane-min-ratio",
        type=float,
        default=0.02,
        help="Minimum ratio of points for valid plane (default: 0.02)"
    )
    seg_group.add_argument(
        "--max-planes",
        type=int,
        default=10,
        help="Maximum number of planes to detect (default: 10)"
    )
    seg_group.add_argument(
        "--no-smooth-organic",
        action="store_true",
        help="Disable Taubin smoothing on organic regions"
    )
    seg_group.add_argument(
        "--smooth-iterations",
        type=int,
        default=5,
        help="Number of Taubin smoothing iterations (default: 5)"
    )
    seg_group.add_argument(
        "--use-alpha-shapes",
        action="store_true",
        help="Use alpha shapes instead of Delaunay for plane meshing"
    )
    
    # Mesh fixing options
    fix_group = parser.add_argument_group("Mesh Fixing Options")
    fix_group.add_argument(
        "--flip-normals",
        action="store_true",
        help="Flip all mesh normals (use if mesh appears inside-out in Unity)"
    )
    fix_group.add_argument(
        "--double-sided",
        action="store_true",
        help="Make mesh double-sided (fixes see-through faces, doubles triangle count)"
    )
    fix_group.add_argument(
        "--smooth",
        action="store_true",
        help="Apply Taubin smoothing to final mesh"
    )
    fix_group.add_argument(
        "--smooth-final-iterations",
        type=int,
        default=5,
        help="Number of final smoothing iterations (default: 5)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--keep-intermediate", "-k",
        action="store_true",
        help="Keep intermediate point cloud file"
    )
    output_group.add_argument(
        "--intermediate-dir", "-i",
        type=str,
        default=None,
        help="Directory for intermediate files (default: same as output)"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    result = run_pipeline(
        args.input,
        args.output,
        opacity_threshold=args.opacity,
        max_scale=args.max_scale,
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
        keep_intermediate=args.keep_intermediate,
        intermediate_dir=args.intermediate_dir,
        verbose=not args.quiet
    )
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
