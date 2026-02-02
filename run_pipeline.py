"""
Splat to Mesh Pipeline Runner

Runs the complete pipeline from Gaussian Splat PLY to Unity-ready mesh.
Designed to run in Docker with configuration via environment variables.

Environment Variables:
    INPUT_FILE          - Input Gaussian Splat PLY file (required)
    OUTPUT_FILE         - Output mesh file (required)
    OPACITY_THRESHOLD   - Minimum opacity for points (default: 0.3)
    MAX_SCALE           - Maximum Gaussian scale to include (optional)
    VOXEL_SIZE          - Voxel size for downsampling (default: auto, 0=disabled)
    TARGET_TRIANGLES    - Target triangle count for simplification (optional)
    OUTLIER_STD_RATIO   - Outlier removal std ratio (default: 2.0)
    THIN_GEOMETRY       - Enable thin geometry mode (default: true)
    FLIP_NORMALS        - Flip mesh normals (default: false)
    DOUBLE_SIDED        - Make mesh double-sided (default: false)
    SMOOTH_FINAL        - Apply smoothing (default: false)
    SMOOTH_ITERATIONS   - Number of smoothing iterations (default: 5)
    KEEP_INTERMEDIATE   - Keep intermediate point cloud (default: false)
    VERBOSE             - Print progress (default: true)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

from splat_to_pointcloud import splat_to_pointcloud
from pointcloud_to_mesh import pointcloud_to_mesh


def get_env_bool(name, default=False):
    """Get boolean from environment variable."""
    val = os.environ.get(name, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def get_env_float(name, default=None):
    """Get float from environment variable."""
    val = os.environ.get(name)
    if val is None or val == '':
        return default
    try:
        return float(val)
    except ValueError:
        print(f"[WARNING] Invalid float for {name}: {val}, using default: {default}")
        return default


def get_env_int(name, default=None):
    """Get int from environment variable."""
    val = os.environ.get(name)
    if val is None or val == '':
        return default
    try:
        return int(val)
    except ValueError:
        print(f"[WARNING] Invalid int for {name}: {val}, using default: {default}")
        return default


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


def run_pipeline():
    """Run the complete splat-to-mesh pipeline using environment variables."""
    
    # Read configuration from environment
    input_file = os.environ.get('INPUT_FILE')
    output_file = os.environ.get('OUTPUT_FILE')
    
    if not input_file:
        print("[ERROR] INPUT_FILE environment variable is required")
        print("Example: INPUT_FILE=model.ply")
        return None
    
    if not output_file:
        print("[ERROR] OUTPUT_FILE environment variable is required")
        print("Example: OUTPUT_FILE=mesh.obj")
        return None
    
    # Parse all configuration
    opacity_threshold = get_env_float('OPACITY_THRESHOLD', 0.3)
    max_scale = get_env_float('MAX_SCALE', None)
    voxel_size = get_env_float('VOXEL_SIZE', None)
    target_triangles = get_env_int('TARGET_TRIANGLES', None)
    outlier_std_ratio = get_env_float('OUTLIER_STD_RATIO', 2.0)
    thin_geometry = get_env_bool('THIN_GEOMETRY', True)
    flip_normals = get_env_bool('FLIP_NORMALS', False)
    double_sided = get_env_bool('DOUBLE_SIDED', False)
    smooth_final = get_env_bool('SMOOTH_FINAL', False)
    smooth_iterations = get_env_int('SMOOTH_ITERATIONS', 5)
    keep_intermediate = get_env_bool('KEEP_INTERMEDIATE', False)
    verbose = get_env_bool('VERBOSE', True)
    
    # Setup paths
    input_path = Path('/data') / input_file
    output_path = Path('/data') / output_file
    
    pipeline_start = time.time()
    
    # Validate input
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return None
    
    # Generate intermediate point cloud filename
    pointcloud_path = output_path.parent / f"{input_path.stem}_pointcloud.ply"
    
    if verbose:
        log_header("SPLAT TO MESH PIPELINE (BPA)")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("  INPUT/OUTPUT:")
        print(f"    Input file:  {input_path}")
        print(f"    Input size:  {format_size(input_path)}")
        print(f"    Output file: {output_path}")
        print()
        print("  CONFIGURATION:")
        print(f"    OPACITY_THRESHOLD:  {opacity_threshold}")
        if max_scale is not None:
            print(f"    MAX_SCALE:          {max_scale}")
        print(f"    VOXEL_SIZE:         {voxel_size if voxel_size is not None else 'auto'}")
        if target_triangles is not None:
            print(f"    TARGET_TRIANGLES:   {target_triangles}")
        print(f"    OUTLIER_STD_RATIO:  {outlier_std_ratio}")
        print(f"    THIN_GEOMETRY:      {thin_geometry}")
        print(f"    FLIP_NORMALS:       {flip_normals}")
        print(f"    DOUBLE_SIDED:       {double_sided}")
        print(f"    SMOOTH_FINAL:       {smooth_final}")
        if smooth_final:
            print(f"    SMOOTH_ITERATIONS:  {smooth_iterations}")
        print(f"    KEEP_INTERMEDIATE:  {keep_intermediate}")
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
        print("[HINT] Try lowering OPACITY_THRESHOLD (e.g., OPACITY_THRESHOLD=0.1)")
        return None
    
    if verbose:
        print()
        print(f"  [STAGE 1 COMPLETE] Extracted {num_points:,} points in {format_time(stage1_time)}")
        if pointcloud_path.exists():
            print(f"  [STAGE 1 OUTPUT] Point cloud size: {format_size(pointcloud_path)}")
    
    # Stage 2: Convert point cloud to mesh using BPA
    if verbose:
        log_header("STAGE 2: Mesh Generation (BPA)")
        print()
    
    stage2_start = time.time()
    result = pointcloud_to_mesh(
        str(pointcloud_path),
        str(output_path),
        voxel_size=voxel_size,
        target_triangles=target_triangles,
        outlier_std_ratio=outlier_std_ratio,
        thin_geometry=thin_geometry,
        flip_normals=flip_normals,
        double_sided=double_sided,
        smooth_final=smooth_final,
        smooth_iterations=smooth_iterations,
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
            print(f"  OUTPUT:     {output_path}")
            if output_path.exists():
                print(f"  FILE SIZE:  {format_size(output_path)}")
            print()
            print(f"  TIMING:")
            print(f"    Stage 1 (extraction): {format_time(stage1_time)}")
            print(f"    Stage 2 (meshing):    {format_time(stage2_time)}")
            print(f"    Total:                {format_time(pipeline_time)}")
        else:
            print()
            print(f"  STATUS: FAILED")
            print()
            print("  TROUBLESHOOTING:")
            print("    - Check the error messages above")
            print("    - Try OPACITY_THRESHOLD=0.1 to extract more points")
            print("    - Use KEEP_INTERMEDIATE=true to inspect point cloud")
        print()
        print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = run_pipeline()
    sys.exit(0 if result else 1)
