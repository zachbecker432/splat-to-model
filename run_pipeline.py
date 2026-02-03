"""
Splat to Mesh Pipeline Runner

Runs the complete pipeline from Gaussian Splat PLY to Unity-ready mesh.
Designed to run in Docker with configuration via environment variables.

Supports two modes:
1. MESH mode (default): Splat -> Point Cloud -> Mesh (for traditional Unity workflow)
2. ENHANCE mode: Splat -> Enhanced Splat (for direct splat rendering in Unity)

Environment Variables:
    MODE                - "mesh" or "enhance" (default: mesh)
    INPUT_FILE          - Input Gaussian Splat PLY file (required)
    OUTPUT_FILE         - Output mesh file (required)
    
    # Mesh mode settings:
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
    RECONSTRUCTION_METHOD - "bpa", "poisson", or "hybrid" (default: hybrid)
    POISSON_DEPTH       - Octree depth for Poisson (default: 8, higher=more detail)
    HOLE_FILL_DISTANCE  - Distance factor for hybrid hole filling (default: 3.0)
    KEEP_INTERMEDIATE   - Keep intermediate point cloud (default: false)
    
    # Enhance mode settings:
    REMOVE_FLOATERS     - Remove outlier Gaussians (default: true)
    DENSIFY_SPARSE      - Add Gaussians in sparse regions (default: true)
    SPLIT_LARGE         - Split large Gaussians (default: true)
    GRID_RESOLUTION     - Grid resolution for sparse detection (default: 50)
    DENSITY_THRESHOLD   - Percentile for sparse threshold (default: 10)
    
    # Floater removal parameters:
    FLOATER_STD_THRESHOLD   - Std dev threshold for position outliers (default: 3.0)
    FLOATER_MIN_OPACITY     - Minimum opacity threshold (default: 0.05)
    
    # Densification parameters:
    DENSIFY_K_NEIGHBORS     - Number of neighbors to interpolate from (default: 5)
    DENSIFY_JITTER          - Random offset factor for new Gaussians (default: 0.02)
    
    # Split parameters:
    SPLIT_THRESHOLD_PERCENTILE - Split Gaussians above this percentile (default: 95)
    SPLIT_MAX_NEW              - Maximum new Gaussians from splitting (default: 10000)
    
    # Plane detection & flattening (for walls/floors/ceilings):
    FLATTEN_PLANES          - Enable plane detection and flattening (default: false)
    PLANE_MAX               - Maximum planes to detect (default: 10)
    PLANE_DISTANCE          - Distance threshold for plane inliers (default: 0.05)
    PLANE_MIN_RATIO         - Minimum ratio of points for valid plane (default: 0.05)
    PLANE_FLATTEN_STRENGTH  - How strongly to flatten to planes (default: 0.8)
    
    # Depth consistency filter (removes reflection artifacts):
    DEPTH_FILTER            - Enable depth consistency filtering (default: false)
    DEPTH_NEIGHBORS         - Number of neighbors for depth analysis (default: 20)
    DEPTH_STD_THRESHOLD     - Std devs from surface to be outlier (default: 2.0)
    DEPTH_REMOVE_OUTLIERS   - Remove outliers instead of projecting (default: false)
    
    # Surface thickness compression:
    COMPRESS_THICKNESS      - Enable thickness compression (default: false)
    THICKNESS_NEIGHBORS     - Number of neighbors for analysis (default: 15)
    THICKNESS_MAX_FACTOR    - Max thickness factor before compression (default: 3.0)
    THICKNESS_STRENGTH      - Compression strength (default: 0.7)
    
    VERBOSE             - Print progress (default: true)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

from splat_to_pointcloud import splat_to_pointcloud
from pointcloud_to_mesh import pointcloud_to_mesh
from splat_enhance import enhance_splat


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
    reconstruction_method = os.environ.get('RECONSTRUCTION_METHOD', 'hybrid').lower()
    poisson_depth = get_env_int('POISSON_DEPTH', 8)
    hole_fill_distance = get_env_float('HOLE_FILL_DISTANCE', 3.0)
    keep_intermediate = get_env_bool('KEEP_INTERMEDIATE', False)
    verbose = get_env_bool('VERBOSE', True)
    
    # Validate reconstruction method
    if reconstruction_method not in ('bpa', 'poisson', 'hybrid'):
        print(f"[WARNING] Invalid RECONSTRUCTION_METHOD: {reconstruction_method}, using 'hybrid'")
        reconstruction_method = 'hybrid'
    
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
        method_label = reconstruction_method.upper()
        log_header(f"SPLAT TO MESH PIPELINE ({method_label})")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("  INPUT/OUTPUT:")
        print(f"    Input file:  {input_path}")
        print(f"    Input size:  {format_size(input_path)}")
        print(f"    Output file: {output_path}")
        print()
        print("  CONFIGURATION:")
        print(f"    OPACITY_THRESHOLD:     {opacity_threshold}")
        if max_scale is not None:
            print(f"    MAX_SCALE:             {max_scale}")
        print(f"    VOXEL_SIZE:            {voxel_size if voxel_size is not None else 'auto'}")
        if target_triangles is not None:
            print(f"    TARGET_TRIANGLES:      {target_triangles}")
        print(f"    OUTLIER_STD_RATIO:     {outlier_std_ratio}")
        print(f"    RECONSTRUCTION_METHOD: {reconstruction_method.upper()}")
        if reconstruction_method in ('poisson', 'hybrid'):
            print(f"    POISSON_DEPTH:         {poisson_depth}")
        if reconstruction_method == 'hybrid':
            print(f"    HOLE_FILL_DISTANCE:    {hole_fill_distance}")
        print(f"    THIN_GEOMETRY:         {thin_geometry}")
        print(f"    FLIP_NORMALS:          {flip_normals}")
        print(f"    DOUBLE_SIDED:          {double_sided}")
        print(f"    SMOOTH_FINAL:          {smooth_final}")
        if smooth_final:
            print(f"    SMOOTH_ITERATIONS:     {smooth_iterations}")
        print(f"    KEEP_INTERMEDIATE:     {keep_intermediate}")
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
    
    # Stage 2: Convert point cloud to mesh
    if verbose:
        log_header(f"STAGE 2: Mesh Generation ({reconstruction_method.upper()})")
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
        reconstruction_method=reconstruction_method,
        poisson_depth=poisson_depth,
        hole_fill_distance_factor=hole_fill_distance,
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


def run_enhance_pipeline():
    """Run the splat enhancement pipeline for direct splat rendering."""
    
    # Read configuration from environment
    input_file = os.environ.get('INPUT_FILE')
    output_file = os.environ.get('OUTPUT_FILE')
    
    if not input_file:
        print("[ERROR] INPUT_FILE environment variable is required")
        print("Example: INPUT_FILE=model.ply")
        return None
    
    if not output_file:
        print("[ERROR] OUTPUT_FILE environment variable is required")
        print("Example: OUTPUT_FILE=enhanced.ply")
        return None
    
    # Parse enhancement configuration
    remove_floaters = get_env_bool('REMOVE_FLOATERS', True)
    densify_sparse = get_env_bool('DENSIFY_SPARSE', True)
    split_large = get_env_bool('SPLIT_LARGE', True)
    grid_resolution = get_env_int('GRID_RESOLUTION', 50)
    density_threshold = get_env_int('DENSITY_THRESHOLD', 10)
    
    # Floater removal parameters
    floater_std_threshold = get_env_float('FLOATER_STD_THRESHOLD', 3.0)
    floater_min_opacity = get_env_float('FLOATER_MIN_OPACITY', 0.05)
    
    # Densification parameters
    densify_k_neighbors = get_env_int('DENSIFY_K_NEIGHBORS', 5)
    densify_jitter = get_env_float('DENSIFY_JITTER', 0.02)
    
    # Split parameters
    split_threshold_percentile = get_env_int('SPLIT_THRESHOLD_PERCENTILE', 95)
    split_max_new = get_env_int('SPLIT_MAX_NEW', 10000)
    
    # Plane detection & flattening parameters
    flatten_planes = get_env_bool('FLATTEN_PLANES', False)
    plane_max = get_env_int('PLANE_MAX', 10)
    plane_distance = get_env_float('PLANE_DISTANCE', 0.05)
    plane_min_ratio = get_env_float('PLANE_MIN_RATIO', 0.05)
    plane_flatten_strength = get_env_float('PLANE_FLATTEN_STRENGTH', 0.8)
    
    # Depth consistency filter parameters
    depth_filter = get_env_bool('DEPTH_FILTER', False)
    depth_neighbors = get_env_int('DEPTH_NEIGHBORS', 20)
    depth_std_threshold = get_env_float('DEPTH_STD_THRESHOLD', 2.0)
    depth_remove_outliers = get_env_bool('DEPTH_REMOVE_OUTLIERS', False)
    
    # Thickness compression parameters
    compress_thickness = get_env_bool('COMPRESS_THICKNESS', False)
    thickness_neighbors = get_env_int('THICKNESS_NEIGHBORS', 15)
    thickness_max_factor = get_env_float('THICKNESS_MAX_FACTOR', 3.0)
    thickness_strength = get_env_float('THICKNESS_STRENGTH', 0.7)
    
    verbose = get_env_bool('VERBOSE', True)
    
    # Setup paths
    input_path = Path('/data') / input_file
    output_path = Path('/data') / output_file
    
    pipeline_start = time.time()
    
    # Validate input
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return None
    
    if verbose:
        log_header("SPLAT ENHANCEMENT PIPELINE")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("  INPUT/OUTPUT:")
        print(f"    Input file:  {input_path}")
        print(f"    Input size:  {format_size(input_path)}")
        print(f"    Output file: {output_path}")
        print()
        print("  CONFIGURATION:")
        print(f"    REMOVE_FLOATERS:           {remove_floaters}")
        print(f"    DENSIFY_SPARSE:            {densify_sparse}")
        print(f"    SPLIT_LARGE:               {split_large}")
        print(f"    FLATTEN_PLANES:            {flatten_planes}")
        print(f"    DEPTH_FILTER:              {depth_filter}")
        print(f"    COMPRESS_THICKNESS:        {compress_thickness}")
        print(f"    GRID_RESOLUTION:           {grid_resolution}")
        print(f"    DENSITY_THRESHOLD:         {density_threshold}")
        if remove_floaters:
            print(f"    FLOATER_STD_THRESHOLD:     {floater_std_threshold}")
            print(f"    FLOATER_MIN_OPACITY:       {floater_min_opacity}")
        if densify_sparse:
            print(f"    DENSIFY_K_NEIGHBORS:       {densify_k_neighbors}")
            print(f"    DENSIFY_JITTER:            {densify_jitter}")
        if split_large:
            print(f"    SPLIT_THRESHOLD_PERCENTILE: {split_threshold_percentile}")
            print(f"    SPLIT_MAX_NEW:             {split_max_new}")
        if flatten_planes:
            print(f"    PLANE_MAX:                 {plane_max}")
            print(f"    PLANE_DISTANCE:            {plane_distance}")
            print(f"    PLANE_MIN_RATIO:           {plane_min_ratio}")
            print(f"    PLANE_FLATTEN_STRENGTH:    {plane_flatten_strength}")
        if depth_filter:
            print(f"    DEPTH_NEIGHBORS:           {depth_neighbors}")
            print(f"    DEPTH_STD_THRESHOLD:       {depth_std_threshold}")
            print(f"    DEPTH_REMOVE_OUTLIERS:     {depth_remove_outliers}")
        if compress_thickness:
            print(f"    THICKNESS_NEIGHBORS:       {thickness_neighbors}")
            print(f"    THICKNESS_MAX_FACTOR:      {thickness_max_factor}")
            print(f"    THICKNESS_STRENGTH:        {thickness_strength}")
        print("=" * 70)
        print()
    
    # Run enhancement
    try:
        stats = enhance_splat(
            str(input_path),
            str(output_path),
            remove_floaters_enabled=remove_floaters,
            densify_sparse_enabled=densify_sparse,
            split_large_enabled=split_large,
            plane_detection_enabled=flatten_planes,
            depth_consistency_enabled=depth_filter,
            thickness_compression_enabled=compress_thickness,
            grid_resolution=grid_resolution,
            density_threshold_percentile=density_threshold,
            floater_std_threshold=floater_std_threshold,
            floater_min_opacity=floater_min_opacity,
            densify_k_neighbors=densify_k_neighbors,
            densify_jitter=densify_jitter,
            split_threshold_percentile=split_threshold_percentile,
            split_max_new=split_max_new,
            plane_max_planes=plane_max,
            plane_distance_threshold=plane_distance,
            plane_min_inliers_ratio=plane_min_ratio,
            plane_flatten_strength=plane_flatten_strength,
            depth_k_neighbors=depth_neighbors,
            depth_std_threshold=depth_std_threshold,
            depth_flatten_outliers=not depth_remove_outliers,
            thickness_k_neighbors=thickness_neighbors,
            thickness_max_factor=thickness_max_factor,
            thickness_compression_strength=thickness_strength,
            verbose=verbose
        )
        
        pipeline_time = time.time() - pipeline_start
        
        if verbose:
            log_header("ENHANCEMENT PIPELINE SUMMARY")
            print()
            print(f"  STATUS:     SUCCESS")
            print(f"  OUTPUT:     {output_path}")
            if output_path.exists():
                print(f"  FILE SIZE:  {format_size(output_path)}")
            print()
            print(f"  CHANGES:")
            print(f"    Floaters removed:      {stats.get('floaters_removed', 0):,}")
            if flatten_planes:
                print(f"    Planes detected:       {stats.get('planes_detected', 0)}")
                print(f"    Flattened to planes:   {stats.get('plane_flattened', 0):,}")
            if depth_filter:
                print(f"    Depth outliers fixed:  {stats.get('depth_outliers', 0):,}")
            if compress_thickness:
                print(f"    Thickness compressed:  {stats.get('thickness_compressed', 0):,}")
            print(f"    Sparse regions filled: {stats.get('sparse_added', 0):,}")
            print(f"    Large Gaussians split: {stats.get('split_added', 0):,}")
            print()
            print(f"  TIMING: {format_time(pipeline_time)}")
            print()
            print("=" * 70)
        
        return stats
        
    except Exception as e:
        print(f"[ERROR] Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    mode = os.environ.get('MODE', 'mesh').lower()
    
    if mode == 'enhance':
        result = run_enhance_pipeline()
    else:
        result = run_pipeline()
    
    sys.exit(0 if result else 1)
