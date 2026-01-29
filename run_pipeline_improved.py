"""
IMPROVED Splat to Mesh Pipeline Runner

Enhanced pipeline with better quality output:
- Multi-stage filtering to remove bad data
- RANSAC plane detection for sharp walls/floors
- Hybrid meshing (planes + Poisson)
- Better outlier removal

Usage:
    python run_pipeline_improved.py input_splat.ply output_mesh.obj [options]

Quality Presets:
    --preset fast      : Quick processing, lower quality
    --preset balanced  : Good balance of speed and quality (default)
    --preset quality   : Higher quality, slower
    --preset ultra     : Maximum quality, slowest
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Import the improved pipeline modules
from splat_to_pointcloud_improved import splat_to_pointcloud_improved
from pointcloud_to_mesh_improved import pointcloud_to_mesh_improved


# Quality presets
# NOTE: Filtering is now much more conservative to avoid removing valid geometry
PRESETS = {
    'fast': {
        'opacity_threshold': 0.5,
        'use_scale_filter': False,  # Disabled - was too aggressive
        'use_density_filter': False,
        'use_color_filter': False,
        'method': 'poisson',
        'poisson_depth': 9,
        'use_planes': False,
        'aggressive_outlier': False,
        'voxel_size': None,  # No downsampling
    },
    'balanced': {
        'opacity_threshold': 0.4,
        'use_scale_filter': False,  # Disabled for safety
        'use_density_filter': False,
        'use_color_filter': False,
        'method': 'poisson',  # Stick with proven Poisson
        'poisson_depth': 10,
        'use_planes': False,  # Plane detection can help but needs tuning
        'aggressive_outlier': False,  # Use standard outlier removal
        'voxel_size': None,
    },
    'quality': {
        'opacity_threshold': 0.4,
        'use_scale_filter': False,
        'use_density_filter': False,
        'use_color_filter': False,
        'method': 'poisson',
        'poisson_depth': 11,
        'use_planes': False,
        'aggressive_outlier': False,
        'voxel_size': None,
    },
    'ultra': {
        'opacity_threshold': 0.4,
        'use_scale_filter': False,
        'use_density_filter': False,
        'use_color_filter': False,
        'method': 'poisson',
        'poisson_depth': 11,
        'use_planes': False,
        'aggressive_outlier': False,
        'voxel_size': None,
    },
    # INTERIOR preset - specifically tuned for building interiors
    # Based on your best settings: voxel_size=None, depth=11, opacity=0.4
    'interior': {
        'opacity_threshold': 0.4,
        'use_scale_filter': False,
        'use_density_filter': False,
        'use_color_filter': False,
        'method': 'poisson',
        'poisson_depth': 11,
        'use_planes': False,  # Can enable with --use-planes if walls need sharpening
        'aggressive_outlier': False,
        'voxel_size': None,
    },
    # EXPERIMENTAL preset - tries the new features (use with caution)
    'experimental': {
        'opacity_threshold': 0.4,
        'use_scale_filter': True,
        'scale_percentile_low': 1,  # Very conservative
        'scale_percentile_high': 99,
        'use_density_filter': False,  # Still too risky
        'use_color_filter': False,
        'method': 'hybrid',
        'poisson_depth': 11,
        'use_planes': True,
        'aggressive_outlier': True,
        'voxel_size': None,
    },
}


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


def run_pipeline_improved(
    input_splat,
    output_mesh,
    preset='balanced',
    keep_intermediate=False,
    intermediate_dir=None,
    verbose=True,
    **override_params
):
    """
    Run the improved splat-to-mesh pipeline.
    
    Args:
        input_splat: Path to input Gaussian Splat PLY file
        output_mesh: Path to output mesh file
        preset: Quality preset ('fast', 'balanced', 'quality', 'ultra')
        keep_intermediate: Keep intermediate point cloud file
        intermediate_dir: Directory for intermediate files
        verbose: Print progress information
        **override_params: Override any preset parameters
    
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
    
    # Get preset parameters and apply overrides
    if preset not in PRESETS:
        print(f"[WARNING] Unknown preset '{preset}', using 'balanced'")
        preset = 'balanced'
    
    params = PRESETS[preset].copy()
    params.update({k: v for k, v in override_params.items() if v is not None})
    
    # Determine intermediate file location
    if intermediate_dir:
        intermediate_path = Path(intermediate_dir)
    else:
        intermediate_path = output_path.parent
    
    intermediate_path.mkdir(parents=True, exist_ok=True)
    pointcloud_path = intermediate_path / f"{input_path.stem}_pointcloud_improved.ply"
    
    if verbose:
        log_header("IMPROVED SPLAT TO MESH PIPELINE")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("  INPUT/OUTPUT:")
        print(f"    Input file:  {input_splat}")
        print(f"    Input size:  {format_size(input_path)}")
        print(f"    Output file: {output_mesh}")
        print()
        print(f"  PRESET: {preset.upper()}")
        print()
        print("  PARAMETERS:")
        print(f"    Point Extraction:")
        print(f"      Opacity threshold:    {params.get('opacity_threshold')}")
        print(f"      Scale filter:         {params.get('use_scale_filter')}")
        print(f"      Density filter:       {params.get('use_density_filter')}")
        print(f"      Color filter:         {params.get('use_color_filter')}")
        print(f"    Mesh Generation:")
        print(f"      Method:               {params.get('method')}")
        print(f"      Poisson depth:        {params.get('poisson_depth')}")
        print(f"      Plane detection:      {params.get('use_planes')}")
        print(f"      Aggressive outliers:  {params.get('aggressive_outlier')}")
        print("=" * 70)
    
    # Stage 1: Extract point cloud with improved filtering
    if verbose:
        log_header("STAGE 1: Improved Point Cloud Extraction")
    
    stage1_start = time.time()
    num_points = splat_to_pointcloud_improved(
        str(input_path),
        str(pointcloud_path),
        opacity_threshold=params.get('opacity_threshold', 0.5),
        use_scale_filter=params.get('use_scale_filter', True),
        scale_percentile_low=params.get('scale_percentile_low', 5),
        scale_percentile_high=params.get('scale_percentile_high', 95),
        use_density_filter=params.get('use_density_filter', True),
        density_std_ratio=params.get('density_std_ratio', 2.0),
        use_color_filter=params.get('use_color_filter', True),
        verbose=verbose
    )
    stage1_time = time.time() - stage1_start
    
    if num_points == 0:
        print("[ERROR] No points extracted from splat file")
        print("[HINT] Try lowering opacity threshold or relaxing filters")
        return None
    
    if verbose:
        print()
        print(f"  [STAGE 1 COMPLETE] Extracted {num_points:,} points in {format_time(stage1_time)}")
        if pointcloud_path.exists():
            print(f"  [STAGE 1 OUTPUT] Point cloud size: {format_size(pointcloud_path)}")
    
    # Stage 2: Convert point cloud to mesh with improved methods
    if verbose:
        log_header("STAGE 2: Improved Mesh Generation")
    
    stage2_start = time.time()
    result = pointcloud_to_mesh_improved(
        str(pointcloud_path),
        str(output_path),
        method=params.get('method', 'hybrid'),
        poisson_depth=params.get('poisson_depth', 10),
        use_planes=params.get('use_planes', True),
        aggressive_outlier=params.get('aggressive_outlier', True),
        verbose=verbose
    )
    stage2_time = time.time() - stage2_start
    
    if verbose and result:
        print()
        print(f"  [STAGE 2 COMPLETE] Generated mesh in {format_time(stage2_time)}")
        if output_path.exists():
            print(f"  [STAGE 2 OUTPUT] Mesh size: {format_size(output_path)}")
    
    # Clean up intermediate file
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
            print(f"  PRESET:     {preset.upper()}")
            print(f"  OUTPUT:     {output_mesh}")
            if output_path.exists():
                print(f"  FILE SIZE:  {format_size(output_path)}")
            print()
            print(f"  TIMING:")
            print(f"    Stage 1 (extraction): {format_time(stage1_time)}")
            print(f"    Stage 2 (meshing):    {format_time(stage2_time)}")
            print(f"    Total:                {format_time(pipeline_time)}")
            print()
            print("  IMPROVEMENTS APPLIED:")
            print("    - Multi-stage outlier removal")
            if params.get('use_scale_filter'):
                print("    - Gaussian scale filtering (removes floaters)")
            if params.get('use_density_filter'):
                print("    - Spatial density filtering (removes isolated points)")
            if params.get('use_planes'):
                print("    - RANSAC plane detection (sharp walls/floors)")
            if params.get('method') == 'hybrid':
                print("    - Hybrid meshing (planes + Poisson)")
            print()
            print("  NEXT STEPS:")
            print("    1. Import the mesh into Unity")
            print("    2. Compare with previous pipeline results")
            print("    3. Adjust preset or parameters as needed")
        else:
            print()
            print(f"  STATUS: FAILED")
            print()
            print("  TROUBLESHOOTING:")
            print("    - Try a different preset (--preset fast)")
            print("    - Relax filtering with --opacity 0.2")
            print("    - Use --keep-intermediate to inspect point cloud")
        print()
        print("=" * 70)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Improved Gaussian Splat to Unity mesh converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Presets:
  interior     : Tuned for building interiors (default) - your best settings
  fast         : Quick processing, depth 9
  balanced     : Good balance, depth 10
  quality      : High quality, depth 11
  ultra        : Same as quality (conservative)
  experimental : Tries new features (plane detection, filtering) - USE WITH CAUTION

Examples:
  # Interior building (recommended for your use case)
  python run_pipeline_improved.py model.ply mesh.obj --preset interior

  # Same as your best settings: depth=11, opacity=0.4, no downsampling
  python run_pipeline_improved.py model.ply mesh.obj

  # Try plane detection for sharper walls (experimental)
  python run_pipeline_improved.py model.ply mesh.obj --use-planes

  # Custom opacity
  python run_pipeline_improved.py model.ply mesh.obj --opacity 0.3

  # Lower depth for faster processing
  python run_pipeline_improved.py model.ply mesh.obj --depth 9

Note: The experimental filters (scale, density) are DISABLED by default
because they can remove valid geometry in large interior spaces.
        """
    )
    
    parser.add_argument("input", help="Input Gaussian Splat PLY file")
    parser.add_argument("output", help="Output mesh file (OBJ or PLY)")
    
    # Preset
    parser.add_argument(
        "--preset", "-p",
        choices=['fast', 'balanced', 'quality', 'ultra', 'interior', 'experimental'],
        default='interior',
        help="Quality preset (default: interior - tuned for building interiors)"
    )
    
    # Override options
    override_group = parser.add_argument_group("Override Options (modify preset)")
    override_group.add_argument(
        "--opacity", "-o",
        type=float,
        help="Override opacity threshold"
    )
    override_group.add_argument(
        "--depth", "-d",
        type=int,
        help="Override Poisson depth"
    )
    override_group.add_argument(
        "--method", "-m",
        choices=['hybrid', 'poisson', 'ball_pivoting'],
        help="Override meshing method"
    )
    override_group.add_argument(
        "--use-planes",
        action="store_true",
        help="Enable RANSAC plane detection (for sharper walls)"
    )
    override_group.add_argument(
        "--no-planes",
        action="store_true",
        help="Disable plane detection"
    )
    override_group.add_argument(
        "--use-scale-filter",
        action="store_true",
        help="Enable scale filtering (experimental)"
    )
    override_group.add_argument(
        "--no-scale-filter",
        action="store_true",
        help="Disable scale filtering"
    )
    override_group.add_argument(
        "--use-density-filter",
        action="store_true",
        help="Enable density filtering (experimental, can remove valid points)"
    )
    override_group.add_argument(
        "--no-density-filter",
        action="store_true",
        help="Disable density filtering"
    )
    override_group.add_argument(
        "--aggressive-outlier",
        action="store_true",
        help="Enable aggressive multi-stage outlier removal"
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
        help="Directory for intermediate files"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    # Build override params
    overrides = {}
    if args.opacity is not None:
        overrides['opacity_threshold'] = args.opacity
    if args.depth is not None:
        overrides['poisson_depth'] = args.depth
    if args.method is not None:
        overrides['method'] = args.method
    # Plane detection
    if args.use_planes:
        overrides['use_planes'] = True
    if args.no_planes:
        overrides['use_planes'] = False
    # Scale filter
    if args.use_scale_filter:
        overrides['use_scale_filter'] = True
    if args.no_scale_filter:
        overrides['use_scale_filter'] = False
    # Density filter
    if args.use_density_filter:
        overrides['use_density_filter'] = True
    if args.no_density_filter:
        overrides['use_density_filter'] = False
    # Aggressive outlier
    if args.aggressive_outlier:
        overrides['aggressive_outlier'] = True
    
    result = run_pipeline_improved(
        args.input,
        args.output,
        preset=args.preset,
        keep_intermediate=args.keep_intermediate,
        intermediate_dir=args.intermediate_dir,
        verbose=not args.quiet,
        **overrides
    )
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
