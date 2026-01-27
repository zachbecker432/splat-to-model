"""
Splat to Mesh Pipeline Runner

Runs the complete pipeline from Gaussian Splat PLY to Unity-ready mesh.
Combines splat_to_pointcloud.py and pointcloud_to_mesh.py into a single command.

Usage:
    python run_pipeline.py input_splat.ply output_mesh.obj [options]
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Import the pipeline modules
from splat_to_pointcloud import splat_to_pointcloud
from pointcloud_to_mesh import pointcloud_to_mesh


def run_pipeline(
    input_splat,
    output_mesh,
    opacity_threshold=0.3,
    poisson_depth=9,
    density_threshold=0.01,
    voxel_size=None,
    target_triangles=None,
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
        keep_intermediate: Keep intermediate point cloud file
        intermediate_dir: Directory for intermediate files (default: same as output)
        verbose: Print progress information
    
    Returns:
        Path to output mesh if successful, None otherwise
    """
    input_path = Path(input_splat)
    output_path = Path(output_mesh)
    
    # Validate input
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_splat}")
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
        print("=" * 60)
        print("SPLAT TO MESH PIPELINE")
        print("=" * 60)
        print(f"Input:  {input_splat}")
        print(f"Output: {output_mesh}")
        print("=" * 60)
        print()
    
    # Stage 1: Extract point cloud from splat
    if verbose:
        print("STAGE 1: Extracting point cloud from Gaussian Splat")
        print("-" * 60)
    
    num_points = splat_to_pointcloud(
        str(input_path),
        str(pointcloud_path),
        opacity_threshold=opacity_threshold,
        verbose=verbose
    )
    
    if num_points == 0:
        print("ERROR: No points extracted from splat file")
        return None
    
    if verbose:
        print()
    
    # Stage 2: Convert point cloud to mesh
    if verbose:
        print("STAGE 2: Converting point cloud to mesh")
        print("-" * 60)
    
    result = pointcloud_to_mesh(
        str(pointcloud_path),
        str(output_path),
        poisson_depth=poisson_depth,
        density_threshold=density_threshold,
        voxel_size=voxel_size,
        target_triangles=target_triangles,
        verbose=verbose
    )
    
    # Clean up intermediate file unless requested to keep
    if not keep_intermediate and pointcloud_path.exists():
        pointcloud_path.unlink()
        if verbose:
            print(f"\nRemoved intermediate file: {pointcloud_path}")
    elif keep_intermediate and verbose:
        print(f"\nKept intermediate file: {pointcloud_path}")
    
    if verbose:
        print()
        print("=" * 60)
        if result:
            print("PIPELINE COMPLETE")
            print(f"Output mesh: {output_mesh}")
        else:
            print("PIPELINE FAILED")
        print("=" * 60)
    
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

  # Mobile-optimized (fewer triangles)
  python run_pipeline.py model.ply mesh.obj --simplify 50000

  # Keep intermediate point cloud for inspection
  python run_pipeline.py model.ply mesh.obj --keep-intermediate

Quality Presets:
  Low (fast):    --depth 7 --opacity 0.5 --simplify 20000
  Medium:        --depth 8 --opacity 0.3 --simplify 50000
  High:          --depth 9 --opacity 0.2
  Ultra:         --depth 10 --opacity 0.1
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
    
    # Mesh generation options
    mesh_group = parser.add_argument_group("Mesh Generation Options")
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
        help="Voxel size for point cloud downsampling (default: none)"
    )
    mesh_group.add_argument(
        "--simplify", "-s",
        type=int,
        default=None,
        help="Target number of triangles (default: no simplification)"
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
        poisson_depth=args.depth,
        density_threshold=args.density_threshold,
        voxel_size=args.voxel_size,
        target_triangles=args.simplify,
        keep_intermediate=args.keep_intermediate,
        intermediate_dir=args.intermediate_dir,
        verbose=not args.quiet
    )
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
