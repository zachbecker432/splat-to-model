"""
Splat to Point Cloud Converter

Extracts a standard point cloud (XYZ + RGB) from a Gaussian Splat PLY file
exported from Postshot/3DGS tools.

Usage:
    python splat_to_pointcloud.py input.ply output.ply [--opacity 0.3]
"""

import argparse
import numpy as np
import time
from plyfile import PlyData, PlyElement


def log_step(msg, indent=2):
    """Print a log message with consistent formatting."""
    prefix = " " * indent
    print(f"{prefix}[INFO] {msg}")


def sigmoid(x):
    """Sigmoid activation function."""
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sh_to_rgb(sh_dc):
    """
    Convert Spherical Harmonics DC component to RGB.
    The DC component in 3DGS represents the base color.
    """
    # SH to color conversion: C = SH_DC * SH_C0 + 0.5
    # where SH_C0 = 0.28209479177387814
    SH_C0 = 0.28209479177387814
    color = sh_dc * SH_C0 + 0.5
    return np.clip(color * 255, 0, 255).astype(np.uint8)


def find_field(names, candidates):
    """Find the first matching field name from a list of candidates."""
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def extract_opacity(vertex, verbose=True):
    """
    Extract opacity values from vertex data.
    Auto-detects whether values are logits (need sigmoid) or direct 0-1 values.
    
    Returns:
        opacity array (0-1 range), or None if no opacity field found
    """
    names = vertex.data.dtype.names
    
    # Try various opacity field names
    opacity_field = find_field(names, ['opacity', 'alpha', 'a', 'opacity_raw'])
    
    if opacity_field is None:
        return None
    
    opacity_raw = np.array(vertex[opacity_field])
    
    if verbose:
        print(f"  Found opacity field: '{opacity_field}'")
        print(f"  Raw opacity range: {opacity_raw.min():.3f} to {opacity_raw.max():.3f}")
    
    # Auto-detect if values need sigmoid transformation
    # If values are mostly outside 0-1 range, they're likely logits
    if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
        if verbose:
            print("  Detected logit format, applying sigmoid...")
        opacity = sigmoid(opacity_raw)
    else:
        # Values already in ~0-1 range
        if verbose:
            print("  Detected direct opacity values (0-1 range)")
        opacity = np.clip(opacity_raw, 0, 1)
    
    return opacity


def extract_colors(vertex, verbose=True):
    """
    Extract RGB colors from vertex data.
    Handles multiple naming conventions used by different 3DGS implementations.
    """
    names = vertex.data.dtype.names
    
    if verbose:
        print("  Looking for color fields...")
    
    # Try SH DC coefficients (standard 3DGS format)
    if all(f'f_dc_{i}' in names for i in range(3)):
        if verbose:
            print("  Found SH DC format (f_dc_0/1/2)")
        r = sh_to_rgb(np.array(vertex['f_dc_0']))
        g = sh_to_rgb(np.array(vertex['f_dc_1']))
        b = sh_to_rgb(np.array(vertex['f_dc_2']))
        return r, g, b
    
    # Try direct RGB uint8 (common in Postshot exports)
    if all(c in names for c in ['red', 'green', 'blue']):
        if verbose:
            print("  Found direct RGB format (red/green/blue)")
        r = np.array(vertex['red'])
        g = np.array(vertex['green'])
        b = np.array(vertex['blue'])
        
        # Check the data type and range
        if verbose:
            print(f"  Color dtype: {r.dtype}, range: {r.min()}-{r.max()}")
        
        # Handle different color formats
        if r.dtype == np.uint8:
            return r, g, b
        elif r.max() <= 1.0:
            # Float 0-1 range
            r = np.clip(r * 255, 0, 255).astype(np.uint8)
            g = np.clip(g * 255, 0, 255).astype(np.uint8)
            b = np.clip(b * 255, 0, 255).astype(np.uint8)
        else:
            # Assume 0-255 range stored as float
            r = np.clip(r, 0, 255).astype(np.uint8)
            g = np.clip(g, 0, 255).astype(np.uint8)
            b = np.clip(b, 0, 255).astype(np.uint8)
        return r, g, b
    
    # Try r/g/b shorthand
    if all(c in names for c in ['r', 'g', 'b']):
        if verbose:
            print("  Found RGB format (r/g/b)")
        r = np.array(vertex['r'])
        g = np.array(vertex['g'])
        b = np.array(vertex['b'])
        if r.dtype != np.uint8:
            if r.max() <= 1.0:
                r = np.clip(r * 255, 0, 255).astype(np.uint8)
                g = np.clip(g * 255, 0, 255).astype(np.uint8)
                b = np.clip(b * 255, 0, 255).astype(np.uint8)
            else:
                r = np.clip(r, 0, 255).astype(np.uint8)
                g = np.clip(g, 0, 255).astype(np.uint8)
                b = np.clip(b, 0, 255).astype(np.uint8)
        return r, g, b
    
    # Try alternative SH naming
    if all(f'sh_dc_{i}' in names for i in range(3)):
        if verbose:
            print("  Found SH DC format (sh_dc_0/1/2)")
        r = sh_to_rgb(np.array(vertex['sh_dc_0']))
        g = sh_to_rgb(np.array(vertex['sh_dc_1']))
        b = sh_to_rgb(np.array(vertex['sh_dc_2']))
        return r, g, b
    
    # Try diffuse color naming (some tools use this)
    if all(f'diffuse_{c}' in names for c in ['red', 'green', 'blue']):
        if verbose:
            print("  Found diffuse color format")
        r = np.array(vertex['diffuse_red'])
        g = np.array(vertex['diffuse_green'])
        b = np.array(vertex['diffuse_blue'])
        if r.dtype != np.uint8:
            r = np.clip(r * 255 if r.max() <= 1.0 else r, 0, 255).astype(np.uint8)
            g = np.clip(g * 255 if g.max() <= 1.0 else g, 0, 255).astype(np.uint8)
            b = np.clip(b * 255 if b.max() <= 1.0 else b, 0, 255).astype(np.uint8)
        return r, g, b
    
    # Fallback: white
    print("  WARNING: Could not find color data, using white")
    print(f"  Available fields: {names}")
    n = len(vertex)
    return (
        np.ones(n, dtype=np.uint8) * 255,
        np.ones(n, dtype=np.uint8) * 255,
        np.ones(n, dtype=np.uint8) * 255
    )


def splat_to_pointcloud(input_path, output_path, opacity_threshold=0.5, verbose=True):
    """
    Extract point cloud from Gaussian Splat PLY file.
    
    Args:
        input_path: Path to input .ply splat file
        output_path: Path to output .ply point cloud file
        opacity_threshold: Minimum opacity to include point (0-1)
        verbose: Print progress information
    
    Returns:
        Number of points in output point cloud
    """
    start_time = time.time()
    
    if verbose:
        log_step(f"Reading PLY file: {input_path}")
    
    read_start = time.time()
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']
    read_time = time.time() - read_start
    
    if verbose:
        log_step(f"File read complete ({read_time:.2f}s)")
        print()
        print("  " + "-" * 50)
        print("  INPUT FILE ANALYSIS")
        print("  " + "-" * 50)
        print(f"    Total gaussians/points: {len(vertex):,}")
        print(f"    Number of properties:   {len(vertex.data.dtype.names)}")
        print(f"    Properties: {list(vertex.data.dtype.names)}")
        print()
    
    # Extract positions
    names = vertex.data.dtype.names
    
    # Find position fields (handle different naming conventions)
    x_field = find_field(names, ['x', 'px', 'pos_x', 'position_x'])
    y_field = find_field(names, ['y', 'py', 'pos_y', 'position_y'])
    z_field = find_field(names, ['z', 'pz', 'pos_z', 'position_z'])
    
    if not all([x_field, y_field, z_field]):
        print(f"  [ERROR] Could not find position fields (x, y, z)")
        print(f"  [ERROR] Available fields: {names}")
        return 0
    
    x = np.array(vertex[x_field])
    y = np.array(vertex[y_field])
    z = np.array(vertex[z_field])
    
    if verbose:
        print("  " + "-" * 50)
        print("  POSITION DATA")
        print("  " + "-" * 50)
        print(f"    X range: [{x.min():.4f}, {x.max():.4f}] (span: {x.max()-x.min():.4f})")
        print(f"    Y range: [{y.min():.4f}, {y.max():.4f}] (span: {y.max()-y.min():.4f})")
        print(f"    Z range: [{z.min():.4f}, {z.max():.4f}] (span: {z.max()-z.min():.4f})")
        
        # Calculate approximate scale
        scale = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())
        print(f"    Approximate scale: {scale:.4f} units")
        print()
    
    # Extract and apply opacity filter
    if verbose:
        print("  " + "-" * 50)
        print("  OPACITY FILTERING")
        print("  " + "-" * 50)
    
    opacity = extract_opacity(vertex, verbose=verbose)
    
    if opacity is not None:
        mask = opacity > opacity_threshold
        points_kept = np.sum(mask)
        points_removed = len(opacity) - points_kept
        pct_kept = (points_kept / len(opacity)) * 100
        
        if verbose:
            print(f"    Opacity range (normalized): {opacity.min():.4f} to {opacity.max():.4f}")
            print(f"    Opacity threshold: {opacity_threshold}")
            print(f"    Points passing filter: {points_kept:,} / {len(opacity):,} ({pct_kept:.1f}%)")
            print(f"    Points filtered out: {points_removed:,}")
            
            # Histogram of opacity values
            hist_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, _ = np.histogram(opacity, bins=hist_bins)
            print()
            print("    Opacity distribution:")
            for i in range(len(hist)):
                bar_len = int(hist[i] / len(opacity) * 40)
                pct = hist[i] / len(opacity) * 100
                marker = " <-- threshold" if hist_bins[i] <= opacity_threshold < hist_bins[i+1] else ""
                print(f"      {hist_bins[i]:.1f}-{hist_bins[i+1]:.1f}: {'#' * bar_len} ({pct:.1f}%){marker}")
        print()
    else:
        if verbose:
            print("    [WARNING] No opacity data found, keeping all points")
        mask = np.ones(len(x), dtype=bool)
    
    # Extract colors
    if verbose:
        print("  " + "-" * 50)
        print("  COLOR EXTRACTION")
        print("  " + "-" * 50)
    
    r, g, b = extract_colors(vertex, verbose=verbose)
    
    if verbose:
        print(f"    Red range:   [{r.min()}, {r.max()}]")
        print(f"    Green range: [{g.min()}, {g.max()}]")
        print(f"    Blue range:  [{b.min()}, {b.max()}]")
        
        # Check if colors look valid
        if r.max() == r.min() == 255 and g.max() == g.min() == 255 and b.max() == b.min() == 255:
            print("    [WARNING] All colors are white - color extraction may have failed")
        print()
    
    # Apply mask
    x, y, z = x[mask], y[mask], z[mask]
    r, g, b = r[mask], g[mask], b[mask]
    
    if len(x) == 0:
        print("  [ERROR] No points remaining after filtering!")
        print("  [HINT] Try lowering the --opacity threshold (e.g., --opacity 0.1)")
        return 0
    
    # Create output PLY with standard point cloud format
    if verbose:
        print("  " + "-" * 50)
        print("  WRITING OUTPUT")
        print("  " + "-" * 50)
    
    write_start = time.time()
    vertices = np.array(
        list(zip(x, y, z, r, g, b)),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_path)
    write_time = time.time() - write_start
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"    Output file: {output_path}")
        print(f"    Total points written: {len(vertices):,}")
        print(f"    Write time: {write_time:.2f}s")
        print(f"    Total extraction time: {total_time:.2f}s")
        print()
    
    return len(vertices)


def inspect_ply(input_path):
    """
    Inspect a PLY file and print detailed information about its structure.
    Useful for debugging field name issues.
    """
    print(f"Inspecting PLY file: {input_path}")
    print("=" * 60)
    
    plydata = PlyData.read(input_path)
    
    print(f"Elements in file: {[el.name for el in plydata.elements]}")
    print()
    
    for element in plydata.elements:
        print(f"Element: '{element.name}' ({len(element)} entries)")
        print("-" * 40)
        
        for prop in element.properties:
            # Get sample values
            data = np.array(element[prop.name])
            print(f"  {prop.name:20s} dtype={str(data.dtype):10s} range=[{data.min():.4f}, {data.max():.4f}]")
        print()
    
    # Provide recommendations based on what we found
    vertex = plydata['vertex']
    names = vertex.data.dtype.names
    
    print("=" * 60)
    print("ANALYSIS:")
    print("-" * 40)
    
    # Check for position
    if all(f in names for f in ['x', 'y', 'z']):
        print("[OK] Position fields found (x, y, z)")
    else:
        print("[!!] Position fields NOT found - file may not be a valid point cloud")
    
    # Check for opacity
    opacity_field = find_field(names, ['opacity', 'alpha', 'a'])
    if opacity_field:
        opacity_raw = np.array(vertex[opacity_field])
        if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
            print(f"[OK] Opacity field '{opacity_field}' found (logit format, needs sigmoid)")
        else:
            print(f"[OK] Opacity field '{opacity_field}' found (direct 0-1 format)")
    else:
        print("[!!] No opacity field found - all points will be kept")
    
    # Check for colors
    if all(f'f_dc_{i}' in names for i in range(3)):
        print("[OK] Color fields found (SH DC format: f_dc_0/1/2)")
    elif all(c in names for c in ['red', 'green', 'blue']):
        print("[OK] Color fields found (RGB format: red/green/blue)")
    elif all(c in names for c in ['r', 'g', 'b']):
        print("[OK] Color fields found (RGB format: r/g/b)")
    else:
        print("[!!] No standard color fields found - mesh will be white")
        print(f"     Available fields: {list(names)}")
    
    # Check for scale (indicates it's a Gaussian Splat, not just point cloud)
    if any('scale' in n for n in names):
        print("[INFO] Scale fields found - this appears to be a Gaussian Splat file")
    else:
        print("[INFO] No scale fields - this may be a regular point cloud (not Gaussian Splat)")
    
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract point cloud from Gaussian Splat PLY file",
        epilog="""
Examples:
  # Inspect a PLY file to see its structure
  python splat_to_pointcloud.py model.ply --inspect
  
  # Basic extraction
  python splat_to_pointcloud.py model.ply output.ply
  
  # With higher opacity filter (less noise)
  python splat_to_pointcloud.py model.ply output.ply --opacity 0.5
        """
    )
    parser.add_argument("input", help="Input .ply splat file")
    parser.add_argument("output", nargs='?', default=None, help="Output .ply point cloud file")
    parser.add_argument(
        "--opacity", "-o",
        type=float,
        default=0.5,
        help="Minimum opacity threshold (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--inspect", "-i",
        action="store_true",
        help="Just inspect the PLY file structure without converting"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_ply(args.input)
    else:
        if args.output is None:
            print("ERROR: Output file required (unless using --inspect)")
            parser.print_help()
            return
        
        splat_to_pointcloud(
            args.input,
            args.output,
            opacity_threshold=args.opacity,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
