"""
Splat to Point Cloud Converter

Extracts a standard point cloud (XYZ + RGB) from a Gaussian Splat PLY file
exported from Postshot/3DGS tools.

Usage:
    python splat_to_pointcloud.py input.ply output.ply [--opacity 0.3]
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


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


def extract_colors(vertex):
    """
    Extract RGB colors from vertex data.
    Handles multiple naming conventions used by different 3DGS implementations.
    """
    names = vertex.data.dtype.names
    
    # Try SH DC coefficients (standard 3DGS format)
    if all(f'f_dc_{i}' in names for i in range(3)):
        r = sh_to_rgb(np.array(vertex['f_dc_0']))
        g = sh_to_rgb(np.array(vertex['f_dc_1']))
        b = sh_to_rgb(np.array(vertex['f_dc_2']))
        return r, g, b
    
    # Try direct RGB (some exporters use this)
    if all(c in names for c in ['red', 'green', 'blue']):
        r = np.array(vertex['red'])
        g = np.array(vertex['green'])
        b = np.array(vertex['blue'])
        # If already uint8, use directly; otherwise scale
        if r.dtype != np.uint8:
            r = np.clip(r * 255 if r.max() <= 1.0 else r, 0, 255).astype(np.uint8)
            g = np.clip(g * 255 if g.max() <= 1.0 else g, 0, 255).astype(np.uint8)
            b = np.clip(b * 255 if b.max() <= 1.0 else b, 0, 255).astype(np.uint8)
        return r, g, b
    
    # Try alternative SH naming
    if all(f'sh_dc_{i}' in names for i in range(3)):
        r = sh_to_rgb(np.array(vertex['sh_dc_0']))
        g = sh_to_rgb(np.array(vertex['sh_dc_1']))
        b = sh_to_rgb(np.array(vertex['sh_dc_2']))
        return r, g, b
    
    # Fallback: white
    print("Warning: Could not find color data, using white")
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
    if verbose:
        print(f"Reading splat from: {input_path}")
    
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']
    
    if verbose:
        print(f"Total gaussians: {len(vertex)}")
        print(f"Available properties: {vertex.data.dtype.names}")
    
    # Extract positions
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    
    # Extract and apply opacity filter
    names = vertex.data.dtype.names
    if 'opacity' in names:
        opacity_raw = np.array(vertex['opacity'])
        opacity = sigmoid(opacity_raw)
        mask = opacity > opacity_threshold
        if verbose:
            print(f"Opacity range (after sigmoid): {opacity.min():.3f} - {opacity.max():.3f}")
            print(f"Points after opacity filter (>{opacity_threshold}): {np.sum(mask)}")
    else:
        if verbose:
            print("No opacity data found, keeping all points")
        mask = np.ones(len(x), dtype=bool)
    
    # Extract colors
    r, g, b = extract_colors(vertex)
    
    # Apply mask
    x, y, z = x[mask], y[mask], z[mask]
    r, g, b = r[mask], g[mask], b[mask]
    
    # Create output PLY with standard point cloud format
    vertices = np.array(
        list(zip(x, y, z, r, g, b)),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_path)
    
    if verbose:
        print(f"Saved point cloud to: {output_path}")
        print(f"Total points: {len(vertices)}")
    
    return len(vertices)


def main():
    parser = argparse.ArgumentParser(
        description="Extract point cloud from Gaussian Splat PLY file"
    )
    parser.add_argument("input", help="Input .ply splat file")
    parser.add_argument("output", help="Output .ply point cloud file")
    parser.add_argument(
        "--opacity", "-o",
        type=float,
        default=0.5,
        help="Minimum opacity threshold (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    splat_to_pointcloud(
        args.input,
        args.output,
        opacity_threshold=args.opacity,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
