# Mesh Quality Improvements Guide

This document outlines techniques to improve mesh quality from Gaussian Splat to mesh conversion, specifically addressing "bubbly" surfaces and non-flat walls.

---

## Table of Contents

1. [Point Cloud Preprocessing](#1-point-cloud-preprocessing)
2. [Alternative Reconstruction Algorithms](#2-alternative-reconstruction-algorithms)
3. [Post-Processing / Mesh Refinement](#3-post-processing--mesh-refinement)
4. [Hybrid Approaches](#4-hybrid-approaches)
5. [Parameter Tuning](#5-parameter-tuning)
6. [Tool Reference](#6-tool-reference)

---

## 1. Point Cloud Preprocessing

### 1.1 Improved Normal Estimation

The default normal estimation uses simple KD-tree neighbor search. For architectural scenes with flat surfaces, more sophisticated approaches help.

#### Plane-Aware Normal Estimation

- **Principal Component Analysis (PCA)** with larger neighborhoods for flat regions
- **Region growing** to detect planar patches and enforce consistent normals
- **RANSAC-based plane fitting** to identify planar regions before normal estimation

#### Parameters to Tune

| Parameter | Current | Recommended | Effect |
|-----------|---------|-------------|--------|
| `max_nn` | 30 | 50-100 | More neighbors = smoother normals on flat surfaces |
| `search_radius` | auto (~1% of min dimension) | 2-5% of min dimension | Larger radius = more averaging |

#### Code Example (Open3D)

```python
# Increase neighbors for smoother normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius * 2,  # Double the radius
        max_nn=100  # Increase from 30
    )
)

# Use camera location for consistent orientation (if available)
pcd.orient_normals_towards_camera_location(camera_location)
```

---

### 1.2 Point Cloud Denoising

Add explicit denoising before reconstruction to reduce noise that causes bumpy surfaces.

#### Available Algorithms

| Algorithm | Library | Best For | Preserves Edges |
|-----------|---------|----------|-----------------|
| **Statistical Outlier Removal** | Open3D | Isolated noise points | N/A |
| **Radius Outlier Removal** | Open3D | Sparse outliers | N/A |
| **Bilateral Filtering** | Open3D (custom) | General noise | Yes |
| **Moving Least Squares (MLS)** | PCL, CGAL | Surface smoothing | Somewhat |
| **Jet Smoothing** | CGAL | Mathematical smoothing | No |
| **WLOP (Weighted Locally Optimal Projection)** | CGAL | Heavy noise/outliers | Somewhat |

#### Code Example - Statistical Outlier Removal

```python
# Already in pipeline, but can be more aggressive
pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=50,      # Increase from 30
    std_ratio=1.5         # Decrease from 2.0 (more aggressive)
)
```

#### Code Example - Radius Outlier Removal

```python
# Remove points with few neighbors in radius
pcd, ind = pcd.remove_radius_outlier(
    nb_points=16,         # Minimum neighbors required
    radius=0.05           # Search radius
)
```

---

### 1.3 Plane Detection and Segmentation

Detect planar regions explicitly before reconstruction. This prevents flat walls from becoming bumpy.

#### RANSAC Plane Segmentation

```python
import open3d as o3d

def segment_planes(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000, min_points=100):
    """
    Segment point cloud into planar and non-planar regions.
    
    Returns:
        planes: List of (plane_model, plane_points) tuples
        remaining: Points that don't belong to any plane
    """
    planes = []
    remaining = pcd
    
    while len(remaining.points) > min_points:
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers) < min_points:
            break
            
        plane_cloud = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
        
        planes.append((plane_model, plane_cloud))
    
    return planes, remaining
```

---

## 2. Alternative Reconstruction Algorithms

Poisson reconstruction excels at organic shapes but struggles with architectural/planar geometry.

### 2.1 Algorithm Comparison

| Algorithm | Best For | Sharp Edges | Watertight | Speed | Library |
|-----------|----------|-------------|------------|-------|---------|
| **Poisson** | Organic shapes | Poor | Yes | Medium | Open3D |
| **Screened Poisson** | General purpose | Moderate | Yes | Medium | Open3D |
| **Ball Pivoting (BPA)** | Uniform density | Good | No | Fast | Open3D |
| **Alpha Shapes** | Concave objects | Good | No | Fast | Open3D |
| **Marching Cubes** | Volumetric data | Moderate | Yes | Fast | scikit-image |
| **Delaunay 3D** | Convex shapes | Moderate | Yes | Fast | scipy |

### 2.2 Ball Pivoting Algorithm (BPA)

Better at preserving sharp features than Poisson. Works best with uniform point density.

```python
# Compute necessary normals first
pcd.estimate_normals()

# Calculate appropriate radii based on point spacing
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radii = [avg_dist * 0.5, avg_dist, avg_dist * 2, avg_dist * 4]

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector(radii)
)
```

### 2.3 Alpha Shapes

Good for concave objects while preserving sharp edges.

```python
# Alpha value controls level of detail (smaller = more detail)
alpha = 0.03
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
```

### 2.4 Screened Poisson (Improved Parameters)

```python
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=11,           # Increase from 9 for more detail
    width=0,
    scale=1.1,
    linear_fit=True     # Better fit to actual points
)
```

### 2.5 Neural Reconstruction Methods

For best quality (requires additional setup):

| Method | Description | Quality | Speed | Setup Complexity |
|--------|-------------|---------|-------|------------------|
| **NeuS / NeuS2** | Neural implicit surface | Excellent | Slow | High |
| **NeuralAngelo** | High-fidelity neural surfaces | Excellent | Very Slow | High |
| **Gaussian Opacity Fields** | Direct from Gaussians | Very Good | Medium | Medium |
| **2DGS** | 2D Gaussian Splatting | Very Good | Medium | Medium |

---

## 3. Post-Processing / Mesh Refinement

### 3.1 Mesh Smoothing

**Warning:** Smoothing reduces detail. Apply selectively or use feature-preserving methods.

#### Smoothing Algorithms

| Algorithm | Shrinkage | Feature Preservation | Library |
|-----------|-----------|---------------------|---------|
| **Laplacian** | Yes | Poor | Open3D, MeshLab |
| **Taubin** | No | Moderate | Open3D, MeshLab |
| **HC Laplacian** | Minimal | Moderate | MeshLab |
| **Bilateral** | No | Good | MeshLab, libigl |
| **L0 Smoothing** | No | Excellent | Research code |

#### Code Example - Taubin Smoothing

```python
# Taubin smoothing (volume-preserving)
mesh = mesh.filter_smooth_taubin(
    number_of_iterations=10,
    lambda_filter=0.5,
    mu=-0.53  # Slightly larger magnitude than lambda to prevent shrinkage
)
```

#### Code Example - Laplacian Smoothing

```python
# Simple Laplacian (will shrink mesh)
mesh = mesh.filter_smooth_laplacian(
    number_of_iterations=5,
    lambda_filter=0.5
)
```

---

### 3.2 Planar Region Detection and Flattening

Post-process the mesh to detect and flatten planar regions:

1. Segment mesh faces by normal similarity (region growing)
2. Fit planes to each region using least squares
3. Project vertices onto fitted planes
4. Blend at region boundaries

```python
def flatten_planar_regions(mesh, angle_threshold=5.0, min_faces=50):
    """
    Detect planar regions in mesh and flatten them.
    
    Args:
        mesh: Open3D TriangleMesh
        angle_threshold: Max angle (degrees) between face normals in same region
        min_faces: Minimum faces to consider a region
    
    Returns:
        Modified mesh with flattened planar regions
    """
    # Implementation would:
    # 1. Compute face normals
    # 2. Region grow based on normal similarity
    # 3. For each large enough region, fit a plane
    # 4. Project vertices to plane
    # 5. Handle boundary blending
    pass
```

---

### 3.3 Feature-Preserving Smoothing

| Algorithm | Description | Library |
|-----------|-------------|---------|
| **Bilateral Mesh Denoising** | Smooths while preserving edges | MeshLab, libigl |
| **Guided Normal Filtering** | Uses guide normals to preserve features | Research |
| **L0 Smoothing** | Piecewise constant regions | Research |
| **Rolling Guidance Filter** | Iterative edge-preserving | Research |

#### MeshLab Filters (via PyMeshLab)

```python
import pymeshlab

ms = pymeshlab.MeshSet()
ms.load_new_mesh("input.obj")

# Two-step bilateral smoothing
ms.apply_filter('meshing_surface_subdivision_midpoint')
ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving', 
                iterations=5)

ms.save_current_mesh("output.obj")
```

---

### 3.4 Remeshing

Clean up mesh topology for better quality.

| Algorithm | Result | Best For | Library |
|-----------|--------|----------|---------|
| **Isotropic Remeshing** | Uniform triangles | General cleanup | CGAL, PyMeshLab |
| **Quadric Decimation** | Reduced poly count | Optimization | Open3D |
| **Instant Meshes** | Clean quads/tris | Retopology | Standalone |
| **Subdivision** | More polygons | Adding detail | Open3D |

#### Code Example - Subdivision + Smoothing

```python
# Subdivide then smooth for gradual refinement
mesh = mesh.subdivide_midpoint(number_of_iterations=1)
mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
```

---

## 4. Hybrid Approaches

### 4.1 Segmented Reconstruction Pipeline (Recommended)

Process planar and organic regions separately for best results.

```
Pipeline:
1. Segment point cloud:
   - Planar regions (walls, floors, ceilings)
   - Organic regions (furniture, objects, vegetation)

2. Reconstruct each type differently:
   - Planar: Create clean quad/tri meshes from fitted planes
   - Organic: Use Poisson with high depth

3. Merge meshes with proper boundary stitching
```

#### Benefits

- Walls and floors become perfectly flat
- Organic details preserved with Poisson
- Can apply different texture strategies per region

---

### 4.2 Multi-Resolution Reconstruction

```
1. Create coarse mesh (depth=7-8) for overall shape
2. Create fine mesh (depth=10-11) for details
3. Use coarse mesh normals to guide/regularize fine mesh
```

---

### 4.3 Gaussian-Aware Reconstruction

Leverage Gaussian Splat properties for better reconstruction:

| Gaussian Property | How to Use |
|-------------------|------------|
| **Scale (covariance)** | Weight points by inverse scale (smaller = more reliable) |
| **Orientation** | Initialize normals from Gaussian orientation |
| **Opacity** | Filter low-opacity Gaussians (already doing this) |
| **Spherical Harmonics** | Could extract view-dependent appearance |

```python
def filter_by_gaussian_scale(vertex, max_scale=0.1):
    """
    Filter out Gaussians that are too large (background/noise).
    """
    # Extract scale values (log-space in most implementations)
    scale_0 = np.exp(vertex['scale_0'])
    scale_1 = np.exp(vertex['scale_1'])
    scale_2 = np.exp(vertex['scale_2'])
    
    # Compute average scale
    avg_scale = (scale_0 + scale_1 + scale_2) / 3
    
    # Keep only reasonably-sized Gaussians
    mask = avg_scale < max_scale
    return mask
```

---

## 5. Parameter Tuning

### 5.1 splat_to_pointcloud.py

| Parameter | Default | Try | Effect |
|-----------|---------|-----|--------|
| `--opacity` | 0.5 | 0.7-0.9 | Keep only confident points |
| (new) `--max-scale` | None | 0.05-0.1 | Filter large/blobby Gaussians |

### 5.2 pointcloud_to_mesh.py

| Parameter | Default | Try | Effect |
|-----------|---------|-----|--------|
| `--depth` | 9 | 10-11 | More geometric detail |
| `--voxel-size` | auto | 0 | Keep all points (no downsampling) |
| `--outlier-std` | 2.0 | 1.0-1.5 | More aggressive noise removal |
| `--density-threshold` | 0.01 | 0.05-0.1 | Remove more low-confidence areas |

### 5.3 Normal Estimation (internal)

| Parameter | Current | Try | Effect |
|-----------|---------|-----|--------|
| `max_nn` | 30 | 60-100 | Smoother normals |
| `search_radius` | 1% of min dim | 2-5% | Larger averaging area |
| `orient_k` | 15 | 30-50 | More consistent orientation |

---

## 6. Tool Reference

### Python Libraries

| Library | Install | Purpose |
|---------|---------|---------|
| **Open3D** | `pip install open3d` | Point cloud & mesh processing |
| **PyMeshLab** | `pip install pymeshlab` | MeshLab filters in Python |
| **trimesh** | `pip install trimesh` | Mesh manipulation |
| **CGAL** | `pip install cgal` | Advanced geometry algorithms |
| **libigl** | `pip install libigl` | Geometry processing |
| **scipy** | `pip install scipy` | Delaunay triangulation |
| **scikit-image** | `pip install scikit-image` | Marching cubes |

### Standalone Tools

| Tool | Purpose | Link |
|------|---------|------|
| **MeshLab** | Interactive mesh processing | meshlab.net |
| **Blender** | Full 3D suite | blender.org |
| **Instant Meshes** | Automatic retopology | github.com/wjakob/instant-meshes |
| **CloudCompare** | Point cloud processing | cloudcompare.org |

---

## Implementation Checklist

### Quick Wins (Already Tried)
- [x] Increase Poisson depth to 10-11
- [x] Disable voxel downsampling
- [x] Increase opacity threshold
- [x] Adjust outlier removal

### Medium Effort
- [ ] Add Taubin smoothing as post-processing
- [ ] Implement bilateral point cloud filtering
- [ ] Add RANSAC plane detection
- [ ] Try Ball Pivoting Algorithm
- [ ] Try Alpha Shapes

### High Effort (Best Results)
- [ ] Implement segmented reconstruction (planes vs. organic)
- [ ] Add Gaussian scale filtering
- [ ] Implement planar region flattening post-process
- [ ] Try neural methods (NeuS2, etc.)

---

## Troubleshooting

### Problem: Mesh has holes

**Solutions:**
- Decrease density threshold
- Use Poisson (creates watertight mesh) instead of BPA
- Increase point cloud density (disable downsampling)

### Problem: Walls are wavy/bubbly

**Solutions:**
- Implement plane segmentation and separate reconstruction
- Increase normal estimation neighbors (max_nn=100)
- Apply Taubin smoothing
- Use planar region flattening post-process

### Problem: Lost fine details

**Solutions:**
- Increase Poisson depth
- Use segmented reconstruction (high depth for organic, planes for flat)
- Reduce or eliminate smoothing
- Disable or reduce downsampling

### Problem: Noisy/spiky mesh

**Solutions:**
- More aggressive outlier removal (std_ratio=1.0)
- Add bilateral denoising to point cloud
- Apply light Taubin smoothing
- Increase opacity threshold in splat extraction

---

## Next Steps

1. Start with **segmented reconstruction** for architectural scenes
2. Add **Taubin smoothing** for organic regions
3. Experiment with **Ball Pivoting** as an alternative to Poisson
4. Consider **neural methods** for production-quality results
