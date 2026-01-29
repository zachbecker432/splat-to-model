# Understanding Gaussian Splatting

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [What is a Gaussian?](#what-is-a-gaussian)
3. [From NeRFs to Gaussian Splatting](#from-nerfs-to-gaussian-splatting)
4. [How 3D Gaussian Splatting Works](#how-3d-gaussian-splatting-works)
5. [The Math (Simplified)](#the-math-simplified)
6. [The .ply/.splat File Format](#the-plysplat-file-format)
7. [Why Convert to Meshes?](#why-convert-to-meshes)
8. [How This Project Fits In](#how-this-project-fits-in)

---

## The Big Picture

**Gaussian Splatting** is a technique for representing 3D scenes using millions of tiny, fuzzy 3D ellipsoids (shaped like squashed spheres) instead of traditional triangles or voxels. 

Think of it like this:
- **Traditional 3D**: Build a model out of tiny triangles (like origami)
- **Voxels**: Build a model out of tiny cubes (like Minecraft)
- **Gaussian Splatting**: Build a model out of millions of tiny, fuzzy, colored blobs

The key insight is that these "blobs" (Gaussians) can be rendered extremely fast using a technique called "splatting" - projecting 3D blobs onto a 2D screen.

---

## What is a Gaussian?

A **Gaussian** (named after mathematician Carl Friedrich Gauss) is a mathematical function that describes a "bell curve" distribution.

### 1D Gaussian (Bell Curve)
```
         *
        * *
       *   *
      *     *
    *         *
  *             *
*                 *
```
The value is highest in the center and smoothly fades to zero as you move away.

### 2D Gaussian (Fuzzy Circle)
Imagine a soft, glowing dot - brightest in the center, fading smoothly at the edges.

### 3D Gaussian (Fuzzy Ellipsoid)
Now imagine that in 3D - a fuzzy, glowing blob. It can be:
- **Stretched** in any direction (ellipsoid, not just sphere)
- **Rotated** to any orientation
- **Colored** with RGB values
- **Transparent** (has an opacity value)

---

## From NeRFs to Gaussian Splatting

### The Problem: Novel View Synthesis

Given a set of photos of an object/scene taken from different angles, can we render what it would look like from a NEW angle we never photographed?

### Previous Solutions

#### 1. Traditional Photogrammetry
- Reconstruct a mesh from photos
- Texture map the mesh
- **Problem**: Struggles with reflections, transparency, fine details, hair, fur, etc.

#### 2. Neural Radiance Fields (NeRF) - 2020
- Train a neural network to encode the entire scene
- Query the network: "What color/density is at position (x,y,z) when viewed from direction (dx,dy,dz)?"
- Use ray marching to render images
- **Pros**: Amazing quality, handles complex materials
- **Cons**: SLOW to train (hours/days), SLOW to render (seconds per frame)

#### 3. 3D Gaussian Splatting (3DGS) - 2023
- Represent the scene as millions of 3D Gaussians
- Each Gaussian stores its own position, shape, color, opacity
- Render by "splatting" Gaussians onto the screen
- **Pros**: Fast training (minutes), REAL-TIME rendering (100+ FPS), excellent quality
- **Cons**: Large file sizes, not directly compatible with game engines

---

## How 3D Gaussian Splatting Works

### Training Phase (Creating a Splat from Photos)

```
Photos (100-500 images)
        |
        v
+-------------------+
| Structure from    |  --> Sparse point cloud
| Motion (SfM)      |      + camera positions
+-------------------+
        |
        v
+-------------------+
| Initialize        |  --> Each point becomes 
| Gaussians         |      a small Gaussian
+-------------------+
        |
        v
+-------------------+
| Optimization      |  --> Adjust Gaussians to
| Loop (thousands   |      match input photos
| of iterations)    |
+-------------------+
        |
        v
Final Gaussian Splat
(millions of Gaussians)
```

### What Gets Optimized?

For each Gaussian, the algorithm learns:

1. **Position** (x, y, z) - Where is the center?
2. **Covariance Matrix** (3x3) - Shape and orientation
   - This defines how "stretched" the Gaussian is and in what directions
   - Often stored as: 3 scale values + 4 quaternion rotation values
3. **Color** (RGB or Spherical Harmonics) - What color is it?
   - Spherical Harmonics allow view-dependent colors (shininess, reflections)
4. **Opacity** (alpha) - How transparent is it?

### Adaptive Density Control

During training, the algorithm also:
- **Splits** large Gaussians that are too big (under-reconstruction)
- **Clones** Gaussians in areas that need more detail
- **Prunes** Gaussians that are too transparent or too small

This allows the model to automatically add detail where needed.

### Rendering Phase (Displaying a Splat)

```
For each frame:
1. Project all Gaussians from 3D to 2D (camera projection)
2. Sort Gaussians by depth (back to front)
3. For each pixel, blend overlapping Gaussians
4. Output final image
```

The key insight: This can be done on the GPU using a custom CUDA renderer, achieving 100+ FPS.

---

## The Math (Simplified)

### The 3D Gaussian Function

A 3D Gaussian is defined as:

```
G(x) = exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))
```

Where:
- `x` = point in 3D space
- `mu` = center position of the Gaussian
- `Sigma` = 3x3 covariance matrix (defines shape)
- `Sigma^(-1)` = inverse of covariance matrix

### The Covariance Matrix

The covariance matrix `Sigma` can be decomposed as:

```
Sigma = R * S * S^T * R^T
```

Where:
- `R` = 3x3 rotation matrix (orientation)
- `S` = 3x3 diagonal scaling matrix (size in each axis)

This is why splat files store rotation (as quaternion) and scale separately.

### Projecting to 2D

When rendering, the 3D Gaussian is projected to a 2D Gaussian on screen:

```
Sigma' = J * W * Sigma * W^T * J^T
```

Where:
- `J` = Jacobian of the projective transformation
- `W` = viewing transformation matrix
- `Sigma'` = resulting 2D covariance (an ellipse on screen)

### Alpha Blending

Pixels are computed by blending overlapping Gaussians front-to-back:

```
C = Sum(c_i * alpha_i * T_i)
```

Where:
- `c_i` = color of Gaussian i
- `alpha_i` = opacity of Gaussian i
- `T_i` = transmittance (how much light gets through previous Gaussians)

---

## The .ply/.splat File Format

### What's Stored in a Gaussian Splat File?

Each Gaussian stores approximately 59 values:

| Property | Count | Description |
|----------|-------|-------------|
| Position | 3 | x, y, z coordinates |
| Scale | 3 | sx, sy, sz (log scale) |
| Rotation | 4 | Quaternion (w, x, y, z) |
| Opacity | 1 | Sigmoid-encoded alpha |
| SH (DC) | 3 | Base color (RGB as spherical harmonic DC component) |
| SH (rest) | 45 | Higher-order spherical harmonics for view-dependent color |

**Total: ~59 floats per Gaussian = ~236 bytes per Gaussian**

A typical scene has 1-5 million Gaussians = **200MB - 1GB+ file size**

### PLY Format Structure

```ply
ply
format binary_little_endian 1.0
element vertex 1500000
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
... (45 more f_rest properties for spherical harmonics)
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
<binary data>
```

---

## Why Convert to Meshes?

Gaussian Splats are amazing for rendering, but they have limitations:

### Problems with Splats in Game Engines

1. **No Native Support**
   - Unity/Unreal don't understand Gaussian splats
   - Requires custom renderers (not easy to integrate)

2. **No Physics Interaction**
   - Can't collide with a splat
   - Can't make splat objects fall, bounce, etc.

3. **No Standard Lighting**
   - Splats have baked-in lighting from training photos
   - They don't respond to new light sources in your game

4. **No Standard Shadows**
   - Splats don't cast shadows on other objects
   - They don't receive shadows from other objects

5. **Memory & Performance**
   - Millions of Gaussians = heavy on GPU memory
   - Sorting every frame = computational cost

### The Solution: Convert to Mesh

Converting to a traditional mesh gives you:
- Full physics support
- Standard rendering pipeline
- Unity/Unreal native support  
- LOD (Level of Detail) support
- Standard lighting and shadows
- Smaller file sizes (potentially)

**The trade-off**: You lose some visual fidelity, especially for fuzzy/transparent things.

---

## How This Project Fits In

### The Pipeline

```
Gaussian Splat (.ply)
        |
        | splat_to_pointcloud.py
        |   - Read Gaussian positions
        |   - Sample colors from Gaussians
        |   - Optional: Intelligently sample within each Gaussian
        v
Point Cloud (.ply)
        |
        | pointcloud_to_mesh.py
        |   - Estimate normals
        |   - Surface reconstruction (Poisson or Ball Pivoting)
        |   - Clean up mesh
        |   - Transfer colors
        v
Mesh (.ply / .obj)
        |
        | Unity PlyImporter
        v
Unity GameObject with Mesh + Vertex Colors
```

### What We're Doing at Each Step

#### Step 1: Splat to Point Cloud

The splat file contains Gaussians (fuzzy blobs), not points. We need to extract points:

1. **Simple approach**: Use Gaussian centers as points
   - Fast but loses detail
   
2. **Better approach**: Sample multiple points within each Gaussian
   - Sample based on Gaussian's size/shape
   - Keep the density proportional to opacity

#### Step 2: Point Cloud to Mesh

Turn the cloud of points into a solid surface:

1. **Estimate surface normals** - Which way does each point "face"?
2. **Surface reconstruction** - Connect points into triangles
   - Poisson: Smooth, watertight, but can over-smooth
   - Ball Pivoting: Preserves detail, but can have holes

#### Step 3: Import to Unity

Standard mesh import, but we preserve vertex colors from the point cloud.

---

## Key Takeaways

1. **Gaussian Splats are not traditional 3D models** - They're millions of fuzzy, colored blobs

2. **They're learned from photos** - An optimization process figures out the right blobs to recreate any view

3. **They render via "splatting"** - Project 3D blobs to 2D, blend them together

4. **Converting to mesh is lossy but necessary** - For game engines, physics, standard rendering

5. **The conversion extracts the "surface"** - We're essentially finding where all the opaque Gaussians are and building a mesh around them

---

## Visual Mental Model

Imagine you're trying to recreate a photograph of a fluffy cat using only:

**Traditional mesh**: Cut out tiny paper triangles and glue them together. Works okay for the body, but the fur looks like spiky polygons.

**Gaussian Splatting**: Use millions of tiny cotton balls of different colors, sizes, and shapes. Place them in 3D space. The fur looks fluffy and natural because cotton balls have soft edges that blend together.

**Converting splat to mesh**: Take all those cotton balls and wrap them in plastic wrap. Now you have a solid surface you can touch, but you've lost the fluffy cotton ball effect.

---

## Further Reading

- [Original 3DGS Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
- [NeRF Paper](https://www.matthewtancik.com/nerf) - Understanding what came before
- [Poisson Surface Reconstruction](https://hhoppe.com/poissonrecon.htm) - The meshing algorithm
- [Spherical Harmonics Explained](https://www.shadertoy.com/view/lsfXWH) - Interactive visualization

---

## Glossary

| Term | Definition |
|------|------------|
| **Gaussian** | A bell-curve distribution; in 3D, a fuzzy ellipsoid |
| **Splatting** | Rendering technique that projects 3D primitives onto 2D |
| **Covariance Matrix** | Defines the shape/orientation of a Gaussian |
| **Quaternion** | A 4-number representation of 3D rotation |
| **Spherical Harmonics** | Basis functions for encoding view-dependent color |
| **SfM** | Structure from Motion - extracting 3D from 2D photos |
| **NeRF** | Neural Radiance Field - predecessor to Gaussian Splatting |
| **Poisson Reconstruction** | Algorithm to create watertight mesh from points |
| **Alpha Blending** | Combining transparent layers based on opacity |
| **PLY** | Polygon File Format - stores points/meshes |
