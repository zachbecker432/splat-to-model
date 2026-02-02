# Splat to Mesh Pipeline

Convert Gaussian Splat PLY files (from Postshot/Jawset) to Unity-ready 3D meshes using Ball Pivoting Algorithm (BPA).

```
Gaussian Splat (.ply) --> Point Cloud --> Mesh (.obj) --> Unity
```

## Prerequisites

- Docker Desktop

## Quick Start

1. Place your `.ply` file in the `data/` folder
2. Edit `docker-compose.yml` to set your input/output filenames
3. Run:

```powershell
docker compose up --build
```

## Configuration

All settings are configured via environment variables in `docker-compose.yml`:

```yaml
services:
  splat-to-mesh:
    environment:
      # Required
      - INPUT_FILE=model.ply      # Your input file in data/
      - OUTPUT_FILE=mesh.obj      # Output filename in data/
      
      # Point extraction
      - OPACITY_THRESHOLD=0.3     # 0-1, lower = more points
      # - MAX_SCALE=0.05          # Filter large Gaussians
      
      # Mesh generation  
      # - VOXEL_SIZE=0            # 0 = no downsampling
      # - TARGET_TRIANGLES=50000  # Simplify mesh
      - OUTLIER_STD_RATIO=2.0     # Lower = more aggressive cleanup
      - THIN_GEOMETRY=true        # Better for thin features
      
      # Mesh fixes
      - FLIP_NORMALS=false        # Fix inside-out mesh
      - DOUBLE_SIDED=false        # Visible from both sides
      - SMOOTH_FINAL=false        # Apply smoothing
      # - SMOOTH_ITERATIONS=5
      
      # Output
      - KEEP_INTERMEDIATE=false   # Keep point cloud file
      - VERBOSE=true
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INPUT_FILE` | Input Gaussian Splat PLY file (required) | - |
| `OUTPUT_FILE` | Output mesh file (required) | - |
| `OPACITY_THRESHOLD` | Minimum opacity for points (0-1). Lower = more points | 0.3 |
| `MAX_SCALE` | Maximum Gaussian scale to include | - |
| `VOXEL_SIZE` | Voxel size for downsampling (0 = disabled) | auto |
| `TARGET_TRIANGLES` | Target triangle count for simplification | - |
| `OUTLIER_STD_RATIO` | Outlier removal aggressiveness (lower = more) | 2.0 |
| `THIN_GEOMETRY` | Add extra radii for thin features | true |
| `FLIP_NORMALS` | Flip mesh normals if inside-out | false |
| `DOUBLE_SIDED` | Make mesh visible from both sides | false |
| `SMOOTH_FINAL` | Apply Taubin smoothing | false |
| `SMOOTH_ITERATIONS` | Number of smoothing iterations | 5 |
| `KEEP_INTERMEDIATE` | Keep intermediate point cloud | false |
| `VERBOSE` | Print progress information | true |

## Quality Presets

Edit the environment variables in `docker-compose.yml`:

**Low Quality (Fast)**
```yaml
- OPACITY_THRESHOLD=0.5
- TARGET_TRIANGLES=20000
```

**Medium Quality**
```yaml
- OPACITY_THRESHOLD=0.3
- TARGET_TRIANGLES=50000
```

**High Quality**
```yaml
- OPACITY_THRESHOLD=0.2
- VOXEL_SIZE=0
```

**Ultra Quality**
```yaml
- OPACITY_THRESHOLD=0.1
- VOXEL_SIZE=0
```

## Example Usage

```powershell
# Build and run with default settings
docker compose up --build

# Run with custom env vars (one-off)
docker compose run -e INPUT_FILE=scan.ply -e OUTPUT_FILE=result.obj splat-to-mesh

# Or directly with docker run
docker run -v ./data:/data \
  -e INPUT_FILE=model.ply \
  -e OUTPUT_FILE=mesh.obj \
  -e OPACITY_THRESHOLD=0.2 \
  -e DOUBLE_SIDED=true \
  splat-to-mesh:latest
```

## Platform-Specific Triangle Counts

| Platform | Recommended `TARGET_TRIANGLES` |
|----------|-------------------------------|
| Mobile | 10,000 - 50,000 |
| Desktop | 50,000 - 200,000 |
| High-end Desktop | 200,000+ |

## Unity Import

1. Drag the `.obj` file into your Unity project
2. Create a material with a vertex color shader:
   - URP/HDRP: Create Shader Graph with Vertex Color node
   - Built-in: Use `Particles/Standard Unlit` shader

See `VIDEO_TO_UNITY_QUICKSTART.md` for detailed shader setup.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Empty mesh output | Too few points extracted | Lower `OPACITY_THRESHOLD` (try 0.1) |
| Mesh has holes | Sparse point coverage | Lower `OPACITY_THRESHOLD` |
| See-through faces | Inconsistent normals | Set `DOUBLE_SIDED=true` |
| Mesh inside-out | Wrong normal direction | Set `FLIP_NORMALS=true` |
| Too many polygons | No simplification | Set `TARGET_TRIANGLES=50000` |
| No colors on mesh | Color transfer failed | Check input PLY has color data |

## File Structure

```
project/
    data/
        model.ply              # Input: Gaussian Splat from Postshot
        mesh.obj               # Output: Unity-ready mesh
    
    # Pipeline scripts
    run_pipeline.py            # Main pipeline (reads env vars)
    splat_to_pointcloud.py     # Stage 1: Extract points
    pointcloud_to_mesh.py      # Stage 2: Generate mesh (BPA)
    
    # Docker files
    Dockerfile
    docker-compose.yml
    requirements.txt
```

## Full Workflow

1. **Capture video** of your object (30-90 seconds, orbit around it)
2. **Process in Postshot** (Jawset) to create Gaussian Splat
3. **Export as PLY** from Postshot
4. **Place PLY in `data/` folder**
5. **Edit `docker-compose.yml`** with your filenames
6. **Run `docker compose up --build`**
7. **Import mesh to Unity** and apply vertex color material

See `VIDEO_TO_UNITY_QUICKSTART.md` for detailed instructions.
