# Splat to Mesh Pipeline

Convert Gaussian Splat PLY files (from Postshot/Jawset) to Unity-ready 3D meshes.

```
Gaussian Splat (.ply) --> Point Cloud --> Mesh (.obj) --> Unity
```

## Prerequisites

**Option A: Docker (Recommended)**
- Docker Desktop

**Option B: Local Python**
- Python 3.8+

## Quick Start

### Using Docker

```powershell
# Build the image (first time only)
docker build -t splat-to-mesh:latest .

# Run the pipeline
.\docker-run.ps1 -InputFile .\model.ply -OutputFile .\mesh.obj
```

Or with the batch file:

```cmd
docker-run.bat model.ply mesh.obj
```

### Using Python Directly

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py model.ply mesh.obj
```

## Usage

### Basic Command

```powershell
# Docker
.\docker-run.ps1 -InputFile <input.ply> -OutputFile <output.obj>

# Python
python run_pipeline.py <input.ply> <output.obj>
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--opacity` / `-o` | Minimum opacity threshold (0-1). Lower = more points but noisier | 0.3 |
| `--depth` / `-d` | Poisson reconstruction depth (6-11). Higher = more detail | 9 |
| `--simplify` / `-s` | Target triangle count. Reduces mesh complexity | None |
| `--density-threshold` / `-t` | Remove low-density vertices (0-1) | 0.01 |
| `--keep-intermediate` / `-k` | Keep the intermediate point cloud file | False |

### Quality Presets

**Low Quality (Fast)**
```powershell
python run_pipeline.py model.ply mesh.obj --depth 7 --opacity 0.5 --simplify 20000
```

**Medium Quality**
```powershell
python run_pipeline.py model.ply mesh.obj --depth 8 --opacity 0.3 --simplify 50000
```

**High Quality**
```powershell
python run_pipeline.py model.ply mesh.obj --depth 9 --opacity 0.2
```

**Ultra Quality**
```powershell
python run_pipeline.py model.ply mesh.obj --depth 10 --opacity 0.1
```

### Platform-Specific Triangle Counts

| Platform | Recommended `--simplify` |
|----------|-------------------------|
| Mobile | 10,000 - 50,000 |
| Desktop | 50,000 - 200,000 |
| High-end Desktop | 200,000+ |

## Examples

```powershell
# Basic conversion
python run_pipeline.py splats/model.ply meshes/output.obj

# High quality with more points extracted
python run_pipeline.py model.ply mesh.obj --depth 10 --opacity 0.2

# Mobile-optimized mesh
python run_pipeline.py model.ply mesh.obj --simplify 30000

# Keep point cloud for inspection
python run_pipeline.py model.ply mesh.obj --keep-intermediate

# Docker with options
.\docker-run.ps1 -InputFile .\model.ply -OutputFile .\mesh.obj -Depth 10 -Opacity 0.2 -Simplify 50000
```

## Running Steps Separately

You can run the two pipeline stages independently:

```powershell
# Stage 1: Extract point cloud from Gaussian Splat
python splat_to_pointcloud.py model.ply pointcloud.ply --opacity 0.3

# Stage 2: Convert point cloud to mesh
python pointcloud_to_mesh.py pointcloud.ply mesh.obj --depth 9
```

This is useful when you want to:
- Inspect the intermediate point cloud
- Try different mesh settings without re-extracting points
- Debug issues at specific stages

## Unity Import

1. Drag the `.obj` file into your Unity project
2. Create a material with a vertex color shader:
   - URP/HDRP: Create Shader Graph with Vertex Color node
   - Built-in: Use `Particles/Standard Unlit` shader

A simple vertex color shader is included in `VIDEO_TO_UNITY_QUICKSTART.md`.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Empty mesh output | Too few points extracted | Lower `--opacity` (try 0.1-0.2) |
| Mesh has holes | Sparse point coverage | Increase `--depth`, lower `--opacity` |
| Mesh is blobby/smooth | Not enough detail | Increase `--depth` to 10-11 |
| Too many polygons | High depth setting | Use `--simplify` to reduce |
| No colors on mesh | Color transfer failed | Check input PLY has color data |
| Docker build fails | Missing dependencies | Ensure Docker Desktop is running |
| "No module named open3d" | Dependencies not installed | Run `pip install -r requirements.txt` |

### Adjusting Parameters

**If mesh has too many artifacts:**
- Increase `--opacity` (0.4-0.6)
- Increase `--density-threshold` (0.02-0.05)

**If mesh is missing detail:**
- Decrease `--opacity` (0.1-0.2)
- Increase `--depth` (10-11)

**If processing is too slow:**
- Use `--voxel-size 0.01` to downsample point cloud
- Lower `--depth` (7-8)

## File Structure

```
project/
    model.ply              # Input: Gaussian Splat from Postshot
    mesh.obj               # Output: Unity-ready mesh
    
    # Pipeline scripts
    run_pipeline.py        # Main pipeline script
    splat_to_pointcloud.py # Stage 1: Extract points
    pointcloud_to_mesh.py  # Stage 2: Generate mesh
    
    # Docker files
    Dockerfile
    docker-compose.yml
    docker-run.ps1
    docker-run.bat
```

## Full Workflow

1. **Capture video** of your object (30-90 seconds, orbit around it)
2. **Process in Postshot** (Jawset) to create Gaussian Splat
3. **Export as PLY** from Postshot
4. **Run this pipeline** to convert to mesh
5. **Import to Unity** and apply vertex color material

See `VIDEO_TO_UNITY_QUICKSTART.md` for detailed instructions on each step.
