<#
.SYNOPSIS
    Run the splat-to-mesh pipeline in Docker

.DESCRIPTION
    Converts a Gaussian Splat PLY file to a Unity-ready mesh using Docker.
    Automatically handles volume mounting and container execution.

.PARAMETER InputFile
    Path to the input Gaussian Splat PLY file

.PARAMETER OutputFile
    Path for the output mesh file (OBJ recommended)

.PARAMETER Depth
    Poisson octree depth (6-11, default: 9)

.PARAMETER Opacity
    Minimum opacity threshold (0-1, default: 0.3)

.PARAMETER DensityThreshold
    Percentile of low-density vertices to remove (0-1, default: 0.01)

.PARAMETER OutlierStd
    Standard deviation ratio for outlier removal - lower is more aggressive (default: 2.0)

.PARAMETER Simplify
    Target number of triangles (optional)

.PARAMETER Build
    Force rebuild the Docker image

.PARAMETER Inspect
    Just inspect the PLY file structure without converting (useful for debugging)

.EXAMPLE
    .\docker-run.ps1 -InputFile .\model.ply -OutputFile .\mesh.obj

.EXAMPLE
    .\docker-run.ps1 -InputFile .\model.ply -Inspect

.EXAMPLE
    .\docker-run.ps1 -InputFile .\model.ply -OutputFile .\mesh.obj -Depth 10 -Opacity 0.2

.EXAMPLE
    .\docker-run.ps1 -InputFile .\model.ply -OutputFile .\mesh.obj -Opacity 0.5 -DensityThreshold 0.05

.EXAMPLE
    .\docker-run.ps1 -InputFile .\model.ply -OutputFile .\mesh.obj -Simplify 50000
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputFile,
    
    [int]$Depth = 9,
    
    [double]$Opacity = 0.3,
    
    [double]$DensityThreshold = 0.01,
    
    [double]$OutlierStd = 2.0,
    
    [int]$Simplify = 0,
    
    [switch]$Build,
    
    [switch]$KeepIntermediate,
    
    [switch]$Inspect
)

$ErrorActionPreference = "Stop"
$ImageName = "splat-to-mesh:latest"

# Get absolute paths
$InputPath = Resolve-Path $InputFile -ErrorAction Stop
$InputDir = Split-Path $InputPath -Parent
$InputName = Split-Path $InputPath -Leaf

# For output, create directory if it doesn't exist
$OutputDir = Split-Path $OutputFile -Parent
if ($OutputDir -eq "") {
    $OutputDir = "."
}
$OutputDir = (New-Item -ItemType Directory -Force -Path $OutputDir).FullName
$OutputName = Split-Path $OutputFile -Leaf

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SPLAT TO MESH PIPELINE (Docker)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Input:  $InputPath"
Write-Host "Output: $OutputDir\$OutputName"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Build image if requested or if it doesn't exist
$imageExists = docker images -q $ImageName 2>$null
if ($Build -or -not $imageExists) {
    Write-Host "Building Docker image..." -ForegroundColor Yellow
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    docker build -t $ImageName $scriptDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "Docker image built successfully" -ForegroundColor Green
    Write-Host ""
}

# Handle inspect mode separately
if ($Inspect) {
    Write-Host "Inspecting PLY file structure..." -ForegroundColor Yellow
    $dockerArgs = @(
        "run", "--rm",
        "-v", "${InputDir}:/data/input:ro",
        "--entrypoint", "python",
        $ImageName,
        "/app/splat_to_pointcloud.py",
        "/data/input/$InputName",
        "--inspect"
    )
    & docker $dockerArgs
    exit $LASTEXITCODE
}

# Build command arguments
$dockerArgs = @(
    "run", "--rm",
    "-v", "${InputDir}:/data/input:ro",
    "-v", "${OutputDir}:/data/output",
    $ImageName,
    "/data/input/$InputName",
    "/data/output/$OutputName",
    "--depth", $Depth,
    "--opacity", $Opacity,
    "--density-threshold", $DensityThreshold,
    "--outlier-std", $OutlierStd
)

if ($Simplify -gt 0) {
    $dockerArgs += "--simplify"
    $dockerArgs += $Simplify
}

if ($KeepIntermediate) {
    $dockerArgs += "--keep-intermediate"
    $dockerArgs += "--intermediate-dir"
    $dockerArgs += "/data/output"
}

# Run the container
Write-Host "Running pipeline..." -ForegroundColor Yellow
Write-Host "docker $($dockerArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& docker $dockerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "SUCCESS! Output saved to: $OutputDir\$OutputName" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "PIPELINE FAILED" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    exit 1
}
