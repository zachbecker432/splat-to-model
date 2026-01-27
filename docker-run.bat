@echo off
REM Simple batch wrapper for the PowerShell script
REM Usage: docker-run.bat input.ply output.obj [options]
REM
REM For full options, use: powershell -ExecutionPolicy Bypass -File docker-run.ps1 -Help

setlocal

if "%~1"=="" (
    echo Usage: docker-run.bat input.ply output.obj [depth] [opacity] [simplify]
    echo.
    echo Arguments:
    echo   input.ply   - Path to input Gaussian Splat PLY file
    echo   output.obj  - Path for output mesh file
    echo   depth       - Poisson depth 6-11 ^(default: 9^)
    echo   opacity     - Opacity threshold 0-1 ^(default: 0.3^)
    echo   simplify    - Target triangles ^(default: none^)
    echo.
    echo Examples:
    echo   docker-run.bat model.ply mesh.obj
    echo   docker-run.bat model.ply mesh.obj 10 0.2
    echo   docker-run.bat model.ply mesh.obj 9 0.3 50000
    echo.
    echo For more options, use the PowerShell script directly:
    echo   powershell -ExecutionPolicy Bypass -File docker-run.ps1 -InputFile model.ply -OutputFile mesh.obj
    exit /b 1
)

set INPUT=%~1
set OUTPUT=%~2
set DEPTH=%~3
set OPACITY=%~4
set SIMPLIFY=%~5

if "%DEPTH%"=="" set DEPTH=9
if "%OPACITY%"=="" set OPACITY=0.3

if "%SIMPLIFY%"=="" (
    powershell -ExecutionPolicy Bypass -File "%~dp0docker-run.ps1" -InputFile "%INPUT%" -OutputFile "%OUTPUT%" -Depth %DEPTH% -Opacity %OPACITY%
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0docker-run.ps1" -InputFile "%INPUT%" -OutputFile "%OUTPUT%" -Depth %DEPTH% -Opacity %OPACITY% -Simplify %SIMPLIFY%
)
