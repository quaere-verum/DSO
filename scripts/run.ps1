$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot
$ExePath = ".\build\DSO.exe"

# --- Environment Setup ---
# We force these to 1 to prevent OpenBLAS/MKL from fighting with TBB
Write-Host "Configuring thread environment variables..." -ForegroundColor Cyan

$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"

# Optional: If you are using LibTorch with CUDA, sometimes this helps with 
# stability in nested TBB loops, though less relevant for BLAS errors.
$env:TORCH_NUM_THREADS = "1"

# --- Validation ---
if (-not (Test-Path $ExePath)) {
    Write-Error "Error: Could not find the executable at $ExePath"
    Pause
    exit
}

# --- Execution ---
Write-Host "Launching $ExePath..." -ForegroundColor Green
Write-Host "------------------------------------------"

# Use '&' (the call operator) to execute the program. 
# This ensures the script waits for the EXE to finish.
& $ExePath

# --- Cleanup/Wrap-up ---
Write-Host "------------------------------------------"
Write-Host "Execution completed. Press any key to close this window..."
$null = [System.Console]::ReadKey($true)