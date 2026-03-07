# ============================================================
# Deep Stochastic Optimisation – Parameter Sweep Script
# ============================================================

$exe = ".\build\exp_barrier_hedging.exe"

# Fixed parameters
$train_paths = 250000
$eval_paths  = 1000000
$batch_size  = 250000
$maturity    = 1.0
$time_steps  = 252
$lr          = 0.075
$iters       = 150
$strike      = 100.0
$barrier     = 80.0

# Swept parameters
$hedge_freqs = @(252)
$variance_modes = @("instant", "learned")
$risk_measures = @("cvar", "mse")

# Hidden architectures
$hidden_configs = @(
    "",            # Linear
    "16",
    "16 16"
)

# Output root
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root_output = "logs/sweep_$timestamp"
New-Item -ItemType Directory -Force -Path $root_output | Out-Null

Write-Host "Starting sweep..."
Write-Host "Logs stored in $root_output"
Write-Host ""

# ============================================================
# Sweep
# ============================================================

foreach ($freq in $hedge_freqs) {
foreach ($variance in $variance_modes) {
foreach ($risk in $risk_measures) {
foreach ($hidden in $hidden_configs) {

    # Build experiment name
    $hidden_tag = if ($hidden -eq "") { "linear" } else { $hidden.Replace(" ", "_") }
    $run_name = "freq_${freq}_var_${variance}_risk_${risk}_hidden_${hidden_tag}"

    $run_dir = "$root_output/$run_name"
    New-Item -ItemType Directory -Force -Path $run_dir | Out-Null

    Write-Host "--------------------------------------------------"
    Write-Host "Running: $run_name"
    Write-Host "--------------------------------------------------"

    # Construct argument list
    $cliargs = @(
        "--train-paths", $train_paths,
        "--eval-paths",  $eval_paths,
        "--batch-size",  $batch_size,
        "--maturity",    $maturity,
        "--time-steps",   $time_steps,
        "--hedge-freq",   $freq,
        "--variance-mode", $variance,
        "--risk",         $risk,
        "--output",       $run_dir,
        "--device",       "cuda",
        "--lr",           $lr,
        "--iters",        $iters
        "--strike",       $strike
        "--barrier",      $barrier
    )

    # Add hidden only if not linear
    if ($hidden -ne "") {
        $cliargs += "--hidden"
        $cliargs += $hidden
    }

    # Run experiment
    & $exe $cliargs

    Write-Host "Finished: $run_name"
    Write-Host ""
}}}}
Write-Host "Sweep completed."