# Launch vendor-neutral Direct3D 12 experiments without changing the upstream installation.
# Install into an isolated directory first:
#   .\Setup.cmd -InstallDir "$env:LOCALAPPDATA\Victory RTX"
# Then run:
#   .\Start-RTX.ps1
#   .\Start-RTX.ps1 -Mode StaticBVH
[CmdletBinding()]
param(
    [ValidateSet('Baseline', 'StaticBVH')]
    [string] $Mode = 'Baseline',
    [ValidateSet('Original', 'Residency', 'Throughput', 'P0')]
    [string] $MemoryProfile = 'Residency',
    [switch] $StrictWave,
    [string] $InstallDir = (Join-Path $env:LOCALAPPDATA 'Victory RTX')
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $InstallDir 'victory-play.exe'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw 'Install Victory RTX first using Setup.cmd -InstallDir with the same directory.'
}
$rtFlag = if ($Mode -eq 'StaticBVH') { '--rt-static-bvh' } else { '--no-rt' }
$settings = @{
    VICTORY_FSR_4K = '0'
    VICTORY_AMBIENT_FALLBACK = '1'
    VICTORY_ADAPTIVE_MEMORY = $(if ($MemoryProfile -eq 'Original') { '0' } else { '1' })
}
$launchArgs = @($rtFlag, '--console', 'series-x')
if ($MemoryProfile -in @('Throughput', 'P0')) {
    $settings['VICTORY_FILL_WORKERS'] = '4'
    $settings['VICTORY_PARALLEL_PRELUDES'] = '1'
    $launchArgs += '--local-gpu-memory'
}
if ($MemoryProfile -eq 'P0') {
    # Capture changed bytes with their originating submission where supported;
    # upstream falls back to its ordered readback path if capture is unavailable.
    $settings['VICTORY_RELAY'] = '2'
    $settings['VICTORY_CAPTURE_READBACKS'] = '1'
}
if ($StrictWave) {
    # Never knowingly use the approximate two independent Wave32 halves.
    # Unsupported shaders may be refused: keep this as a diagnostic A/B mode.
    $settings['VICTORY_WAVE_LOCAL'] = 'idiom'
    $settings['VICTORY_WAVE_WIDE'] = '0'
    $settings['VICTORY_WAVE_WHY'] = '1'
}
$previous = @{}
foreach ($key in $settings.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
try {
    foreach ($key in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    Write-Host "Victory RTX experimental profile: $Mode ($rtFlag), memory: $MemoryProfile. FSR disabled."
    Write-Host "Runtime: $launcher"
    Write-Host "Logs: $(Join-Path $InstallDir 'logs')"
    Push-Location -LiteralPath $InstallDir
    try {
        & $launcher @launchArgs
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
exit $code
