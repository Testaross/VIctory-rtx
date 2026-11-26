# Victory RTX: one default rendering path, no profile switches.
[CmdletBinding()]
param(
    [string] $InstallDir = (Join-Path $env:LOCALAPPDATA 'Victory RTX')
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $InstallDir 'victory-play.exe'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw 'Install Victory RTX using Setup.cmd first.'
}
# Native display path and full guest ray tracing; no synthetic ambient fill.
# Keep synchronization enabled and let the runtime query GPU capabilities.
$settings = @{
    VICTORY_FSR_4K = '0'
    VICTORY_AMBIENT_FALLBACK = '0'
    VICTORY_ADAPTIVE_MEMORY = '1'
    VICTORY_FILL_WORKERS = '4'
    VICTORY_PARALLEL_PRELUDES = '1'
    VICTORY_RELAY = '2'
    VICTORY_CAPTURE_READBACKS = '1'
    VICTORY_GROUP_SYNC = '2'
    VICTORY_F16_NEAREST = '1'
}
$previous = @{}
foreach ($key in $settings.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
try {
    foreach ($key in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    Write-Host 'Victory RTX: strict shader translation, full experimental ray tracing, P0 memory.'
    Write-Host "Logs: $(Join-Path $InstallDir 'logs')"
    Push-Location -LiteralPath $InstallDir
    try {
        & $launcher --rt --console series-x --local-gpu-memory
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
